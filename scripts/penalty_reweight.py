"""成本口径检验：主要结论在“缺供只按少收的零售收入计价”（惩罚 ×1，等价于按站点真实电价和零售价最大化经营利润）
以及“完全不计惩罚、只看电费和缺供电量”两种口径下是否成立。
- 小时模型：结果/bss_anchor_plans.parquet 中本就算了 ×1 档，直接取。
- 电池级仿真：各候选方案的成本 = 电费 + 终值 + m × 当月最高零售总价 × 未满足电量；把 m 从 3 换成 1 后重新做每日滚动调参。
输出：结果/penalty_reweight.txt"""
import ast
import pathlib
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(HERE.parent))
import data as D  # noqa: E402
from exp_packs_value import pick, rel_ci, START  # noqa: E402

OUT = ROOT / "results"
TOT = {m: float(D.price("A7", m, "total").max()) for m in range(1, 13)}
lines = []


def say(s):
    lines.append(s)
    print(s)


def tot_of(days):
    return pd.Series([TOT[d.month] for d in days], index=days)


def reweight(M, S, m_from, m_to):
    """M、S：日 × 方案 的成本、未满足电量（kWh）。"""
    return M - (m_from - m_to) * S.mul(tot_of(M.index), axis=0)


def roll(M, cols, fb, days):
    ks = [pick(M[cols], d, fb) for d in days]
    return ks, np.array([M.loc[d, k] for d, k in zip(days, ks)])


def take(M, ks, days):
    return np.array([M.loc[d, k] for d, k in zip(days, ks)])


def fmt(a, b):
    m, lo, hi = rel_ci(a, b)
    return f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


# ======================================================================= 1. 小时模型，×1 档
say("## 1 小时模型（A7 数据锚定网格），只取惩罚 ×1 档")
A = pd.read_parquet(OUT / "bss_anchor_plans.parquet")
par = A.cfg.apply(lambda c: ast.literal_eval(c) if isinstance(c, str) else tuple(c))
A["m"] = par.str[2]
for m in (1, 3, 10):
    g = A[A.m == m]
    piv = g.groupby(["cfg", "model"])[["reg_mid", "reg_default", "mae"]].sum().reset_index()
    best_diff, taus = [], []
    for c, h in piv.groupby("cfg"):
        best_diff.append(h.loc[h.mae.idxmin(), "model"] != h.loc[h.reg_mid.idxmin(), "model"])
        taus.append(kendalltau(h.mae, h.reg_mid)[0])
    d = g.pivot_table(index="day", columns="model", values="reg_mid", aggfunc="sum").sort_index()
    s = (f"  惩罚 ×{m}：中点计划下 MAE 最优≠遗憾最优的配置 {100 * np.mean(best_diff):.0f}%，Kendall τ 中位 {np.median(taus):.2f}；"
         f"LQR/LightGBM {fmt(d['lqr'], d['lgb_local'])}；TimesFM/LightGBM {fmt(d['timesfm3_cov'], d['lgb_local'])}；"
         f"中点 vs 默认 日遗憾合计 {100 * (g.reg_mid.sum() / g.reg_default.sum() - 1):+.1f}%")
    say(s)

# ======================================================================= 2. 电池级：上海，规则 / 调参管线 / 固定计划
say("\n## 2 电池级仿真（A7，上海电价）：×3 复现、×1 重算、不计惩罚时电费与缺供分开看")
RU = pd.read_parquet(OUT / "bl_rule.parquet")
FC = pd.read_parquet(OUT / "bl_fc.parquet")
SQ = pd.read_parquet(OUT / "bl_tariffs_sq.parquet")
for (c, pc), g in RU.groupby(["cfg", "pc"], sort=False):
    mf = ast.literal_eval(c)[2]
    if mf != 3:
        continue
    Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
    Ms = g.pivot(index="day", columns="th", values="short").sort_index()
    base = g.drop_duplicates("day").set_index("day")
    h = FC[(FC.cfg == c) & (FC.pc == pc)]
    Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
    Fs = h.pivot(index="day", columns="cand", values="short").sort_index()
    days = Fc.index[Fc.index >= START]
    lgb = [k for k in Fc.columns if not k.startswith("pi@")]
    k0 = Mc[Mc.index < pd.Timestamp("2024-04-01")].sum().idxmin()
    sq = SQ[(SQ.cfg == c) & (SQ.pc == pc) & (SQ.tariff == "shanghai")]
    out = [f"{c} {pc:.0f} kW："]
    for mt in (3, 1):
        Mr, Fr = reweight(Mc, Ms, mf, mt), reweight(Fc, Fs, mf, mt)
        kr, rule = roll(Mr, list(Mr.columns), k0, days)
        kf, fcr = roll(Fr, lgb, "mean@0|mid", days)
        seg = (f" ×{mt}：管线 {fmt(fcr, rule)}；最大 vs 默认 {fmt(Fr.loc[days, 'mean@0|max'], Fr.loc[days, 'mean@0|default'])}；"
               f"最大 vs 最小 {fmt(Fr.loc[days, 'mean@0|max'], Fr.loc[days, 'mean@0|min'])}")
        if len(sq):
            Qc = sq.pivot(index="day", columns="cand", values="cost").sort_index()
            Qs = sq.pivot(index="day", columns="cand", values="short").sort_index()
            Qr = reweight(Qc, Qs, mf, mt)
            kq, sqr = roll(Qr, list(Qr.columns), "mean@0|mid", days)
            seg += f"；近期曲线 vs 规则 {fmt(sqr, rule)}；LightGBM vs 近期曲线 {fmt(fcr, sqr)}"
        out.append(seg)
        if mt == 1:   # ×1 下选出的策略，电费与缺供分开
            dem = base.loc[days, "dem"].sum()
            e_rule = take(Mr, kr, days) - tot_of(days).to_numpy() * take(Ms, kr, days) * 1
            e_fc = take(Fr, kf, days) - tot_of(days).to_numpy() * take(Fs, kf, days) * 1
            e = lambda col: Fr.loc[days, col].to_numpy() - tot_of(days).to_numpy() * Fs.loc[days, col].to_numpy()
            out.append(f"  ×1 选出的策略，不计惩罚：电费（含终值）管线 vs 规则 {100 * (e_fc.sum() / e_rule.sum() - 1):+.1f}%，"
                       f"缺供占需求 规则 {100 * take(Ms, kr, days).sum() / dem:.2f}% / 管线 {100 * take(Fs, kf, days).sum() / dem:.2f}%；"
                       f"固定计划 电费 最大 vs 默认 {100 * (e('mean@0|max').sum() / e('mean@0|default').sum() - 1):+.1f}%，"
                       f"缺供 默认 {100 * Fs.loc[days, 'mean@0|default'].sum() / dem:.2f}% → 最大 {100 * Fs.loc[days, 'mean@0|max'].sum() / dem:.2f}%"
                       f"（最小 {100 * Fs.loc[days, 'mean@0|min'].sum() / dem:.2f}%）")
    say("".join(out[:1]) + "\n" + "\n".join(out[1:]))

# ======================================================================= 3. 其他电价：计划价值与预测价值（×1）
say("\n## 3 其他电价（6 h，2.5×，原 ×3）：×1 重算的计划价值与预测价值")
TR = pd.read_parquet(OUT / "bl_tariffs_rule.parquet")
TF = pd.read_parquet(OUT / "bl_tariffs_fc.parquet")
for (tf, c, pc), g in TR.groupby(["tariff", "cfg", "pc"], sort=False):
    Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
    Ms = g.pivot(index="day", columns="th", values="short").sort_index()
    h = TF[(TF.tariff == tf) & (TF.cfg == c) & (TF.pc == pc)]
    q = SQ[(SQ.tariff == tf) & (SQ.cfg == c) & (SQ.pc == pc)]
    Fc, Fs = (h.pivot(index="day", columns="cand", values=v).sort_index() for v in ("cost", "short"))
    Qc, Qs = (q.pivot(index="day", columns="cand", values=v).sort_index() for v in ("cost", "short"))
    days = Fc.index[Fc.index >= START]
    k0 = Mc[Mc.index < pd.Timestamp("2024-04-01")].sum().idxmin()
    lgb = [k for k in Fc.columns if not k.startswith("pi@")]
    segs = []
    for mt in (3, 1):
        Mr, Fr, Qr = reweight(Mc, Ms, 3, mt), reweight(Fc, Fs, 3, mt), reweight(Qc, Qs, 3, mt)
        _, rule = roll(Mr, list(Mr.columns), k0, days)
        _, fcr = roll(Fr, lgb, "mean@0|mid", days)
        _, sqr = roll(Qr, list(Qr.columns), "mean@0|mid", days)
        segs.append(f"×{mt}：近期曲线 vs 规则 {fmt(sqr, rule)}；LightGBM vs 近期曲线 {fmt(fcr, sqr)}")
    say(f"  {tf} {pc:.0f} kW：" + "；".join(segs))

# ======================================================================= 4. 电池级 9 模型：精度与成本排序（×1）
say("\n## 4 电池级仿真 9 个模型：MAE 与成本的 Kendall τ（原 ×3 与 ×1）")
FM = pd.read_parquet(OUT / "bl_fc_models.parquet")
MA = pd.read_parquet(OUT / "bl_models.parquet").groupby("model").mae.mean()
for (c, pc), g in FM.groupby(["cfg", "pc"]):
    segs = []
    for tie in ("default", "mid", "max"):
        col = f"mean@0|{tie}"
        h = g[(g.cand == col) & (g.day >= START)].copy()
        h["tot"] = h.day.dt.month.map(TOT)
        res = []
        for mt in (3, 1):
            cost = (h.cost - (3 - mt) * h.tot * h.short).groupby(h.model).sum()
            res.append(kendalltau(MA.loc[cost.index], cost)[0])
        segs.append(f"{tie} τ ×3 {res[0]:.2f} / ×1 {res[1]:.2f}")
    say(f"  {c} {pc:.0f} kW：" + "；".join(segs))

(OUT / "penalty_reweight.txt").write_text("\n".join(lines), encoding="utf-8")
