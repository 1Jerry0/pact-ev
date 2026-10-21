"""电池级仿真在利润口径下的汇总（exp_bl_profit.py 的输出）。
一、表 5 的利润口径版本：相对每日调参规则的成本（%），未调参计划（LightGBM 均值，最小/默认/中点/最大）、调参管线、近期曲线管线、完全信息；各自缺供率。
二、服务水平口径：缺供不超过需求 1% 时各策略族的最低电费（以完全信息计划为基准的超额 %），以及滚动选择版本。
三、9 个模型：MAE 与成本的 Kendall τ、TimesFM / LightGBM、最大缓冲 vs 默认。
四、其他电价下的计划价值与预测价值。
输出：结果/blp_summary.txt、结果/blp_tradeoff_points.parquet"""
import pathlib
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(HERE.parent))
from exp_packs_value import rel_ci  # noqa: E402

OUT = ROOT / "results"
EVAL0, TUNE0, TARGET = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01"), 0.01
lines = []


def say(s=""):
    lines.append(s)
    print(s)


def fmt(a, b):
    m, lo, hi = rel_ci(a, b)
    return f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}"


def pick(M, d, fb):
    past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
    return past.sum().idxmin() if len(past) >= 7 else fb


def pick_qos(E, S, dem, d, fb):
    sl = slice(d - pd.Timedelta(days=29), d - pd.Timedelta(days=2))
    e, s, q = E.loc[sl], S.loc[sl], dem.loc[sl]
    if len(e) < 7:
        return fb
    share = s.sum() / q.sum()
    ok = share[share <= TARGET]
    return e.sum()[ok.index].idxmin() if len(ok) else share.idxmin()


def roll(M, Sx, days, fb, cols=None):
    cols = list(M.columns) if cols is None else cols
    ks = [pick(M[cols], d, fb) for d in days]
    return np.array([M.loc[d, k] for d, k in zip(days, ks)]), np.array([Sx.loc[d, k] for d, k in zip(days, ks)])


def section_main(tariff_rule, tariff_fc, label):
    RU, FC = tariff_rule, tariff_fc
    for X in (RU, FC):
        X["cost"] = X.E + X.Pen
    pts = []
    for (tf, c, pc), g in RU.groupby(["tariff", "cfg", "pc"], sort=False):
        Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
        Ms = g.pivot(index="day", columns="th", values="S").sort_index()
        Me = g.pivot(index="day", columns="th", values="E").sort_index()
        dem = g.drop_duplicates("day").set_index("day").dem.sort_index()
        h = FC[(FC.tariff == tf) & (FC.cfg == c) & (FC.pc == pc)].copy()
        h["k"] = h.model + ":" + h.cand
        Fc, Fs, Fe = (h.pivot(index="day", columns="k", values=v).sort_index() for v in ("cost", "S", "E"))
        days = Fc.index[Fc.index >= TUNE0]
        rule_cols = [k for k in Mc.columns if k >= 0]
        k0 = Mc.loc[Mc.index < EVAL0, rule_cols].sum().idxmin()
        rule, rule_s = roll(Mc, Ms, days, k0, rule_cols)
        lg = [k for k in Fc.columns if k.startswith("lgb_local:")]
        sq = [k for k in Fc.columns if k.startswith("seasonal_q:")]
        lgb, lgb_s = roll(Fc, Fs, days, "lgb_local:mean@0|mid", lg)
        rec, rec_s = roll(Fc, Fs, days, "seasonal_q:mean@0|mid", sq)
        D = dem.loc[days].sum()
        s = f"[{tf}] {c} {pc:.0f} kW：规则 {rule.mean():,.0f} 元/日，缺供 {100 * rule_s.sum() / D:.2f}%；相对规则 %："
        if tf == "shanghai":
            for t in ("min", "default", "mid", "max"):
                col = f"lgb_local:mean@0|{t}"
                s += f" {t} {fmt(Fc.loc[days, col].to_numpy(), rule)}（缺供 {100 * Fs.loc[days, col].sum() / D:.2f}%）"
            s += f"；最大 vs 默认 {fmt(Fc.loc[days, 'lgb_local:mean@0|max'].to_numpy(), Fc.loc[days, 'lgb_local:mean@0|default'].to_numpy())}"
            s += f"；最大 vs 最小 {fmt(Fc.loc[days, 'lgb_local:mean@0|max'].to_numpy(), Fc.loc[days, 'lgb_local:mean@0|min'].to_numpy())}"
            s += f"；换下即充 {fmt(Mc.loc[days, -1].to_numpy(), rule)}（缺供 {100 * Ms.loc[days, -1].sum() / D:.2f}%）"
        if "pi:count@0|mid" in Fc:
            pi, pi_s = roll(Fc, Fs, days, "pi:count@0|mid", [k for k in Fc.columns if k.startswith("pi:")])
            s += f"；完全信息 {fmt(pi, rule)}"
        s += f"；事后最优阈值规则 {fmt(Mc.loc[days, rule_cols].min(axis=1).to_numpy(), rule)}"
        s += f"；调参管线 {fmt(lgb, rule)}（缺供 {100 * lgb_s.sum() / D:.2f}%）；近期曲线 vs 规则 {fmt(rec, rule)}；LightGBM vs 近期曲线 {fmt(lgb, rec)}"
        say(s)
        # 服务水平口径
        base_E = None
        if tf == "shanghai" and "pi:count@0|mid" in Fe:
            base_E = Fe.loc[days, [k for k in Fe.columns if k.startswith("pi:")]].sum().min()
        tot = lambda E_, S_: (E_.loc[days].sum(), S_.loc[days].sum() / D)
        Er, Sr = tot(Me[rule_cols], Ms[rule_cols])
        El, Sl = tot(Fe[lg], Fs[lg])
        Eq, Sq = tot(Fe[sq], Fs[sq])
        ref = Er[Sr <= TARGET].min() if (Sr <= TARGET).any() else np.nan

        def best(E_, S_):
            ok = S_ <= TARGET
            return (E_[ok].min() / ref - 1) * 100 if ok.any() and np.isfinite(ref) else np.nan
        s2 = f"    缺供 ≤ 1%：规则最低电费 {ref / len(days):,.0f} 元/日；LightGBM 策略族 {best(El, Sl):+.1f}%，近期曲线 {best(Eq, Sq):+.1f}%（相对规则）"
        for t in ("default", "mid", "max"):
            cols = [k for k in lg if k.endswith(f"|{t}")]
            s2 += f"；只用 {t} {best(El[cols], Sl[cols]):+.1f}%"
        qd = dem
        kr = [pick_qos(Me[rule_cols], Ms[rule_cols], qd, d, k0) for d in days]
        rE, rS = np.array([Me.loc[d, k] for d, k in zip(days, kr)]), np.array([Ms.loc[d, k] for d, k in zip(days, kr)])
        kl = [pick_qos(Fe[lg], Fs[lg], qd, d, "lgb_local:mean@0|mid") for d in days]
        lE, lS = np.array([Fe.loc[d, k] for d, k in zip(days, kl)]), np.array([Fs.loc[d, k] for d, k in zip(days, kl)])
        kq = [pick_qos(Fe[sq], Fs[sq], qd, d, "seasonal_q:mean@0|mid") for d in days]
        qE, qS = np.array([Fe.loc[d, k] for d, k in zip(days, kq)]), np.array([Fs.loc[d, k] for d, k in zip(days, kq)])
        s2 += (f"；滚动选择：规则 缺供 {100 * rS.sum() / D:.2f}%，近期曲线 缺供 {100 * qS.sum() / D:.2f}% 电费 vs 规则 {fmt(qE, rE)}，"
               f"LightGBM 缺供 {100 * lS.sum() / D:.2f}% 电费 vs 近期曲线 {fmt(lE, qE)}")
        say(s2)
        for nm, E_, S_ in (("rule", Er, Sr), ("lgb", El, Sl), ("recent", Eq, Sq)):
            pts.append(pd.DataFrame({"tariff": tf, "cfg": c, "pc": pc, "family": nm, "k": E_.index.astype(str),"E": E_.values / len(days), "share": S_.values,
                                     "E_rule_best": ref / len(days), "E_pi": (base_E / len(days)) if base_E else np.nan}))
    return pts


say(f"## 一、二 上海电价（利润口径；服务水平口径）")
pts = []
if (OUT / "blp_main_fc.parquet").exists():
    pts += section_main(pd.read_parquet(OUT / "blp_main_rule.parquet"), pd.read_parquet(OUT / "blp_main_fc.parquet"), "main")

if (OUT / "blp_models_fc.parquet").exists():
    say("\n## 三 9 个模型（利润口径）")
    M = pd.read_parquet(OUT / "blp_models_fc.parquet")
    M["cost"] = M.E + M.Pen
    M = M[M.day >= EVAL0]
    for (c, pc), g in M.groupby(["cfg", "pc"], sort=False):
        s = f"{c} {pc:.0f} kW："
        for t in ("default", "mid", "max"):
            h = g[g.cand == f"mean@0|{t}"]
            a = h.groupby("model").agg(mae=("mae", "mean"), cost=("cost", "sum"), S=("S", "sum"))
            d = h.pivot(index="day", columns="model", values="cost")
            s += f" {t}: τ {kendalltau(a.mae, a.cost)[0]:.2f}，最省 {a.cost.idxmin()}，TimesFM/LGB {fmt(d['timesfm3_cov'], d['lgb_local'])}，LQR/LGB {fmt(d['lqr'], d['lgb_local'])}；"
        P = g.pivot_table(index=["day", "model"], columns="cand", values="cost")
        per = P.groupby("model").sum()
        s += f" 最大 vs 默认 逐模型 {100 * (per['mean@0|max'] / per['mean@0|default'] - 1).min():+.1f}…{100 * (per['mean@0|max'] / per['mean@0|default'] - 1).max():+.1f}%"
        s += f"（{int((per['mean@0|max'] < per['mean@0|default']).sum())}/9 更省）；最大 vs 最小 {100 * (per['mean@0|max'] / per['mean@0|min'] - 1).min():+.1f}…{100 * (per['mean@0|max'] / per['mean@0|min'] - 1).max():+.1f}%"
        say(s)

if (OUT / "blp_tariffs_fc.parquet").exists():
    say("\n## 四 其他电价（6 小时，2.5 倍，利润口径）")
    pts += section_main(pd.read_parquet(OUT / "blp_tariffs_rule.parquet"), pd.read_parquet(OUT / "blp_tariffs_fc.parquet"), "tariffs")

if pts:
    pd.concat(pts, ignore_index=True).to_parquet(OUT / "blp_tradeoff_points.parquet")
(OUT / "blp_summary.txt").write_text("\n".join(lines), encoding="utf-8")
