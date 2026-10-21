"""成本口径重算的分析（小时模型，A7，数据锚定网格 6 种资源配置）。输入：结果/cb_hourly_plans.parquet、cb_hourly_rule.parquet。
口径一 利润：成本 = 电费（含日末估值）+ 按逐小时零售总价计的少收收入；遗憾 = 成本 − 完全信息成本。
口径二 服务水平：缺供电量不超过需求的 1%（Sun, Tan & Tsang 2018 的拒绝服务概率 0.01）时的电费。
口径三 权衡曲线：各策略的（缺供率，超额电费）点及下包络（Gardner 1990；Kourentzes et al. 2020）。
输出：结果/cb_hourly_summary.txt、结果/cb_tradeoff_points.parquet"""
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
EVAL0, TUNE0 = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01")
TARGET = 0.01
NAMES = {"naive": "季节性朴素", "naive7": "周朴素", "seasonal_q": "季节经验分位数", "lqr": "线性分位数回归", "lgb_local": "LightGBM（单站）",
         "lgb_global": "LightGBM（全局）", "chronos2_cov": "Chronos-2", "timesfm3_cov": "TimesFM-3.0", "combo5": "等权组合"}
lines = []


def say(s=""):
    lines.append(s)
    print(s)


def fmt(a, b):
    m, lo, hi = rel_ci(a, b)
    flag = "*" if lo > 0 or hi < 0 else ""
    return f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{flag}"


def pick_profit(M, d, fb):
    past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
    return past.sum().idxmin() if len(past) >= 7 else fb


def pick_qos(E, S, dem, d, fb):
    """服务水平约束下滚动选择：过去 28 天（截至 D−2）缺供率不超过 1% 的策略中电费最低者；都不满足则取缺供率最低者。"""
    sl = slice(d - pd.Timedelta(days=29), d - pd.Timedelta(days=2))
    e, s, q = E.loc[sl], S.loc[sl], dem.loc[sl]
    if len(e) < 7:
        return fb
    share = s.sum() / q.sum()
    ok = share[share <= TARGET]
    return e.sum()[ok.index].idxmin() if len(ok) else share.idxmin()


P = pd.read_parquet(OUT / "cb_hourly_plans.parquet")
P = P[P.model != "check_x1max"]
R = pd.read_parquet(OUT / "cb_hourly_rule.parquet")
for X in (P, R):
    X["cost"] = X.E + X.Pen
    X["reg"] = X.cost - (X.E_opt + X.Pen_opt)
P = P[P.day >= EVAL0]
models = list(NAMES)

# ------------------------------------------------------------------ 口径一：利润
say("## 口径一 利润（缺供按逐小时零售总价计少收收入）")
mm = P[(P.inp == "mean")]
tot = mm.groupby(["tie"])[["reg", "S", "dem"]].sum()
say("均值预测、9 模型、6 配置合计：" + "；".join(f"{t} 遗憾 {r.reg / 1e3:.0f} 千元、缺供率 {100 * r.S / r.dem:.2f}%" for t, r in tot.iterrows()))
d_mid = mm[mm.tie == "mid"].groupby("day").reg.sum()
d_def = mm[mm.tie == "default"].groupby("day").reg.sum()
d_max = mm[mm.tie == "max"].groupby("day").reg.sum()
say(f"中点 vs 默认 日遗憾合计 {fmt(d_mid, d_def)}；最大缓冲 vs 默认 {fmt(d_max, d_def)}")
for c, g in mm.groupby("cfg"):
    a, b = g[g.tie == "mid"].groupby("day").reg.sum(), g[g.tie == "default"].groupby("day").reg.sum()
    say(f"  {c}：中点 vs 默认 {fmt(a, b)}")

say("\n精度与价值（中点计划、均值预测）：")
bd, taus = [], []
for c, g in mm[mm.tie == "mid"].groupby("cfg"):
    agg = g.groupby("model").agg(mae=("mae", "mean"), reg=("reg", "sum"))
    bd.append(agg.mae.idxmin() != agg.reg.idxmin())
    taus.append(kendalltau(agg.mae, agg.reg)[0])
    say(f"  {c}：MAE 最优 {NAMES[agg.mae.idxmin()]}，遗憾最低 {NAMES[agg.reg.idxmin()]}，τ {taus[-1]:.2f}")
say(f"  MAE 最优≠遗憾最低的配置 {100 * np.mean(bd):.0f}%，τ 中位数 {np.median(taus):.2f}")
D = mm[mm.tie == "mid"].pivot_table(index="day", columns="model", values="reg", aggfunc="sum")
say(f"  LQR/LightGBM {fmt(D['lqr'], D['lgb_local'])}；TimesFM/LightGBM {fmt(D['timesfm3_cov'], D['lgb_local'])}；Chronos/LightGBM {fmt(D['chronos2_cov'], D['lgb_local'])}")
Dd = mm[mm.tie == "default"].pivot_table(index="day", columns="model", values="reg", aggfunc="sum")
say(f"  （求解器默认计划）LQR/LightGBM {fmt(Dd['lqr'], Dd['lgb_local'])}；TimesFM/LightGBM {fmt(Dd['timesfm3_cov'], Dd['lgb_local'])}")

say("\n计划价值与预测价值（利润口径，5 月起；规则与管线每天按过去 28 天成本重选参数）：")
for c, g in P.groupby("cfg"):
    r = R[R.cfg == c]
    Mr = r.pivot(index="day", columns="th", values="cost").sort_index()
    k0 = Mr[Mr.index < EVAL0].sum().idxmin()
    days = Mr.index[Mr.index >= TUNE0]
    rule = np.array([Mr.loc[d, pick_profit(Mr, d, k0)] for d in days])
    opt = r.drop_duplicates("day").set_index("day").loc[days]
    out = {}
    for m in ("seasonal_q", "lgb_local"):
        h = g[g.model == m].copy()
        h["k"] = h.inp + "|" + h.tie
        Mf = h.pivot(index="day", columns="k", values="cost").sort_index()
        out[m] = np.array([Mf.loc[d, pick_profit(Mf, d, "mean|mid")] for d in days])
        out[m + "_fix"] = Mf.loc[days, "mean|mid"].to_numpy()
    o = (opt.E_opt + opt.Pen_opt).to_numpy()
    say(f"  {c}：遗憾占规则遗憾 近期曲线（中点，固定）{(out['seasonal_q_fix'] - o).sum() / (rule - o).sum():.2f}、LightGBM（中点，固定）{(out['lgb_local_fix'] - o).sum() / (rule - o).sum():.2f}；"
        f"调参后成本 近期曲线 vs 规则 {fmt(out['seasonal_q'], rule)}，LightGBM vs 近期曲线 {fmt(out['lgb_local'], out['seasonal_q'])}")

# ------------------------------------------------------------------ 口径二：服务水平 1%
say(f"\n## 口径二 服务水平（缺供 ≤ 需求的 {100 * TARGET:.0f}%），电费以完全信息电费为基准的超额百分比")
pts = []
for c, g in P.groupby("cfg"):
    base_E = g.drop_duplicates("day").E_opt.sum()
    dem = g.drop_duplicates("day").dem.sum()
    agg = g.groupby(["model", "inp", "tie"]).agg(E=("E", "sum"), S=("S", "sum")).reset_index()
    agg["share"], agg["exE"] = agg.S / dem, agg.E / base_E - 1
    agg["cfg"] = c
    pts.append(agg)
    r = R[(R.cfg == c) & (R.day >= EVAL0)].groupby("th").agg(E=("E", "sum"), S=("S", "sum")).reset_index()
    r["share"], r["exE"] = r.S / dem, r.E / base_E - 1
    best = {}
    for m, h in agg.groupby("model"):
        ok = h[h.share <= TARGET]
        best[m] = (ok.exE.min(), ok.loc[ok.exE.idxmin(), ["inp", "tie"]].tolist()) if len(ok) else (np.nan, None)
    ok = r[r.share <= TARGET]
    rb = ok.exE.min() if len(ok) else np.nan
    mae = g.groupby("model").mae.mean()
    ranked = sorted(best, key=lambda m: best[m][0] if np.isfinite(best[m][0]) else 9)
    say(f"  {c}：规则 {100 * rb:+.2f}%；" + "，".join(f"{NAMES[m]} {100 * best[m][0]:+.2f}%({best[m][1][0]}|{best[m][1][1]})" if best[m][1] else f"{NAMES[m]} 达不到" for m in ranked))
    v = pd.Series({m: best[m][0] for m in best}).dropna()
    say(f"     MAE 最优 {NAMES[mae.idxmin()]}；1% 服务水平下电费最低 {NAMES[v.idxmin()]}；τ(MAE, 电费) {kendalltau(mae[v.index], v)[0]:.2f}")
    for tie in ("default", "mid", "max"):
        h = agg[(agg.model == "lgb_local") & (agg.tie == tie)].set_index("inp")
        say(f"     LightGBM {tie}：" + "，".join(f"{k} 缺供 {100 * h.loc[k, 'share']:.2f}%/电费 {100 * h.loc[k, 'exE']:+.2f}%" for k in ("mean", "q70", "q90")))
TP = pd.concat(pts, ignore_index=True)
TP.to_parquet(OUT / "cb_tradeoff_points.parquet")

say("\n同一服务水平下的计划选择：各模型只用求解器默认计划（任选分位数）与只用中点计划时，满足缺供 ≤ 1% 的最低超额电费（百分点）")
for c, g in TP.groupby("cfg"):
    seg = []
    for m, h in g.groupby("model"):
        best = {}
        for tie in ("default", "mid", "max"):
            ok = h[(h.tie == tie) & (h.share <= TARGET)]
            best[tie] = ok.exE.min() if len(ok) else np.nan
        seg.append((m, best))
    gaps = [100 * (b["default"] - b["mid"]) for _, b in seg if np.isfinite(b["default"])]
    nd = sum(1 for _, b in seg if not np.isfinite(b["default"]))
    say(f"  {c}：默认 − 中点 {min(gaps):.1f}…{max(gaps):.1f} 个百分点（{len(gaps)} 个模型；另有 {nd} 个模型的默认计划达不到 1%）；"
        + "，".join(f"{NAMES[m]} 默认 {100 * b['default']:+.1f}/中点 {100 * b['mid']:+.1f}/最大 {100 * b['max']:+.1f}" for m, b in seg if m in ("lgb_local", "timesfm3_cov", "lqr")))

say("\n服务水平约束下的滚动调参（5 月起，每天在过去 28 天缺供率 ≤ 1% 的参数中取电费最低者）：")
for c, g in P.groupby("cfg"):
    r = R[R.cfg == c]
    q = r.drop_duplicates("day").set_index("day").dem.sort_index()
    Er, Sr = (r.pivot(index="day", columns="th", values=v).sort_index() for v in ("E", "S"))
    days = Er.index[Er.index >= TUNE0]
    k0 = (Sr[Sr.index < EVAL0].sum() / q[q.index < EVAL0].sum()).idxmin()
    kr = [pick_qos(Er, Sr, q, d, k0) for d in days]
    res = {"规则": (np.array([Er.loc[d, k] for d, k in zip(days, kr)]), np.array([Sr.loc[d, k] for d, k in zip(days, kr)]))}
    for m in models:
        h = g[g.model == m].copy()
        h["k"] = h.inp + "|" + h.tie
        Ef, Sf = (h.pivot(index="day", columns="k", values=v).sort_index() for v in ("E", "S"))
        kk = [pick_qos(Ef, Sf, q, d, "mean|mid") for d in days]
        res[m] = (np.array([Ef.loc[d, k] for d, k in zip(days, kk)]), np.array([Sf.loc[d, k] for d, k in zip(days, kk)]))
    dem = q.loc[days].sum()
    lgbE = res["lgb_local"][0]
    say(f"  {c}：" + "；".join(f"{NAMES.get(k, k)} 缺供 {100 * s.sum() / dem:.2f}%，电费 vs LightGBM {fmt(e, lgbE) if k != 'lgb_local' else '—'}" for k, (e, s) in res.items()))

(OUT / "cb_hourly_summary.txt").write_text("\n".join(lines), encoding="utf-8")
