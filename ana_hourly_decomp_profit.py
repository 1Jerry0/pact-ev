"""小时模型中计划价值与预测价值的分解（利润口径）：数据锚定网格与紧充电网格。
规则（66 组阈值，按 D−29…D−2 的成本滚动选）、求解器默认计划、中点计划（LightGBM 与季节经验分位数，均值预测）的总遗憾之比。
数据锚定网格读 cb_hourly_plans / cb_hourly_rule；紧充电网格的规则在此重算（exp_cost_basis.rule_split），计划遗憾读 cb_t4_A7tight。
输出：结果/hourly_decomp_profit.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(HERE.parent))
import cfg  # noqa: E402
import data as D  # noqa: E402
from exp_cost_basis import rule_split, kappa_profit  # noqa: E402
import sim  # noqa: E402

OUT = ROOT / "results"
TUNE0 = pd.Timestamp("2024-05-01")
lines = []


def pick(M, d, fb):
    past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
    return past.sum().idxmin() if len(past) >= 7 else fb


def tuned_rule_regret(R):
    """R：day × th 的遗憾矩阵（含 3 月起的历史）。返回 5 月起逐日滚动选择的遗憾。"""
    k0 = R[R.index < pd.Timestamp("2024-04-01")].sum().idxmin()
    days = R.index[R.index >= TUNE0]
    return pd.Series([R.loc[d, pick(R, d, k0)] for d in days], index=days)


# ---- 数据锚定网格
P = pd.read_parquet(OUT / "cb_hourly_plans.parquet", filters=[("inp", "==", "mean")])
P = P[P.model.isin(["lgb_local", "seasonal_q"])].copy()
P["reg"] = P.E + P.Pen - P.E_opt - P.Pen_opt
RU = pd.read_parquet(OUT / "cb_hourly_rule.parquet")
RU["reg"] = RU.E + RU.Pen - RU.E_opt - RU.Pen_opt
lines.append("## 数据锚定网格（5 月起；总遗憾之比）")
agg = {"rule": 0, "lgb_default": 0, "lgb_mid": 0, "sq_mid": 0}
for c, g in RU.groupby("cfg"):
    rr = tuned_rule_regret(g.pivot(index="day", columns="th", values="reg").sort_index())
    h = P[(P.cfg == c) & (P.day >= TUNE0)]
    v = {"rule": rr.sum(), "lgb_default": h[(h.model == "lgb_local") & (h.tie == "default")].reg.sum(),
         "lgb_mid": h[(h.model == "lgb_local") & (h.tie == "mid")].reg.sum(), "sq_mid": h[(h.model == "seasonal_q") & (h.tie == "mid")].reg.sum()}
    for k in agg:
        agg[k] += v[k]
    lines.append(f"  {c}：规则/默认 {v['rule'] / v['lgb_default']:.2f}；LightGBM 中点/规则 {v['lgb_mid'] / v['rule']:.2f}；近期曲线中点/规则 {v['sq_mid'] / v['rule']:.2f}")
lines.append(f"  合计：规则/默认 {agg['rule'] / agg['lgb_default']:.2f}；LightGBM 中点/规则 {agg['lgb_mid'] / agg['rule']:.2f}；近期曲线中点/规则 {agg['sq_mid'] / agg['rule']:.2f}")

# ---- 紧充电网格：重算规则
H = D.mp_hourly()
S = cfg.BSS
hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
TIGHT = [(c, p) for c in (2, 4, 8) for p in (1.1, 1.3, 1.6)]
rows = []
for d in pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D"):
    y = H[S].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
    if not np.isfinite(y).all():
        continue
    price, pen = D.price(S, d.month), kappa_profit(d.month)
    for c, p in TIGHT:
        cap, pmax = c * hmean, p * hmean
        opt = sim.bss_plan(y, price, pen, cap, pmax, None)[2]
        E, Sh, Pn = rule_split(y, price, pen, cap, pmax)
        rows += [(d, f"({c}, {p})", i, E[i] + Pn[i] - opt) for i in range(len(E))]
RT = pd.DataFrame(rows, columns=["day", "cfg", "th", "reg"])
T4 = pd.read_parquet(OUT / "cb_t4_A7tight.parquet")
T4 = T4[(T4.model == "lgb_local") & (T4.day >= TUNE0)]
lines.append("\n## 紧充电网格（LightGBM 中点计划 vs 规则）")
better = 0
for c, g in RT.groupby("cfg"):
    rr = tuned_rule_regret(g.pivot(index="day", columns="th", values="reg").sort_index())
    mid = T4[T4.cfg == c].set_index("day").reg_mid.reindex(rr.index)
    better += rr.sum() < mid.sum()
    lines.append(f"  {c}：中点/规则 {mid.sum() / rr.sum():.2f}")
lines.append(f"  规则优于中点计划的配置：{better}/{RT.cfg.nunique()}")
txt = "\n".join(lines)
(OUT / "hourly_decomp_profit.txt").write_text(txt, encoding="utf-8")
print(txt)
