"""表 A6（时间稳定性）的利润口径版本：评估期前半段（4 或 5 月至 8 月）与后半段（9—12 月）。
读 结果\cb_t4_A7.parquet、cb_hourly_plans/rule、结果\利润口径\bl_rule/bl_fc（电池级，6 h）、bl_rule_a8/bl_fc_a8、dfl_x3.parquet。输出：结果\halves_profit.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from analysis import boot_mean  # noqa: E402
from exp_packs_value import rel_ci, pick  # noqa: E402

OUT, PR = ROOT / "results", ROOT / "results" / "profit_basis"
H1 = lambda d: d.month <= 8
H2 = lambda d: d.month >= 9
lines = []


def say(s):
    lines.append(s)
    print(s)


def fmt(t):
    m, lo, hi = t
    return f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


# 小时模型：中点 vs 默认（日度降幅均值）
T = pd.read_parquet(OUT / "cb_t4_A7.parquet")
daily = T.groupby("day")[["reg_mid", "reg_default"]].sum()
for lab, f in (("前半", H1), ("后半", H2), ("全期", lambda d: d.month > 0)):
    x = daily[f(daily.index)]
    m, ci, _ = boot_mean(1 - x.reg_mid / x.reg_default)
    say(f"小时模型 中点 vs 默认 {lab}：降幅 {100 * m:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")

# 小时模型：LightGBM 中点 vs 滚动规则（5 月起）
P = pd.read_parquet(OUT / "cb_hourly_plans.parquet", filters=[("inp", "==", "mean"), ("tie", "==", "mid"), ("model", "==", "lgb_local")])
P["reg"] = P.E + P.Pen - P.E_opt - P.Pen_opt
RU = pd.read_parquet(OUT / "cb_hourly_rule.parquet")
RU["reg"] = RU.E + RU.Pen - RU.E_opt - RU.Pen_opt
rule = []
for c, g in RU.groupby("cfg"):
    M = g.pivot(index="day", columns="th", values="reg").sort_index()
    k0 = M[M.index < pd.Timestamp("2024-04-01")].sum().idxmin()
    days = M.index[M.index >= pd.Timestamp("2024-05-01")]
    rule.append(pd.Series([M.loc[d, pick(M, d, k0)] for d in days], index=days))
rr = pd.concat(rule).groupby(level=0).sum()
lm = P.groupby("day").reg.sum().reindex(rr.index)
for lab, f in (("前半", H1), ("后半", H2), ("全期", lambda d: d.month > 0)):
    k = f(rr.index)
    say(f"小时模型 LightGBM 中点 vs 规则 {lab}：总遗憾 {fmt(rel_ci(lm[k].to_numpy(), rr[k].to_numpy()))}")

# 电池级：管线 vs 规则，最大 vs 最小（6 h）
FC, RB = pd.read_parquet(PR / "bl_fc.parquet"), pd.read_parquet(PR / "bl_rule.parquet")
for pc in (8.0, 12.0, 18.0, 25.0, 50.0):
    g = RB[(RB.cfg == "(6, 2.5, 3)") & (RB.pc == pc)]
    Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
    h = FC[(FC.cfg == "(6, 2.5, 3)") & (FC.pc == pc)]
    Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
    days = Fc.index[Fc.index >= pd.Timestamp("2024-05-01")]
    k0 = Mc[Mc.index < pd.Timestamp("2024-04-01")].sum().idxmin()
    rule = pd.Series([Mc.loc[d, pick(Mc, d, k0)] for d in days], index=days)
    lgb = [k for k in Fc.columns if not k.startswith("pi@")]
    pipe = pd.Series([Fc.loc[d, pick(Fc[lgb], d, "mean@0|mid")] for d in days], index=days)
    seg = []
    for lab, f in (("前半", H1), ("后半", H2)):
        k = f(days)
        seg.append(f"{lab} 管线 {fmt(rel_ci(pipe[k].to_numpy(), rule[k].to_numpy()))}，最大 vs 最小 {fmt(rel_ci(Fc.loc[days[k], 'mean@0|max'].to_numpy(), Fc.loc[days[k], 'mean@0|min'].to_numpy()))}")
    say(f"电池级 6 h {pc:.0f} kW：" + "；".join(seg))

# A8：15 块电池各配充电机
FA, RA = pd.read_parquet(PR / "bl_fc_a8.parquet"), pd.read_parquet(PR / "bl_rule_a8.parquet")
c0 = RA.cfg.iloc[0]
for c in RA.cfg.unique():
    if not c.startswith("(11"):
        continue
    g = RA[RA.cfg == c]
    pc = sorted(g.pc.unique())[-1]                     # 1 小时充满
    g = g[g.pc == pc]
    Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
    h = FA[(FA.cfg == c) & (FA.pc == pc)]
    Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
    days = Fc.index[Fc.index >= pd.Timestamp("2024-05-01")]
    k0 = Mc[Mc.index < pd.Timestamp("2024-04-01")].sum().idxmin() if (Mc.index < pd.Timestamp("2024-04-01")).any() else Mc.columns[0]
    rule = pd.Series([Mc.loc[d, pick(Mc, d, k0)] for d in days], index=days)
    lgb = [k for k in Fc.columns if not k.startswith("pi@")]
    pipe = pd.Series([Fc.loc[d, pick(Fc[lgb], d, "mean@0|mid")] for d in days], index=days)
    a, b = (lambda d: d.month <= 7), (lambda d: d.month >= 8)
    say(f"A8 {c}：5—7 月 管线 vs 规则 {fmt(rel_ci(pipe[a(days)].to_numpy(), rule[a(days)].to_numpy()))}；8—10 月 {fmt(rel_ci(pipe[b(days)].to_numpy(), rule[b(days)].to_numpy()))}")
    break

# 价格隐含分位数 vs 统一分位数（求解器计划）
fp = PR / "dfl_x3.parquet"
if fp.exists():
    X = pd.read_parquet(fp)
    say("dfl_x3 列：" + ", ".join(map(str, X.columns[:20])))
(OUT / "halves_profit.txt").write_text("\n".join(lines), encoding="utf-8")
