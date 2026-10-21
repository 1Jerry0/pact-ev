"""附录表 A3—A5 的利润口径版本（读结果文件生成 markdown）：
A3 电池级全部配置；A4 四种电价下的计划价值与预测价值；A5 电池状态逐日结转。输出：结果\appendix_profit.md"""
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
import cfg  # noqa: E402
from exp_packs_value import rel_ci, pick  # noqa: E402

OUT, PR = ROOT / "results", ROOT / "results" / "profit_basis"
TUNE0 = pd.Timestamp("2024-05-01")
md = []


def b(v, lo, hi, nd=1):
    s = f"{v:+.{nd}f}".replace("-", "−")
    return f"**{s}**" if (lo > 0 or hi < 0) else s


# ---- A3
A = pd.read_csv(PR / "exp_bl.csv")
A = A[A.cfg.str.endswith(", 3)")]
B = pd.read_csv(PR / "exp_bl2_a.csv")[["cfg", "pc", "pih_roll", "pih_roll_lo", "pih_roll_hi"]]
A = A.merge(B, on=["cfg", "pc"])
A["N"] = A.cfg.map(lambda s: eval(s)[0])
A = A.sort_values(["N", "pc"], ascending=[False, True])
md.append("| 充电机；库存 | 规则（元/天） | 换下即充 | 最小缓冲 | 求解器计划 | 中点 | 最大缓冲 | 调参管线 | 完美小时预测 | 完全信息 |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in A.iterrows():
    md.append(f"| {r.pc:.0f} kW；{r.N} h | {r.cost_rule:,.0f} | {b(r.full, r.full_lo, r.full_hi)} | {b(r.fc_min, r.fc_min_lo, r.fc_min_hi)} | "
              f"{b(r.fc_default, r.fc_default_lo, r.fc_default_hi)} | {b(r.fc_mid, r.fc_mid_lo, r.fc_mid_hi)} | {b(r.fc_max, r.fc_max_lo, r.fc_max_hi)} | "
              f"{b(r.fc_roll, r.fc_roll_lo, r.fc_roll_hi)} | {b(r.pih_roll, r.pih_roll_lo, r.pih_roll_hi)} | {b(r.pi_roll, r.pi_roll_lo, r.pi_roll_hi)} |")
md.append("")

# ---- A4（exp_bl_profit 的口径：其他电价下缺供按该电价 + A7 服务费计价）
rows = []
for tf, lab, rfp, ffp in (("shanghai", "上海", "blp_main_rule", "blp_main_fc"), ("guangdong", "广东", "blp_tariffs_rule", "blp_tariffs_fc"),
                          ("pge_ev2a", "PG&E EV2-A", "blp_tariffs_rule", "blp_tariffs_fc"), ("two_block", "两段式", "blp_tariffs_rule", "blp_tariffs_fc")):
    RU, FC = pd.read_parquet(OUT / f"{rfp}.parquet"), pd.read_parquet(OUT / f"{ffp}.parquet")
    for pc in (12.0, 25.0):
        g = RU[(RU.tariff == tf) & (RU.cfg == "(6, 2.5)") & (RU.pc == pc) & (RU.th >= 0)]
        Mc = g.assign(cost=g.E + g.Pen).pivot(index="day", columns="th", values="cost").sort_index()
        h = FC[(FC.tariff == tf) & (FC.cfg == "(6, 2.5)") & (FC.pc == pc)].copy()
        h["k"], h["cost"] = h.model + ":" + h.cand, h.E + h.Pen
        Fc = h.pivot(index="day", columns="k", values="cost").sort_index()
        days = Fc.index[Fc.index >= TUNE0]
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        roll = lambda M, fb, cols: np.array([M.loc[d, pick(M[cols], d, fb)] for d in days])
        cols = lambda p: [k for k in Fc.columns if k.startswith(p)]
        rule = roll(Mc, k0, list(Mc.columns))
        rec = roll(Fc, "seasonal_q:mean@0|mid", cols("seasonal_q:"))
        lgb = roll(Fc, "lgb_local:mean@0|mid", cols("lgb_local:"))
        pi = roll(Fc, "pi:count@0|mid", cols("pi:"))
        hind = Mc.loc[days].min(axis=1).to_numpy()
        mm = rel_ci(Fc.loc[days, "lgb_local:mean@0|max"].to_numpy(), Fc.loc[days, "lgb_local:mean@0|min"].to_numpy())
        f = lambda t, nd=1: f"{100 * t[0]:+.{nd}f} [{100 * t[1]:+.{nd}f}, {100 * t[2]:+.{nd}f}]"
        rows.append(f"| {lab}；{pc:.0f} kW | {rule.mean():,.0f} | {f(rel_ci(rec, rule))} | {f(rel_ci(lgb, rule))} | {f(rel_ci(lgb, rec), 2)} | "
                    f"{100 * rel_ci(pi, rule)[0]:+.1f} | {100 * rel_ci(hind, rule)[0]:+.1f} | {100 * mm[0]:+.1f} |".replace("-", "−"))
md.append("| 电价；充电机 | 规则（元/天） | 近期曲线计划 | LightGBM 管线 | LightGBM 相对于近期曲线计划 | 完全信息 | 事后最优阈值规则 | 最大缓冲相对于最小缓冲 |")
md.append("|---|---|---|---|---|---|---|---|")
md += rows
md.append("")

# ---- A6 时间稳定性（前半 4/5—8 月，后半 9—12 月）
from analysis import boot_mean  # noqa: E402
H1, H2, ALL = (lambda d: d.month <= 8), (lambda d: d.month >= 9), (lambda d: d.month > 0)
f = lambda t: f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]".replace("-", "−")
T = pd.read_parquet(OUT / "cb_t4_A7.parquet").groupby("day")[["reg_mid", "reg_default"]].sum()
cells = []
for h in (H1, H2, ALL):
    x = T[h(T.index)]
    m, ci, _ = boot_mean(1 - x.reg_mid / x.reg_default)
    cells.append(f"{100 * m:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
md.append("| 比较 | 度量 | 前半段 | 后半段 | 全评估期 |")
md.append("|---|---|---|---|---|")
md.append("| 小时模型：中点计划相对于求解器默认计划 | 日遗憾降幅，% | " + " | ".join(cells) + " |")
P = pd.read_parquet(OUT / "cb_hourly_plans.parquet", filters=[("inp", "==", "mean"), ("tie", "==", "mid"), ("model", "==", "lgb_local")])
P["reg"] = P.E + P.Pen - P.E_opt - P.Pen_opt
RU = pd.read_parquet(OUT / "cb_hourly_rule.parquet")
RU["reg"] = RU.E + RU.Pen - RU.E_opt - RU.Pen_opt
rr = []
for c, g in RU.groupby("cfg"):
    M = g.pivot(index="day", columns="th", values="reg").sort_index()
    k0 = M[M.index < cfg.MP_EVAL_START].sum().idxmin()
    dd = M.index[M.index >= TUNE0]
    rr.append(pd.Series([M.loc[d, pick(M, d, k0)] for d in dd], index=dd))
rr = pd.concat(rr).groupby(level=0).sum()
lm = P.groupby("day").reg.sum().reindex(rr.index)
md.append("| 小时模型：LightGBM 中点计划相对于规则 | 总遗憾，% | " + " | ".join(f(rel_ci(lm[h(rr.index)].to_numpy(), rr[h(rr.index)].to_numpy())) for h in (H1, H2, ALL)) + " |")
FC, RB = pd.read_parquet(PR / "bl_fc.parquet"), pd.read_parquet(PR / "bl_rule.parquet")
pipe_c, mm_c = {}, {}
for pc in (8.0, 12.0, 18.0, 25.0, 50.0):
    g = RB[(RB.cfg == "(6, 2.5, 3)") & (RB.pc == pc)]
    Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
    Fc = FC[(FC.cfg == "(6, 2.5, 3)") & (FC.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
    days = Fc.index[Fc.index >= TUNE0]
    k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
    rule = pd.Series([Mc.loc[d, pick(Mc, d, k0)] for d in days], index=days)
    lgb = [k for k in Fc.columns if not k.startswith("pi@")]
    pipe = pd.Series([Fc.loc[d, pick(Fc[lgb], d, "mean@0|mid")] for d in days], index=days)
    pipe_c[pc] = [100 * rel_ci(pipe[h(days)].to_numpy(), rule[h(days)].to_numpy())[0] for h in (H1, H2, ALL)]
    mm_c[pc] = [100 * rel_ci(Fc.loc[days[h(days)], "mean@0|max"].to_numpy(), Fc.loc[days[h(days)], "mean@0|min"].to_numpy())[0] for h in (H1, H2, ALL)]
s = lambda v: f"{v:+.1f}".replace("-", "−")
md.append("| 电池级：调参管线相对于规则，8 / 12 kW | 成本，% | " + " | ".join(" / ".join(s(pipe_c[p][i]) for p in (8.0, 12.0)) for i in range(3)) + " |")
md.append("| ……18 / 25 / 50 kW | 成本，% | " + " | ".join(" / ".join(s(pipe_c[p][i]) for p in (18.0, 25.0, 50.0)) for i in range(3)) + " |")
rng = lambda i: f"{s(max(mm_c[p][i] for p in mm_c))} 至 {s(min(mm_c[p][i] for p in mm_c))}"
md.append("| 电池级：最大缓冲相对于最小缓冲，8—50 kW | 成本，% | " + " | ".join(rng(i) for i in range(3)) + " |")
FA, RA = pd.read_parquet(PR / "bl_fc_a8.parquet"), pd.read_parquet(PR / "bl_rule_a8.parquet")
c = [x for x in RA.cfg.unique() if x.startswith("(11")][0]
g = RA[RA.cfg == c]
pc = sorted(g.pc.unique())[-1]
Mc = g[g.pc == pc].pivot(index="day", columns="th", values="cost").sort_index()
Fc = FA[(FA.cfg == c) & (FA.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
days = Fc.index[Fc.index >= TUNE0]
k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin() if (Mc.index < cfg.MP_EVAL_START).any() else Mc.columns[0]
rule = pd.Series([Mc.loc[d, pick(Mc, d, k0)] for d in days], index=days)
lgb = [k for k in Fc.columns if not k.startswith("pi@")]
pipe = pd.Series([Fc.loc[d, pick(Fc[lgb], d, "mean@0|mid")] for d in days], index=days)
a8 = [100 * rel_ci(pipe[h(days)].to_numpy(), rule[h(days)].to_numpy())[0] for h in ((lambda d: d.month <= 7), (lambda d: d.month >= 8), ALL)]
md.append("| A8，15 块电池：调参管线相对于规则$^a$ | 成本，% | " + " | ".join(s(v) for v in a8) + " |")
R = pd.read_parquet(PR / "dfl_x3.parquet")
M = R[R.tie == "default"].groupby(["day", "cand"]).regret.sum().unstack("cand")
dd = M.index[28:]
uni = ["mean"] + [k for k in M.columns if k.startswith("q")]
pri = ["mean"] + [k for k in M.columns if k.startswith("p")]
su = pd.Series([M.loc[d, pick(M[uni], d, "mean")] for d in dd], index=dd)
sp = pd.Series([M.loc[d, pick(M[pri], d, "mean")] for d in dd], index=dd)
md.append("| 价格隐含分位数相对于统一分位数，求解器默认计划 | 总遗憾，% | " + " | ".join(f(rel_ci(sp[h(dd)].to_numpy(), su[h(dd)].to_numpy())) for h in (H1, H2, ALL)) + " |")
md.append("")
txt = "\n".join(md)
(OUT / "appendix_profit.md").write_text(txt, encoding="utf-8")
print(txt)
