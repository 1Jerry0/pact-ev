"""表 5（A7 电池级仿真）的利润口径版本：读 结果/利润口径/exp_bl.csv（规则、未调参计划、调参管线、完全信息）与 exp_bl2_a.csv（完美小时预测），
6 小时库存 × 8/12/18/25/50 kW（另附 4 小时）；以及各策略的缺供率（读 bl_fc / bl_rule）。输出：结果/table5_profit.md"""
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from exp_packs_value import pick, START  # noqa: E402

P = ROOT / "results" / "profit_basis"
A = pd.read_csv(P / "exp_bl.csv")
B = pd.read_csv(P / "exp_bl2_a.csv")
A = A[A.cfg.str.endswith(", 3)")]
B = B[B.cfg.str.endswith(", 3)")]
M = A.merge(B[["cfg", "pc", "pih_roll", "pih_roll_lo", "pih_roll_hi"]], on=["cfg", "pc"])
FC = pd.read_parquet(P / "bl_fc.parquet")
RU = pd.read_parquet(P / "bl_rule.parquet")


def b(v, lo, hi):
    s = f"{v:+.1f}"
    return f"**{s}**" if (lo > 0 or hi < 0) else s


rows = []
for _, r in M.iterrows():
    c, pc = r.cfg, r.pc
    g = RU[(RU.cfg == c) & (RU.pc == pc)]
    Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
    Ms = g.pivot(index="day", columns="th", values="short").sort_index()
    dem = g.drop_duplicates("day").set_index("day").dem.sort_index()
    h = FC[(FC.cfg == c) & (FC.pc == pc)]
    Fs = h.pivot(index="day", columns="cand", values="short").sort_index()
    days = Fs.index[Fs.index >= START]
    k0 = Mc[Mc.index < pd.Timestamp("2024-04-01")].sum().idxmin()
    ks = [pick(Mc, d, k0) for d in days]
    rs = np.array([Ms.loc[d, k] for d, k in zip(days, ks)]).sum() / dem.loc[days].sum()
    sh = lambda col: 100 * Fs.loc[days, col].sum() / dem.loc[days].sum()
    rows.append({"库存": eval(c)[0], "充电机": f"{pc:.0f} kW（{r.hours:.2f} h）", "规则（元/天）": f"{r.cost_rule:,.0f}",
                 "最小缓冲": b(r.fc_min, r.fc_min_lo, r.fc_min_hi), "求解器计划": b(r.fc_default, r.fc_default_lo, r.fc_default_hi),
                 "中点": b(r.fc_mid, r.fc_mid_lo, r.fc_mid_hi), "最大缓冲": b(r.fc_max, r.fc_max_lo, r.fc_max_hi),
                 "调参管线": f"{b(r.fc_roll, r.fc_roll_lo, r.fc_roll_hi)} [{r.fc_roll_lo:.1f}, {r.fc_roll_hi:.1f}]",
                 "完美小时预测": b(r.pih_roll, r.pih_roll_lo, r.pih_roll_hi), "完全信息": b(r.pi_roll, r.pi_roll_lo, r.pi_roll_hi),
                 "换下即充": b(r.full, r.full_lo, r.full_hi),
                 "缺供率 规则/默认/最大/管线 %": f"{100 * rs:.2f} / {sh('mean@0|default'):.2f} / {sh('mean@0|max'):.2f}"})
T = pd.DataFrame(rows).sort_values(["库存", "充电机"], ascending=[False, True])
md = T.to_markdown(index=False)
(ROOT / "results" / "table5_profit.md").write_text(md, encoding="utf-8")
print(md)
