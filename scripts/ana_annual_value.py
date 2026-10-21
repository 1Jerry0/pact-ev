"""年化的经济意义（A7 电池级仿真，6 小时库存，2.5 倍平均需求，12 / 25 kW，利润口径）。
相对差异均为“总成本之比 − 1”（rel_ci 的点估计），与规则同一批日子，因此每日金额差 = 规则日均成本 × 百分点差。按每年 365 天折算。
输入：结果/利润口径/exp_bl.csv（换下即充、最小/最大缓冲、调参管线）、结果/blp_main_* （LightGBM 管线相对近期曲线计划）。输出：结果/annual_value.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
import cfg  # noqa: E402
from exp_packs_value import rel_ci, pick  # noqa: E402

B = pd.read_csv(ROOT / "results" / "profit_basis" / "exp_bl.csv")
B = B[B.cfg == "(6, 2.5, 3)"].set_index("pc")
RU, FC = pd.read_parquet(ROOT / "results" / "blp_main_rule.parquet"), pd.read_parquet(ROOT / "results" / "blp_main_fc.parquet")
TUNE0 = pd.Timestamp("2024-05-01")
lines = ["A7，6 小时库存，2.5 倍平均需求；每日金额 = 规则日均成本 × 百分点差；年化按 365 天"]
for pc in (12.0, 25.0):
    r = B.loc[pc]
    rule = r.cost_rule
    d_full = rule * r.full / 100
    d_minmax = rule * (r.fc_min - r.fc_max) / 100
    d_pipe = rule * r.fc_roll / 100
    # 预测本身：LightGBM 管线相对近期曲线计划（上海电价，与表 A4 同口径）
    g = RU[(RU.tariff == "shanghai") & (RU.cfg == "(6, 2.5)") & (RU.pc == pc) & (RU.th >= 0)]
    Mc = g.assign(cost=g.E + g.Pen).pivot(index="day", columns="th", values="cost").sort_index()
    h = FC[(FC.tariff == "shanghai") & (FC.cfg == "(6, 2.5)") & (FC.pc == pc)].copy()
    h["k"], h["cost"] = h.model + ":" + h.cand, h.E + h.Pen
    Fc = h.pivot(index="day", columns="k", values="cost").sort_index()
    days = Fc.index[Fc.index >= TUNE0]
    cols = lambda p: [k for k in Fc.columns if k.startswith(p)]
    rec = np.array([Fc.loc[d, pick(Fc[cols("seasonal_q:")], d, "seasonal_q:mean@0|mid")] for d in days])
    lgb = np.array([Fc.loc[d, pick(Fc[cols("lgb_local:")], d, "lgb_local:mean@0|mid")] for d in days])
    m, lo, hi = rel_ci(lgb, rec)
    d_fc = lgb.mean() - rec.mean()
    lines.append(f"{pc:.0f} kW：规则 {rule:,.0f} 元/天；换下即充多花 {d_full:,.0f} 元/天（{d_full * 365 / 1e4:.1f} 万元/年，{r.full:+.1f}%）；"
                 f"最小缓冲比最大缓冲多花 {d_minmax:,.0f} 元/天（{d_minmax * 365 / 1e4:.1f} 万元/年）；调参管线相对规则 {d_pipe:+,.0f} 元/天（{d_pipe * 365 / 1e4:+.1f} 万元/年，{r.fc_roll:+.1f}% [{r.fc_roll_lo:+.1f}, {r.fc_roll_hi:+.1f}]）；"
                 f"预测本身（LightGBM 相对近期曲线）{d_fc:+,.1f} 元/天（{d_fc * 365 / 1e4:+.2f} 万元/年，{100 * m:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}]）")
txt = "\n".join(lines)
(ROOT / "results" / "annual_value.txt").write_text(txt, encoding="utf-8")
print(txt)
