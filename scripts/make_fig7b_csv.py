"""图 7(b) 的深圳 60 区数据点（利润口径）：LightGBM 管线相对近期曲线计划的成本差（%）与 95% 置信区间。
与 利润口径/exp_bl_uev_profile.py 的 analyse() 相同（各区按规则均值归一后等权合并）。输出：结果/利润口径/uev_profile_fc.csv（%）。"""
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
import cfg  # noqa: E402
from exp_packs_value import pick, rel_ci  # noqa: E402

PR = ROOT / "results" / "profit_basis"
SQ, RU, FC = (pd.read_parquet(PR / f) for f in ("bl_uev_profile.parquet", "bl_uev_rule.parquet", "bl_uev_fc.parquet"))
daily = {}
for (z, pc), g in SQ.groupby(["zone", "pc"], sort=False):
    r = RU[(RU.zone == z) & (RU.pc == pc)]
    Mc = pd.DataFrame(np.vstack(r.costs.to_list()), index=pd.DatetimeIndex(r.day)).sort_index()
    Fl = FC[(FC.zone == z) & (FC.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
    Fq = g.pivot(index="day", columns="cand", values="cost").sort_index()
    lgb = [k for k in Fl.columns if not k.startswith("pi@")]
    days = [d for d in Fq.index if d >= cfg.UEV_EVAL_START and d in Fl.index]
    k0 = Mc[Mc.index <= cfg.UEV_DEV_END].sum().idxmin()
    rule = np.array([Mc.loc[d, pick(Mc, d, k0)] for d in days])
    lg = np.array([Fl.loc[d, pick(Fl[lgb], d, "mean@0|mid")] for d in days])
    sq = np.array([Fq.loc[d, pick(Fq, d, "mean@0|mid")] for d in days])
    for nm, x in (("lgb", lg), ("sq", sq)):
        daily.setdefault((pc, nm), []).append(pd.Series(x / rule.mean(), index=days))
rows = []
for pc in sorted({k[0] for k in daily}):
    a = pd.concat(daily[(pc, "lgb")], axis=1).mean(axis=1).to_numpy()
    b = pd.concat(daily[(pc, "sq")], axis=1).mean(axis=1).to_numpy()
    m, lo, hi = rel_ci(a, b)
    rows.append((f"Shenzhen 60 zones, {pc:.0f} kW", round(100 * m, 2), round(100 * lo, 2), round(100 * hi, 2)))
out = pd.DataFrame(rows, columns=["label", "m", "lo", "hi"])
out.to_csv(PR / "uev_profile_fc.csv", index=False)
print(out.to_string(index=False))
