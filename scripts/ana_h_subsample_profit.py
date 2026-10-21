"""H2—H4 的子样本检验（利润口径）：剔除 12 月；只用 7—12 月。读 h2_profit.parquet、cb_t4_A7tight.parquet、rev_dswe_st_profit.parquet 与 rev_dswe_profit_a7_tight.parquet。
输出：结果/h_subsample_profit.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
from analysis import boot_mean  # noqa: E402

OUT = ROOT / "results"
SUBS = {"剔除 12 月": lambda d: d.dt.month != 12, "只用 7—12 月": lambda d: d.dt.month >= 7}
M = pd.read_parquet(OUT / "h2_profit.parquet")
M = M[(M.station == "A7_kwh") & (M.grid == "tight")]
T4 = pd.read_parquet(OUT / "cb_t4_A7tight.parquet")
cols = ["regret", "mae", "rmse", "crps", "dswe"]
P3 = pd.concat([pd.read_parquet(OUT / "rev_dswe_st_profit.parquet")[["unit", "day", "cfg", "model"] + cols],
                pd.read_parquet(OUT / "rev_dswe_profit_a7_tight.parquet")[["unit", "day", "cfg", "model"] + cols]], ignore_index=True)
lines = []
for lab, f in SUBS.items():
    m = M[f(M.day)]
    g = m.groupby(["cfg", "level", "day"]).regret
    disp = ((g.quantile(0.9) - g.quantile(0.1)) / g.mean().clip(lower=1e-9)).groupby("day").mean()
    h2 = boot_mean(disp)
    t = T4[f(T4.day)]
    daily = t.groupby("day")[["reg_mid", "reg_default"]].sum()
    h4 = boot_mean(1 - daily.reg_mid / daily.reg_default)
    X = P3[f(P3.day)].copy()
    for c in cols:
        X[c] = X[c] - X.groupby(["unit", "cfg", "day"])[c].transform("mean")
        X[c] = X[c] / X.groupby(["unit", "cfg"])[c].transform("std").replace(0, np.nan)
    X = X.dropna(subset=cols)
    fit = sm.OLS(X.regret, X[["mae", "rmse", "crps", "dswe"]]).fit(cov_type="hac-groupsum", cov_kwds={"time": pd.factorize(X.day)[0], "maxlags": 7})
    s = (f"{lab}：H2 {h2[0]:.3f} [{h2[1][0]:.3f}, {h2[1][1]:.3f}]；H3 DSWE 系数 {fit.params['dswe']:.3f}（p={fit.pvalues['dswe']:.2g}）；"
         f"H4 {100 * h4[0]:.1f}% [{100 * h4[1][0]:.1f}, {100 * h4[1][1]:.1f}]")
    lines.append(s)
    print(s)
(OUT / "h_subsample_profit.txt").write_text("\n".join(lines), encoding="utf-8")
