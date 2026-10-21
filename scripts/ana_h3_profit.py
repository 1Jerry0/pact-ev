"""H3（DSWE 的解释力）在利润口径下的合并检验：7 个储能站（9 种储能情景，服务费计价）与 A7 换电站合并的同日跨模型遗憾面板。
两个版本：A7 用预先设定的紧充电网格（对应 H3 检验）；A7 用数据锚定网格（对应 5.2 节图 5a）。
输入：rev_dswe_st_profit.parquet、rev_dswe_profit_a7_tight.parquet、rev_dswe_profit_a7.parquet。输出：结果/h3_profit.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
cols = ["regret", "mae", "rmse", "crps", "dswe", "dswe_ante"]
lines = []


def fit(X, label):
    X = X.copy()
    for c in cols:
        X[c] = X[c] - X.groupby(["unit", "cfg", "day"])[c].transform("mean")
        X[c] = X[c] / X.groupby(["unit", "cfg"])[c].transform("std").replace(0, np.nan)
    X = X.dropna(subset=cols)
    tc = pd.factorize(X.day)[0]
    r = {}
    for name, rhs in (("acc", ["mae", "rmse", "crps"]), ("post", ["mae", "rmse", "crps", "dswe"]), ("ante", ["mae", "rmse", "crps", "dswe_ante"])):
        r[name] = sm.OLS(X.regret, X[rhs]).fit(cov_type="hac-groupsum", cov_kwds={"time": tc, "maxlags": 7})
    f = r["post"]
    s = (f"{label}（{len(X):,} 个模型×日×情景）：R² 精度指标 {r['acc'].rsquared:.3f} → +事后 DSWE {f.rsquared:.3f} → +事前 DSWE {r['ante'].rsquared:.3f}；"
         f"DSWE 标准化系数 {f.params['dswe']:.3f}（p={f.pvalues['dswe']:.2g}）；MAE {f.params['mae']:.3f}（p={f.pvalues['mae']:.3f}），RMSE {f.params['rmse']:.3f}（p={f.pvalues['rmse']:.3f}）")
    lines.append(s)
    print(s)


ST = pd.read_parquet(OUT / "rev_dswe_st_profit.parquet")
for suf, label in (("_tight", "H3：储能站 + A7 紧充电网格"), ("", "图 5a：储能站 + A7 数据锚定网格")):
    A = pd.read_parquet(OUT / f"rev_dswe_profit_a7{suf}.parquet")
    A = A.rename(columns={"mid": "regret"}) if "mid" in A else A
    fit(pd.concat([ST[["unit", "day", "cfg", "model"] + cols], A[["unit", "day", "cfg", "model"] + cols]], ignore_index=True), label)
fit(ST[["unit", "day", "cfg", "model"] + cols], "仅储能站")
(OUT / "h3_profit.txt").write_text("\n".join(lines), encoding="utf-8")
