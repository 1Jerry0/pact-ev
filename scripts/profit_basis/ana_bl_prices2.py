"""exp_bl_prices 的补充：DSWE 的解释力是不是只是"平均偏差"？以及事前 DSWE 对整期模型排序的能力。读 bl_prices_panel.parquet。"""
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "formal" / "code"))
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau

OUT = HERE.parents[2] / "results" / "profit_basis"
X0 = pd.read_parquet(OUT / "bl_prices_panel.parquet")
X0["under"] = np.nan
cols = ["cost", "mae", "rmse", "crps", "bias", "dswe", "dswe_ea"]
X = X0.dropna(subset=["dswe_ea"]).copy()
for c in cols:
    X[c] = X[c] - X.groupby(["cfg", "pc", "tie", "day"])[c].transform("mean")
    X[c] = X[c] / X.groupby(["cfg", "pc", "tie"])[c].transform("std").replace(0, np.nan)
X = X.dropna(subset=cols)
t = pd.factorize(X.day)[0]
lines = []
for name, rhs in (("精度", ["mae", "rmse", "crps"]), ("精度+偏差", ["mae", "rmse", "crps", "bias"]), ("精度+偏差+事前DSWE", ["mae", "rmse", "crps", "bias", "dswe_ea"]),
                  ("精度+偏差+事后DSWE", ["mae", "rmse", "crps", "bias", "dswe"]), ("精度+事前DSWE", ["mae", "rmse", "crps", "dswe_ea"])):
    f = sm.OLS(X.cost, X[rhs]).fit(cov_type="hac-groupsum", cov_kwds={"time": t, "maxlags": 7})
    lines.append(f"{name}：R² {f.rsquared:.3f}；" + "，".join(f"{k} {f.params[k]:+.2f}（p={f.pvalues[k]:.1e}）" for k in rhs))
taus = []
for (c, pc, tie), g in X0.dropna(subset=["dswe_ea"]).groupby(["cfg", "pc", "tie"]):
    tot = g.groupby("model")[["cost", "mae", "crps", "dswe", "dswe_ea", "bias"]].sum()
    taus.append(dict(cfg=c, pc=pc, tie=tie, mae=kendalltau(tot.mae, tot.cost).statistic, crps=kendalltau(tot.crps, tot.cost).statistic,
                     dswe=kendalltau(tot.dswe, tot.cost).statistic, dswe_ea=kendalltau(tot.dswe_ea, tot.cost).statistic,
                     negbias=kendalltau(-tot.bias, tot.cost).statistic, best=tot.cost.idxmin(), best_ea=tot.dswe_ea.idxmin(), best_mae=tot.mae.idxmin()))
T = pd.DataFrame(taus)
lines.append("整期九模型排序的 Kendall τ（与成本）：中位 MAE {:.2f}，CRPS {:.2f}，事后 DSWE {:.2f}，事前 DSWE {:.2f}，−偏差 {:.2f}；事前 DSWE 选中成本最低模型 {}/{} 次，MAE {}/{} 次".format(
    T.mae.median(), T.crps.median(), T.dswe.median(), T.dswe_ea.median(), T.negbias.median(), int((T.best == T.best_ea).sum()), len(T), int((T.best == T.best_mae).sum()), len(T)))
lines.append(T.round(2).to_string())
txt = "\n".join(lines)
(OUT / "ana_bl_prices2.txt").write_text(txt, encoding="utf-8")
print(txt)
