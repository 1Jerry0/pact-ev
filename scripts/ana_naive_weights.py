"""误差价格相对“简单成本加权误差”的增量解释力（利润口径；7 个储能站 9 种情景 + A7 数据锚定网格，中点计划）。
三种不随计划变化的加权误差：TW = Σ p_t|e_t|（按电度电价）；CW = Σ (r_t e_t⁺ + p_t e_t⁻)（低估按零售价、高估按电价）；PW = Σ (r_t − p_t) e_t⁺（低估按服务费/毛利）。
回归框架与 ana_h3_profit.py 相同（站点 × 配置 × 日内中心化并标准化，Driscoll–Kraay 7 阶）。
输入：结果/rev_dswe_st_profit.parquet、结果/rev_dswe_profit_a7.parquet。输出：结果/naive_weights_profit.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
import cfg  # noqa: E402
import data as D  # noqa: E402

OUT = ROOT / "results"
ST = pd.read_parquet(OUT / "rev_dswe_st_profit.parquet")
A7 = pd.read_parquet(OUT / "rev_dswe_profit_a7.parquet")
keep = ["unit", "day", "cfg", "model", "regret", "mae", "rmse", "crps", "dswe", "dswe_ante"]
X = pd.concat([ST[keep], A7[keep]], ignore_index=True)
rows = []
for (u, m), _ in X.groupby(["unit", "model"]):
    F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet")
    ser = cfg.BSS if u == "A7" else u                       # A7 的预测序列名为 cfg.BSS（A7_kwh）
    F = F[F.series == ser].set_index("t")
    for d in sorted(X.loc[(X.unit == u) & (X.model == m), "day"].unique()):
        d = pd.Timestamp(d)
        g = F.reindex(pd.date_range(d, periods=24, freq="h"))
        y, mu = g.y.to_numpy(), np.maximum(g["mean"].to_numpy(), 0)
        if not (np.isfinite(y).all() and np.isfinite(mu).all()):
            continue
        p, r = D.price(ser, d.month), D.price(u, d.month, "total")
        e = y - mu
        up, dn = np.maximum(e, 0), np.maximum(-e, 0)
        rows.append((u, d, m, float(p @ np.abs(e)), float(r @ up + p @ dn), float((r - p) @ up)))
W = pd.DataFrame(rows, columns=["unit", "day", "model", "tw", "cw", "pw"])
X = X.merge(W, on=["unit", "day", "model"])
cols = ["regret", "mae", "rmse", "crps", "dswe", "dswe_ante", "tw", "cw", "pw"]
lines = []


def fit(Z, label):
    Z = Z.copy()
    for c in cols:
        Z[c] = Z[c] - Z.groupby(["unit", "cfg", "day"])[c].transform("mean")
        Z[c] = Z[c] / Z.groupby(["unit", "cfg"])[c].transform("std").replace(0, np.nan)
    Z = Z.dropna(subset=cols)
    t = pd.factorize(Z.day)[0]
    base = ["mae", "rmse", "crps"]
    out = []
    for name, extra in (("精度", []), ("+TW", ["tw"]), ("+CW", ["cw"]), ("+PW", ["pw"]), ("+三种简单加权", ["tw", "cw", "pw"]),
                        ("+事后DSWE", ["dswe"]), ("+事前DSWE", ["dswe_ante"]), ("+三种简单加权+事后DSWE", ["tw", "cw", "pw", "dswe"]),
                        ("+三种简单加权+事前DSWE", ["tw", "cw", "pw", "dswe_ante"])):
        f = sm.OLS(Z.regret, Z[base + extra]).fit(cov_type="hac-groupsum", cov_kwds={"time": t, "maxlags": 7})
        s = f"{name} R2={f.rsquared:.3f}"
        if extra and extra[-1].startswith("dswe"):
            s += f"（{extra[-1]} 系数 {f.params[extra[-1]]:.3f}，p={f.pvalues[extra[-1]]:.1e}）"
        out.append(s)
    lines.append(f"{label}（{len(Z)} 个观测）：" + "；".join(out))
    print(lines[-1])


fit(X, "储能站 + A7 合并")
fit(X[X.unit != "A7"], "仅储能站")
for u in sorted(X.unit.unique()):
    fit(X[X.unit == u], u)
(OUT / "naive_weights_profit.txt").write_text("\n".join(lines), encoding="utf-8")
