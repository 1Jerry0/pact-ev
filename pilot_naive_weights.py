"""预检：DSWE 相对“简单成本加权误差”的增量解释力（A7，数据锚定网格，中点计划，利润口径）。
审稿人可能问：直接按电价或按缺供代价给误差加权是否就够了？这里比较三种不依赖计划的加权误差：
  TW  = Σ p_t |e_t|（按电度电价加权）；
  CW  = Σ (r_t e_t⁺ + p_t e_t⁻)（低估按零售价、高估按电价，相当于固定权重的非对称损失）；
  PW  = Σ (r_t − p_t) e_t⁺（只计低估的毛利）；
与随计划变化的 DSWE（rev_dswe_profit_a7.parquet）在同一回归框架下比较日内 R²。输出：结果/pilot_naive_weights.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
import cfg  # noqa: E402
import data as D  # noqa: E402

S = cfg.BSS
X = pd.read_parquet(ROOT / "results" / "rev_dswe_profit_a7.parquet")
rows = []
for m in X.model.unique():
    F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet")
    F = F[F.series == S].set_index("t")
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        g = F.reindex(pd.date_range(d, periods=24, freq="h"))
        y, mu = g.y.to_numpy(), np.maximum(g["mean"].to_numpy(), 0)
        if not (np.isfinite(y).all() and np.isfinite(mu).all()):
            continue
        p, r = D.price(S, d.month), D.price("A7", d.month, "total")
        e = y - mu
        up, dn = np.maximum(e, 0), np.maximum(-e, 0)
        rows.append((d, m, float(p @ np.abs(e)), float(r @ up + p @ dn), float((r - p) @ up)))
W = pd.DataFrame(rows, columns=["day", "model", "tw", "cw", "pw"])
X = X.merge(W, on=["day", "model"])
cols = ["regret", "mae", "rmse", "crps", "dswe", "dswe_ante", "tw", "cw", "pw"]
for c in cols:
    X[c] = X[c] - X.groupby(["cfg", "day"])[c].transform("mean")
    X[c] = X[c] / X.groupby("cfg")[c].transform("std").replace(0, np.nan)
X = X.dropna(subset=cols)
tcode = pd.factorize(X.day)[0]
lines = [f"A7 数据锚定网格，中点计划，{len(X)} 个模型×日×配置；日内标准化后回归，Driscoll–Kraay（7 阶）"]
base = ["mae", "rmse", "crps"]
for name, extra in (("精度指标", []), ("+ 电价加权误差 TW", ["tw"]), ("+ 非对称成本加权 CW", ["cw"]), ("+ 低估毛利加权 PW", ["pw"]),
                    ("+ 三种简单加权一起", ["tw", "cw", "pw"]), ("+ 事后 DSWE", ["dswe"]), ("+ 事前 DSWE", ["dswe_ante"]),
                    ("+ 三种简单加权 + 事后 DSWE", ["tw", "cw", "pw", "dswe"])):
    fit = sm.OLS(X.regret, X[base + extra]).fit(cov_type="hac-groupsum", cov_kwds={"time": tcode, "maxlags": 7})
    lines.append(f"{name}: R2 = {fit.rsquared:.3f}；" + "，".join(f"{k} {fit.params[k]:.3f} (p={fit.pvalues[k]:.3f})" for k in extra))
txt = "\n".join(lines)
(ROOT / "results" / "pilot_naive_weights.txt").write_text(txt, encoding="utf-8")
print(txt)
