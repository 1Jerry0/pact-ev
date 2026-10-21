"""储能站的误差价格与 DSWE 在利润口径下重算：超限负荷按削减充电、损失服务费计价（φ_t = 零售总价 − 电度电价，同 exp_cost_basis_st.py）。
逐日、逐资源情景计算中点计划下的逐时有限差分价格（步长为小时平均需求的 10%，执行与评价同正式分析），
9 个模型的事后与事前 DSWE（事前 = D−29…D−2 的平均价格），与 MAE / RMSE / CRPS 一起回归同日跨模型遗憾（遗憾取自 cb_storage_sched.parquet）。
再与 A7（rev_dswe_profit.py）合并，报告各站与合并样本的日内 R²。输出：结果/rev_dswe_st_profit.txt。用法：python rev_dswe_st_profit.py"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + str(HERE.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
import cfg
import data as D
import sim
from exp_cost_basis_st import POOL, CFGS, fee, exec_split

OUT = ROOT / "results"


def weights(y, price, E, L, phi, tie="mid", rel=0.1):
    delta = rel * max(y.mean(), 1e-6)

    def cost(yhat):
        c, d, _ = sim.st_plan(yhat, price, E, L, phi, tie)
        el, _, pn = exec_split(c, d, y, price, E, L, phi)
        return el + pn
    base = cost(y)
    wu, wo = np.zeros(24), np.zeros(24)
    for t in range(24):
        up = y.copy(); up[t] += delta
        dn = y.copy(); dn[t] = max(dn[t] - delta, 0)
        wo[t] = (cost(up) - base) / delta
        wu[t] = (cost(dn) - base) / max(y[t] - dn[t], 1e-9)
    return wu, wo


def task(station, d, y, price, phi, scale):
    out = []
    for fe, fq in CFGS:
        wu, wo = weights(y, price, fe * scale["dmean"], scale["peaks"][fq], phi)
        out.append((station, d, f"({fe}, {fq})", wu, wo))
    return out


def regress(X, label, say):
    cols = ["regret", "mae", "rmse", "crps", "dswe", "dswe_ante"]
    X = X.copy()
    for c in cols:
        X[c] = X[c] - X.groupby(["unit", "cfg", "day"])[c].transform("mean")
        X[c] = X[c] / X.groupby(["unit", "cfg"])[c].transform("std").replace(0, np.nan)
    X = X.dropna(subset=cols)
    tcode = pd.factorize(X.day)[0]
    res = {}
    for name, rhs in (("精度指标", ["mae", "rmse", "crps"]), ("+事后DSWE", ["mae", "rmse", "crps", "dswe"]), ("+事前DSWE", ["mae", "rmse", "crps", "dswe_ante"])):
        fit = sm.OLS(X.regret, X[rhs]).fit(cov_type="hac-groupsum", cov_kwds={"time": tcode, "maxlags": 7})
        res[name] = fit
    f = res["+事后DSWE"]
    say(f"  {label}：R² 精度指标 {res['精度指标'].rsquared:.3f} → +事后 DSWE {f.rsquared:.3f} → +事前 DSWE {res['+事前DSWE'].rsquared:.3f}；"
        f"DSWE 系数 {f.params['dswe']:.3f}（p={f.pvalues['dswe']:.3f}），MAE {f.params['mae']:.3f}（p={f.pvalues['mae']:.3f}），RMSE p={f.pvalues['rmse']:.3f}")


if __name__ == "__main__":
    lines = []

    def say(s):
        lines.append(s)
        print(s)

    H = D.mp_hourly()
    dev = H.loc[:cfg.MP_DEV_END]
    fp = OUT / "rev_prices_st_profit.pkl"
    if not fp.exists():
        T = []
        for s in cfg.POWER:
            daily, peaks = dev[s].resample("D").sum(), dev[s].resample("D").max()
            scale = dict(dmean=float(daily.mean()), peaks={q: float(peaks.quantile(q)) for q in (0.3, 0.5, 0.7)})
            for d in pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D"):
                y = H[s].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
                if np.isfinite(y).all():
                    T.append((s, d, y, D.price(s, d.month), fee(s, d.month), scale))
        res = Parallel(n_jobs=int(os.environ.get("NJ", 8)), batch_size=4, verbose=2)(delayed(task)(*t) for t in T)
        pd.DataFrame([r for rr in res for r in rr], columns=["station", "day", "cfg", "wu", "wo"]).to_pickle(fp)
    W = pd.read_pickle(fp)
    E = W[W.day >= cfg.MP_EVAL_START]
    for s, g in E.groupby("station"):
        wu, wo = np.vstack(g.wu.to_list()), np.vstack(g.wo.to_list())
        say(f"{s}：平均低估价格 {wu.mean():.3f}，高估 {wo.mean():.3f} 元/kWh")
    Wd = {(r.station, r.day, r.cfg): (r.wu, r.wo) for r in W.itertuples(index=False)}
    SC = pd.read_parquet(OUT / "cb_storage_sched.parquet", filters=[("tie", "==", "mid")])
    SC["regret"] = SC.E + SC.Pen - SC.cost_opt
    rows = []
    for s in cfg.POWER:
        for m in POOL:
            F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet")
            F = F[F.series == s].set_index("t")
            for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
                g = F.reindex(pd.date_range(d, periods=24, freq="h"))
                y, mu = g.y.to_numpy(), np.maximum(g["mean"].to_numpy(), 0)
                if not (np.isfinite(y).all() and np.isfinite(mu).all()):
                    continue
                e = y - mu
                crps = float(sim.crps_q(y, g[cfg.QC].to_numpy()).mean())
                for fe, fq in CFGS:
                    k = f"({fe}, {fq})"
                    if (s, d, k) not in Wd:
                        continue
                    wu, wo = Wd[(s, d, k)]
                    past = [Wd[(s, d - pd.Timedelta(days=j), k)] for j in range(2, 30) if (s, d - pd.Timedelta(days=j), k) in Wd]
                    xa = sim.dswe(e, np.mean([q[0] for q in past], axis=0), np.mean([q[1] for q in past], axis=0)) if len(past) >= 14 else np.nan
                    rows.append((s, d, k, m, float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())), crps, sim.dswe(e, wu, wo), xa))
    X = pd.DataFrame(rows, columns=["unit", "day", "cfg", "model", "mae", "rmse", "crps", "dswe", "dswe_ante"])
    X = X.merge(SC[["station", "day", "cfg", "model", "regret"]].rename(columns={"station": "unit"}), on=["unit", "day", "cfg", "model"])
    X.to_parquet(OUT / "rev_dswe_st_profit.parquet")
    say("同日跨模型遗憾面板（标准化后回归，Driscoll–Kraay 标准误，滞后 7 阶）：")
    for s, g in X.groupby("unit"):
        regress(g, s, say)
    regress(X, "7 个储能站合并", say)
    (OUT / "rev_dswe_st_profit.txt").write_text("\n".join(lines), encoding="utf-8")
