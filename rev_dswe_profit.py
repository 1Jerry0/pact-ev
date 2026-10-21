"""误差价格与 DSWE 在利润口径下重算（A7，数据锚定网格 6 种资源配置）。
缺供按逐小时零售总价计少收收入（同 exp_cost_basis.py），其余设定同 rev_dswe.py：中点 / 求解器默认计划下的逐时有限差分价格，
9 个模型的事后与事前 DSWE，与 MAE / RMSE / CRPS 一起回归同日跨模型遗憾（遗憾取自 cb_hourly_plans.parquet）。
输出：结果/rev_dswe_profit{_tight}.txt、rev_price_profile_profit{_tight}.csv。用法：python rev_dswe_profit.py（环境变量 GRID=tight 时用预先设定的紧充电网格）"""
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
from exp_cost_basis import POOL, CFGS, kappa_profit

OUT = HERE.parents[1] / "results"
S = cfg.BSS
TIGHT = os.environ.get("GRID") == "tight"          # 预先设定的 H3 检验网格（紧充电网格），遗憾取自 cb_t4_A7tight.parquet
if TIGHT:
    CFGS = [(c, p) for c in (2, 4, 8) for p in (1.1, 1.3, 1.6)]
SUF = "_tight" if TIGHT else ""


def task(d, y, price, hmean):
    out = []
    for c, p in CFGS:
        P = sim.Problem("bss", y, price, (c, p, 1), dict(hmean=hmean, tot_max=1.0))
        P.pen = kappa_profit(d.month)
        P.opt = P.optimum(y)
        wu, wo = P.dswe_weights("current", "mid")
        wu0, wo0 = P.dswe_weights("current", "default")
        out.append((d, f"({c}, {p})", wu, wo, wu0, wo0))
    return out


if __name__ == "__main__":
    lines = []

    def say(s):
        lines.append(s)
        print(s)

    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    fp = OUT / f"rev_prices_profit{SUF}.pkl"
    if not fp.exists():
        T = []
        for d in pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D"):
            y = H[S].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
            if np.isfinite(y).all():
                T.append((d, y, D.price(S, d.month), hmean))
        res = Parallel(n_jobs=6, batch_size=2, verbose=2)(delayed(task)(*t) for t in T)
        pd.DataFrame([r for rr in res for r in rr], columns=["day", "cfg", "wu", "wo", "wu0", "wo0"]).to_pickle(fp)
    W = pd.read_pickle(fp)
    E = W[W.day >= cfg.MP_EVAL_START]
    prof = {k: np.mean(np.vstack(E[k].to_list()), axis=0) for k in ("wu", "wo", "wu0", "wo0")}
    pd.DataFrame(prof).to_csv(OUT / f"rev_price_profile_profit{SUF}.csv")
    tou = np.mean([D.price(S, m) for m in range(4, 13)])
    say(f"平均误差价格（元/kWh）：低估 默认 {prof['wu0'].mean():.3f} / 中点 {prof['wu'].mean():.3f}；高估 默认 {prof['wo0'].mean():.3f} / 中点 {prof['wo'].mean():.3f}；评估期平均电价 {tou:.3f}")
    say("低估价格按小时（默认）：" + str(np.round(prof["wu0"], 2).tolist()))
    say("低估价格按小时（中点）：" + str(np.round(prof["wu"], 2).tolist()))
    allw = np.vstack(E.wu0.to_list())
    conc = float(np.mean([np.sort(r)[::-1][:6].sum() / max(r.sum(), 1e-12) for r in allw if r.sum() > 0]))
    zero = float((np.vstack(E.wo0.to_list()) < 1e-9).mean())
    say(f"默认解：最贵 6 个小时承担的低估价格份额 {conc:.3f}；高估价格为 0 的小时比例 {zero:.3f}；低估/高估平均价格之比 {prof['wu0'].mean() / max(prof['wo0'].mean(), 1e-9):.1f}")

    if TIGHT:
        PL = pd.read_parquet(OUT / "cb_t4_A7tight.parquet")[["day", "cfg", "model", "reg_mid", "reg_default"]]
        PL = PL.rename(columns={"reg_mid": "mid", "reg_default": "default"})
    else:
        PL = pd.read_parquet(OUT / "cb_hourly_plans.parquet", filters=[("inp", "==", "mean"), ("tie", "in", ["mid", "default"])])
        PL = PL[PL.model.isin(POOL)]
        PL["regret"] = PL.E + PL.Pen - PL.E_opt - PL.Pen_opt
        PL = PL.pivot_table(index=["day", "cfg", "model"], columns="tie", values="regret").reset_index()
    Wd = {(r.day, r.cfg): (r.wu, r.wo) for r in W.itertuples(index=False)}
    rows = []
    for m in POOL:
        F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet")
        F = F[F.series == S].set_index("t")
        for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
            g = F.reindex(pd.date_range(d, periods=24, freq="h"))
            y, mu = g.y.to_numpy(), np.maximum(g["mean"].to_numpy(), 0)
            if not (np.isfinite(y).all() and np.isfinite(mu).all()):
                continue
            e = y - mu
            crps = float(sim.crps_q(y, g[cfg.QC].to_numpy()).mean())
            for c, p in CFGS:
                k = f"({c}, {p})"
                if (d, k) not in Wd:
                    continue
                wu, wo = Wd[(d, k)]
                past = [Wd[(d - pd.Timedelta(days=j), k)] for j in range(2, 30) if (d - pd.Timedelta(days=j), k) in Wd]
                xa = sim.dswe(e, np.mean([q[0] for q in past], axis=0), np.mean([q[1] for q in past], axis=0)) if len(past) >= 14 else np.nan
                rows.append((d, k, m, float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())), crps, sim.dswe(e, wu, wo), xa))
    X = pd.DataFrame(rows, columns=["day", "cfg", "model", "mae", "rmse", "crps", "dswe", "dswe_ante"]).merge(PL, on=["day", "cfg", "model"])
    X = X.rename(columns={"mid": "regret"}).dropna(subset=["regret", "dswe", "dswe_ante"])
    X.drop(columns=["default"]).assign(unit="A7").to_parquet(OUT / f"rev_dswe_profit_a7{SUF}.parquet")
    cols =["regret", "mae", "rmse", "crps", "dswe", "dswe_ante"]
    for c in cols:
        X[c] = X[c] - X.groupby(["cfg", "day"])[c].transform("mean")
        X[c] = X[c] / X.groupby("cfg")[c].transform("std").replace(0, np.nan)
    X = X.dropna(subset=cols)
    tcode = pd.factorize(X.day)[0]
    for name, rhs in (("通用指标", ["mae", "rmse", "crps"]), ("+事后 DSWE", ["mae", "rmse", "crps", "dswe"]), ("+事前 DSWE", ["mae", "rmse", "crps", "dswe_ante"])):
        fit = sm.OLS(X.regret, X[rhs]).fit(cov_type="hac-groupsum", cov_kwds={"time": tcode, "maxlags": 7})
        say(f"{name}: R2 = {fit.rsquared:.3f}；" + "，".join(f"{k} {fit.params[k]:.3f} (p={fit.pvalues[k]:.3f})" for k in rhs))
    (OUT / f"rev_dswe_profit{SUF}.txt").write_text("\n".join(lines), encoding="utf-8")
