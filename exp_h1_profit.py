"""研究假设 H1 在利润口径下重算（UrbanEV 275 个区，储能辅助充电）：与正式分析 P1 相同的设定——
储能容量为日均电量的 25%，接入上限取调参期日峰值中位数，中点计划、逐时纠偏执行，8 个模型的均值预测；
唯一的改动是超限负荷按削减充电、损失服务费计价：φ_t = A1 站同月零售总价 − 电度电价（UrbanEV 统一使用上海 A1 站的分时电价，与正文一致）。
统计量同 P1：每个区内，模型两两比较时“MAE 更小者遗憾也更小”的比例，对区特征（小时变异系数、24 小时自相关、零需求占比、对数日电量，标准化）回归，HC1 标准误。
MAE 与区特征取自正式输出 uev_feat.parquet（与计价无关）。输出：结果/h1_profit.parquet、h1_profit.txt。用法：python exp_h1_profit.py [run|ana]"""
import itertools
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
from exp_cost_basis_st import exec_split, fee

OUT = ROOT / "results"
MODELS = cfg.UEV_MODELS_STAT + cfg.UEV_MODELS_TSFM + ["combo5"]


def task(zone, y, means, days):
    dev = y.loc[:cfg.UEV_DEV_END]
    E = 0.25 * float(dev.resample("D").sum().mean())
    L = float(dev.resample("D").max().quantile(cfg.UEV_LQ))
    rows = []
    for d in days:
        yv = y.reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
        if not np.isfinite(yv).all() or any(d not in means[m] for m in MODELS):
            continue
        mo = cfg.UEV_PRICE_MONTH[d.month]
        price, phi = D.price("A1", mo), fee("A1", mo)
        _, _, obj = sim.st_plan(yv, price, E, L, phi, None)
        opt = obj + float(price @ yv)
        for m in MODELS:
            c, dd, _ = sim.st_plan(np.maximum(means[m][d], 0), price, E, L, phi, "mid")
            el, _, pn = exec_split(c, dd, yv, price, E, L, phi)
            rows.append((zone, d, m, el + pn - opt))
    return rows


def run():
    V = D.uev_volume()
    days = D.days(cfg.UEV_EVAL_START, cfg.UEV_FC_END)
    F = {m: pd.read_parquet(cfg.OUT / "fc" / f"uev_{m}.parquet", columns=["series", "t", "mean"]) for m in MODELS}
    T = []
    for z in V.columns:
        means = {}
        for m in MODELS:
            g = F[m][(F[m].series == z) & (F[m].t >= cfg.UEV_EVAL_START)].sort_values("t")
            dl = pd.DatetimeIndex(g.t.dt.normalize().unique())
            M = g["mean"].to_numpy().reshape(-1, 24)
            means[m] = {d: M[i] for i, d in enumerate(dl)}
        T.append((z, V[z], means, days))
    print(len(T), "zones", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 6)), batch_size=2, verbose=2)(delayed(task)(*t) for t in T)
    pd.DataFrame([r for rr in res for r in rr], columns=["zone", "day", "model", "regret"]).to_parquet(OUT / "h1_profit.parquet")


def ana():
    S0 = pd.read_parquet(OUT / "h1_profit.parquet")
    Fz0 = pd.read_parquet(cfg.OUT / "uev" / "uev_feat.parquet")
    txt = []
    for label, keep in (("全部评估日", lambda d: np.ones(len(d), bool)), ("剔除疫情调整期", lambda d: ~((d >= cfg.UEV_COVID[0]) & (d <= cfg.UEV_COVID[1])))):
        S = S0[keep(S0.day)]
        Fz = Fz0[keep(Fz0.day)]
        txt.append(f"## {label}\n" + one(S, Fz))
    (OUT / "h1_profit.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


def one(S, Fz):
    out = []
    for z, g in S.groupby("zone"):
        f = Fz[Fz.zone == z]
        pm = f.pivot(index="day", columns="model", values="mae")
        pv = g.pivot(index="day", columns="model", values="regret")
        ms = [m for m in pm.columns if m in pv.columns]
        ag = n = 0
        for a, b in itertools.combinations(ms, 2):
            dm, dv = pm[a] - pm[b], pv[a] - pv[b]
            ok = (dm.abs() > 1e-9) & (dv.abs() > 1e-9)
            ag += int((np.sign(dm[ok]) == np.sign(dv[ok])).sum())
            n += int(ok.sum())
        fz = f.iloc[0]
        out.append(dict(zone=z, agree=ag / n if n else np.nan, cv_hour=fz.cv_hour, acf24=fz.acf24, zero_share=fz.zero_share,
                        log_kwh=np.log(max(fz.mean_kwh, 1e-6))))
    Z = pd.DataFrame(out).dropna(subset=["agree"])
    X = Z[["cv_hour", "acf24", "zero_share", "log_kwh"]]
    X = sm.add_constant((X - X.mean()) / X.std())
    fit = sm.OLS(Z.agree, X, missing="drop").fit(cov_type="HC1")
    return (f"区层面一致率：中位数 {Z.agree.median():.3f}，均值 {Z.agree.mean():.3f}（四分位 {Z.agree.quantile(.25):.3f}—{Z.agree.quantile(.75):.3f}），区数 {len(Z)}\n"
            + fit.summary2().tables[1].round(4).to_string())


if __name__ == "__main__":
    (ana if (sys.argv[1:] or ["run"])[0] == "ana" else run)()
