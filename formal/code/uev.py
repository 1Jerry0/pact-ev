"""UrbanEV 决策与区特征（E6、P1 横截面）。用法：python uev.py
所有区统一采用上海 A1 站的分时电度电价（同月）；储能 E ∈ {10%, 25%, 50%} × 日均电量，接入上限取调参期
（2022-09–10）日峰值中位数，λ = 3，中点选解，现规则执行；申报 τ ∈ {0.67, 0.91}、零容差（校准后分位数）。"""
import numpy as np
import pandas as pd
import cfg
import data as D
import sim
from util import log, pmap, save, done

MODELS = cfg.UEV_MODELS_STAT + cfg.UEV_MODELS_TSFM + ["combo5"]


def zone_features(y):
    dev = y.loc[:cfg.UEV_DEV_END]
    hp = dev.groupby(dev.index.hour)
    return dict(cv_hour=float((hp.std() / hp.mean().replace(0, np.nan)).mean()), acf24=float(dev.autocorr(24)),
                zero_share=float((dev == 0).mean()), mean_kwh=float(dev.resample("D").sum().mean()))


def task(zone, y, arr, days):
    dev = y.loc[:cfg.UEV_DEV_END]
    dmean = float(dev.resample("D").sum().mean())
    peak = float(dev.resample("D").max().quantile(cfg.UEV_LQ))
    sc = dict(dmean=dmean, peaks={cfg.UEV_LQ: peak})
    rows, feats = [], []
    for d in days:
        idx = pd.date_range(d, periods=24, freq="h")
        yv = y.reindex(idx).to_numpy()
        if not np.isfinite(yv).all() or any(d not in arr[m] for m in MODELS):
            continue
        price = D.price("A1", cfg.UEV_PRICE_MONTH[d.month])
        for e_ in cfg.UEV_E:
            P = sim.Problem("storage", yv, price, (e_, cfg.UEV_LQ, cfg.UEV_LAM), sc)
            W = P.dswe_weights() if e_ == 0.25 else None
            for m in MODELS:
                mean, Qr, Qc = arr[m][d]
                err = yv - mean
                r = dict(day=d, E=e_, model=m, regret=P.cost(P.plan(mean)) - P.opt, opt=P.opt)
                if W is not None:
                    r.update(dswe=sim.dswe(err, *W), conc=sim.concentration(*W))
                rows.append(r)
        for m in MODELS:
            mean, Qr, Qc = arr[m][d]
            err = yv - mean
            feats.append(dict(day=d, model=m, mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                              crps=float(sim.crps_q(yv, Qr).mean()), crps_cal=float(sim.crps_q(yv, Qc).mean()),
                              cov90=float(np.mean((yv >= Qr[:, 0]) & (yv <= Qr[:, -1]))),
                              cov90_cal=float(np.mean((yv >= Qc[:, 0]) & (yv <= Qc[:, -1]))),
                              decl67=float(sim.decl_cost(sim.qinterp(Qc, 0.67), yv, price, 0.67, 0).sum()),
                              decl91=float(sim.decl_cost(sim.qinterp(Qc, 0.91), yv, price, 0.91, 0).sum()),
                              decl91_point=float(sim.decl_cost(np.maximum(mean, 0), yv, price, 0.91, 0).sum()),
                              rough=float(np.abs(np.diff(mean)).sum()), kwh=float(yv.sum()),
                              covid=bool(cfg.UEV_COVID[0] <= d <= cfg.UEV_COVID[1])))
    save(pd.DataFrame(rows).assign(zone=zone), cfg.OUT / "uev" / "parts" / f"sched_{zone}.parquet")
    save(pd.DataFrame(feats).assign(zone=zone, **zone_features(y)), cfg.OUT / "uev" / "parts" / f"feat_{zone}.parquet")
    return zone


def run():
    (cfg.OUT / "uev" / "parts").mkdir(parents=True, exist_ok=True)
    V = D.uev_volume()
    days = D.days(cfg.UEV_EVAL_START, cfg.UEV_FC_END)
    F = {}
    for m in MODELS:
        F[m] = pd.read_parquet(cfg.OUT / "fc" / f"uev_{m}.parquet").set_index(["series", "t"]).sort_index()
        F[m + "_cal"] = pd.read_parquet(cfg.OUT / "fc" / f"uev_{m}_cal.parquet").set_index(["series", "t"]).sort_index()
    tasks = []
    for z in V.columns:
        if done(cfg.OUT / "uev" / "parts" / f"sched_{z}.parquet"):
            continue
        arr = {}
        for m in MODELS:
            g, gc = F[m].loc[z], F[m + "_cal"].loc[z]
            g = g[g.index >= cfg.UEV_EVAL_START]
            dl = pd.DatetimeIndex(g.index.normalize().unique())
            M = g["mean"].to_numpy().reshape(-1, 24)
            Q = g[cfg.QC].to_numpy().reshape(-1, 24, len(cfg.QS))
            Qc = gc.loc[g.index, cfg.QC].to_numpy().reshape(-1, 24, len(cfg.QS))
            arr[m] = {d: (M[i], Q[i], Qc[i]) for i, d in enumerate(dl)}
        tasks.append((z, V[z], arr, days))
    log(f"UrbanEV 决策任务 {len(tasks)} 个", "uev")
    pmap(task, tasks, desc="UrbanEV 决策", name="uev", batch_size=1)
    for t in ("sched", "feat"):
        parts = sorted((cfg.OUT / "uev" / "parts").glob(f"{t}_*.parquet"))
        save(pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True), cfg.OUT / "uev" / f"uev_{t}.parquet")
    log("UrbanEV 决策完成", "uev")


if __name__ == "__main__":
    run()
