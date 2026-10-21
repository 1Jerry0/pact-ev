"""MP-EVData 决策层（E2 申报、E5 调度、计划选择、执行方式、情景优化、DSWE）。用法：python dec_mp.py
任务粒度：站点 × 7 天；输出 out/dec/parts/{decl,sched,feat}_{站点}_{起始日}.parquet，可续跑。"""
import numpy as np
import pandas as pd
import cfg
import data as D
import sim
from util import log, pmap, save, done

ALL_MODELS = cfg.MP_MODELS_STAT + cfg.MP_MODELS_TSFM + ["combo5"]


def scales(H):
    dev = H.loc[:cfg.MP_DEV_END]
    out = {}
    for s in cfg.POWER + [cfg.BSS]:
        daily = dev[s].resample("D").sum()
        peaks = dev[s].resample("D").max()
        out[s] = dict(dmean=float(daily.mean()), hmean=float(dev[s].mean()),
                      peaks={q: float(peaks.quantile(q)) for q in (0.3, 0.5, 0.7)})
    return out


def load_fc():
    F = {}
    for m in ALL_MODELS:
        for v in ("", "_cal"):
            f = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}{v}.parquet")
            F[m + v] = f[f.series.isin(cfg.POWER + [cfg.BSS])]
    return F


def task(station, days, arr, hist, y_by_day, scale):
    """arr[m][d] = (mean, Q_raw, Q_cal)；hist[m][d] = 此前 ≤28 天的整日残差矩阵；y_by_day[d] = 实际 24 小时。"""
    kind = "bss" if station == cfg.BSS else "storage"
    grid, rep = (cfg.BSS_GRID, cfg.BSS_REP) if kind == "bss" else (cfg.ST_GRID, cfg.ST_REP)
    ties = ["default", "mid", "max"] if kind == "bss" else ["default", "mid"]
    decl, sched, feat = [], [], []
    for d in days:
        y = y_by_day[d]
        if y is None or not np.isfinite(y).all():
            continue
        p = D.price(station, d.month)
        # ---- 申报（常规站）
        if kind == "storage":
            for m in ALL_MODELS:
                mean, Qr, Qc = arr[m][d]
                for tau in cfg.TAUS:
                    for b in cfg.BANDS:
                        decl.append((d, m, "point", tau, b, sim.decl_cost(mean, y, p, tau, b).sum()))
                        decl.append((d, m, "prob_raw", tau, b, sim.decl_cost(sim.declare_prob(Qr, tau, b), y, p, tau, b).sum()))
                        decl.append((d, m, "prob_cal", tau, b, sim.decl_cost(sim.declare_prob(Qc, tau, b), y, p, tau, b).sum()))
        # ---- 调度
        sc = dict(scale)
        if kind == "bss":
            sc["tot_max"] = float(D.price("A7", d.month, "total").max())
        for par in grid:
            P = sim.Problem(kind, y, p, par, sc)
            wu, wo = P.dswe_weights("current", "mid")
            conc = sim.concentration(wu, wo)
            for m in ALL_MODELS:
                mean = arr[m][d][0]
                e = y - mean
                for tie in ties:
                    pl = P.plan(mean, tie)
                    r = P.cost(pl) - P.opt
                    sched.append((d, str(par), m, "point", tie, "current", r, P.opt))
                    if tie == "mid":
                        h, K = P.recourse(pl)
                        feat.append((d, str(par), m, sim.dswe(e, wu, wo), conc, sim.deficit_rec(e, h, K), sim.dmax(e),
                                     float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())),
                                     float(sim.crps_q(y, arr[m][d][1]).mean()), float(np.abs(np.diff(mean)).sum())))
                        if par in rep and m in cfg.SCHED_MODELS_REP:
                            sched.append((d, str(par), m, "point", "mid", "open", P.cost(pl, mode="open") - P.opt, P.opt))
                            sched.append((d, str(par), m, "point", "mid", "mpc", P.mpc(mean, "mid") - P.opt, P.opt))
            if par in rep:
                for m in cfg.SAA_MODELS:
                    mean, Qr, _ = arr[m][d]
                    H = hist[m].get(d)
                    cands = {}
                    for q in cfg.SAA_QUANTS:
                        f = mean if q == "mean" else sim.qinterp(Qr, q)
                        for tie in ties:
                            cands[(q, tie)] = P.plan(f, tie)
                    for (q, tie), pl in cands.items():
                        if q == 0.7 and tie == "mid":
                            sched.append((d, str(par), m, "q70", "mid", "current", P.cost(pl) - P.opt, P.opt))
                    if H is not None and len(H) >= 7:
                        Ys = np.maximum(mean[None, :] + H, 0)
                        score = {k: np.mean([P.cost(pl, ys) for ys in Ys]) for k, pl in cands.items()}
                        bq = min([k for k in score if k[1] == "default"], key=score.get)
                        bqt = min(score, key=score.get)
                        sched.append((d, str(par), m, "saa_q", "default", "current", P.cost(cands[bq]) - P.opt, P.opt))
                        sched.append((d, str(par), m, "saa_qt", bqt[1], "current", P.cost(cands[bqt]) - P.opt, P.opt))
    tag = f"{station}_{days[0].date()}"
    if decl:
        save(pd.DataFrame(decl, columns=["day", "model", "strategy", "tau", "b", "cost"]).assign(station=station),
             cfg.OUT / "dec" / "parts" / f"decl_{tag}.parquet")
    save(pd.DataFrame(sched, columns=["day", "cfg", "model", "strategy", "tie", "mode", "regret", "opt"]).assign(station=station),
         cfg.OUT / "dec" / "parts" / f"sched_{tag}.parquet")
    save(pd.DataFrame(feat, columns=["day", "cfg", "model", "dswe", "conc", "defrec", "dmax", "mae", "rmse", "crps", "rough"])
         .assign(station=station), cfg.OUT / "dec" / "parts" / f"feat_{tag}.parquet")
    return tag


def run():
    (cfg.OUT / "dec" / "parts").mkdir(parents=True, exist_ok=True)
    H = D.mp_hourly()
    sc_all = scales(H)
    F = load_fc()
    eval_days = D.days(cfg.MP_EVAL_START, cfg.MP_FC_END)
    tasks = []
    for s in cfg.POWER + [cfg.BSS]:
        arr = {m: {} for m in ALL_MODELS}
        hist = {m: {} for m in cfg.SAA_MODELS}
        for m in ALL_MODELS:
            g = F[m][F[m].series == s].sort_values("t")
            gc = F[m + "_cal"][F[m + "_cal"].series == s].set_index("t")
            dd = g.t.dt.normalize().to_numpy()
            M, Q = g["mean"].to_numpy().reshape(-1, 24), g[cfg.QC].to_numpy().reshape(-1, 24, len(cfg.QS))
            Qc = gc.loc[g.t, cfg.QC].to_numpy().reshape(-1, 24, len(cfg.QS))
            dlist = pd.DatetimeIndex(dd[::24])
            for i, d in enumerate(dlist):
                arr[m][d] = (M[i], Q[i], Qc[i])
            if m in cfg.SAA_MODELS:
                R = (g.y.to_numpy() - g["mean"].to_numpy()).reshape(-1, 24)
                for i, d in enumerate(dlist):
                    ok = [j for j, x in enumerate(dlist) if d - pd.Timedelta(days=2 + cfg.SCEN_WINDOW) < x <= d - pd.Timedelta(days=2)
                          and np.isfinite(R[j]).all()]
                    hist[m][d] = R[ok] if ok else None
        y_by_day = {d: H[s].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy() for d in eval_days}
        for k in range(0, len(eval_days), 7):
            chunk = eval_days[k:k + 7]
            if done(cfg.OUT / "dec" / "parts" / f"sched_{s}_{chunk[0].date()}.parquet"):
                continue
            sub = {m: {d: arr[m][d] for d in chunk} for m in ALL_MODELS}
            subh = {m: {d: hist[m].get(d) for d in chunk} for m in cfg.SAA_MODELS}
            tasks.append((s, list(chunk), sub, subh, {d: y_by_day[d] for d in chunk}, sc_all[s]))
    log(f"MP 决策任务 {len(tasks)} 个", "dec_mp")
    pmap(task, tasks, desc="MP 决策", name="dec_mp", batch_size=1)
    for t in ("decl", "sched", "feat"):
        parts = sorted((cfg.OUT / "dec" / "parts").glob(f"{t}_*.parquet"))
        if parts:
            save(pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True), cfg.OUT / "dec" / f"mp_{t}.parquet")
    log("MP 决策完成", "dec_mp")


if __name__ == "__main__":
    run()
