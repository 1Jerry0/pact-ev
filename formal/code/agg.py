"""E3 聚合：概率预测相对点预测的额外价值如何随聚合变化。用法：python agg.py
MP：出租车、公交、居民、全部 7 站四个组合；分布用此前 ≤28 天的整日残差向量（同一天各站一起抽取），
对照为各站残差日期独立打乱（去掉跨站相关）与单站分别申报。UrbanEV：随机抽取 1–275 个区的组合（每个规模 20 次），
画额外价值—聚合规模曲线。统一用 A1 电度电价。"""
import numpy as np
import pandas as pd
import cfg
import data as D
import sim
from util import log, pmap, save, done

GROUPS = {"taxi": ["A1", "A2"], "bus": ["A3", "A4"], "residential": ["A5", "A6"], "all7": cfg.POWER}
BANDS = [0.0, 0.03, 0.10]


def mp_day(d, Yd, Md, Rh, cols, model, rng_seed):
    """Yd、Md：(24, k) 实际与点预测；Rh：(n, 24, k) 此前整日残差向量。"""
    rng = np.random.default_rng(rng_seed)
    price = D.price("A1", d.month)
    rows = []
    joint = Md[None] + Rh
    perm = np.stack([Rh[rng.permutation(len(Rh)), :, j] for j in range(Rh.shape[2])], axis=2)
    indep = Md[None] + perm
    for gname, members in GROUPS.items():
        ix = [cols.index(s) for s in members if s in cols]
        if len(ix) < len(members):
            continue
        ya = Yd[:, ix].sum(1)
        pt = np.maximum(Md[:, ix], 0).sum(1)
        sj, si = np.maximum(joint[:, :, ix], 0).sum(2).T, np.maximum(indep[:, :, ix], 0).sum(2).T
        for tau in cfg.TAUS:
            for b in BANDS:
                sep_pt = sum(sim.decl_cost(np.maximum(Md[:, j], 0), Yd[:, j], price, tau, b).sum() for j in ix)
                sep_pr = sum(sim.decl_cost(sim.declare_samples(np.maximum(joint[:, :, j], 0).T, tau, b), Yd[:, j],
                                           price, tau, b).sum() for j in ix)
                rows.append(dict(day=d, model=model, group=gname, tau=tau, b=b, kwh=ya.sum(),
                                 agg_point=sim.decl_cost(pt, ya, price, tau, b).sum(),
                                 agg_prob=sim.decl_cost(sim.declare_samples(sj, tau, b), ya, price, tau, b).sum(),
                                 agg_prob_indep=sim.decl_cost(sim.declare_samples(si, tau, b), ya, price, tau, b).sum(),
                                 sep_point=sep_pt, sep_prob=sep_pr))
    return rows


def uev_day(d, yv, mean, Rh, sizes, draws, seed):
    rng = np.random.default_rng(seed)
    price = D.price("A1", cfg.UEV_PRICE_MONTH[d.month])
    nz = yv.shape[1]
    rows = []
    for size in sizes:
        for k in range(draws if size < nz else 1):
            ix = rng.choice(nz, size, replace=False) if size < nz else np.arange(nz)
            ya = yv[:, ix].sum(1)
            S = np.maximum(mean[None, :, ix] + Rh[:, :, ix], 0).sum(2).T
            pt = np.maximum(mean[:, ix], 0).sum(1)
            for tau in (0.83, 0.91):
                rows.append(dict(day=d, size=size, draw=k, tau=tau, kwh=ya.sum(),
                                 point=sim.decl_cost(pt, ya, price, tau, 0).sum(),
                                 prob=sim.decl_cost(sim.declare_samples(S, tau, 0), ya, price, tau, 0).sum()))
    return rows


def run():
    H = D.mp_hourly()
    days = D.days(cfg.MP_EVAL_START, cfg.MP_FC_END)
    fp = cfg.OUT / "agg" / "mp_agg.parquet"
    if not done(fp):
        tasks = []
        for model in ("combo5", "chronos2_cov", "lgb_local"):
            F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{model}.parquet")
            F = F[F.series.isin(cfg.POWER)]
            Mw = F.pivot(index="t", columns="series", values="mean")[cfg.POWER]
            Yw = H[cfg.POWER].reindex(Mw.index)
            Rw = (Yw - Mw)
            dl = pd.DatetimeIndex(Mw.index.normalize().unique())
            M = {d: Mw.loc[d:d + pd.Timedelta(hours=23)].to_numpy() for d in dl}
            R = {d: Rw.loc[d:d + pd.Timedelta(hours=23)].to_numpy() for d in dl}
            for i, d in enumerate(days):
                hist = [h for h in dl if d - pd.Timedelta(days=2 + cfg.SCEN_WINDOW) < h <= d - pd.Timedelta(days=2)
                        and np.isfinite(R[h]).all()]
                Yd = Yw.loc[d:d + pd.Timedelta(hours=23)].to_numpy()
                if len(hist) >= 7 and d in M and np.isfinite(Yd).all():
                    tasks.append((d, Yd, M[d], np.stack([R[h] for h in hist]), list(cfg.POWER), model, cfg.SEED + i))
        res = pmap(mp_day, tasks, desc="E3 MP 聚合", name="agg")
        save(pd.DataFrame([r for rr in res for r in rr]), fp)
    fp = cfg.OUT / "agg" / "uev_agg.parquet"
    if not done(fp):
        V = D.uev_volume()
        F = pd.read_parquet(cfg.OUT / "fc" / "uev_combo5.parquet")
        Mw = F.pivot(index="t", columns="series", values="mean")[list(V.columns)]
        Rw = V.reindex(Mw.index) - Mw
        dl = pd.DatetimeIndex(Mw.index.normalize().unique())
        nz = V.shape[1]
        sizes = sorted({s for s in (1, 2, 5, 10, 25, 50, 100, nz) if s <= nz})
        tasks = []
        for i, d in enumerate(D.days(cfg.UEV_EVAL_START, cfg.UEV_FC_END)):
            hist = [h for h in dl if d - pd.Timedelta(days=2 + cfg.SCEN_WINDOW) < h <= d - pd.Timedelta(days=2)]
            idx = pd.date_range(d, periods=24, freq="h")
            Rh = np.stack([Rw.loc[h:h + pd.Timedelta(hours=23)].to_numpy() for h in hist])
            tasks.append((d, V.reindex(idx).to_numpy(), Mw.loc[idx].to_numpy(), np.nan_to_num(Rh), sizes, 20, cfg.SEED + i))
        res = pmap(uev_day, tasks, desc="E3 UrbanEV 聚合", name="agg")
        save(pd.DataFrame([r for rr in res for r in rr]), fp)
    log("E3 完成", "agg")


if __name__ == "__main__":
    run()
