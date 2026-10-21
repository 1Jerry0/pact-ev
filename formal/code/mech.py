"""E4 误差路径机制实验。用法：python mech.py
同一组样本外残差（主分析：5 成员组合）只在同一电价时段内交换（误差指标严格不变）。因素：
B 跨时段位置 × C 偏差 × D 幅度（作用于原始排列，27 组）；F 持续性（带偏好的约束交换链，低/中/高）；
G 执行方式（开环 / 现规则；代表性组合上每 4 天加滚动重优化）；E 资源组合（全网格）。
每天记录正对照（完全信息遗憾）与负对照（零容差申报成本在交换下的变动）。"""
import numpy as np
import pandas as pd
import cfg
import data as D
import sim
from dec_mp import scales
from util import log, pmap, save, done


def tou_blocks(price):
    return [np.where(price == v)[0] for v in np.unique(price)]


def chain(e, y, blocks, kappa, n, rng):
    ep, a = e.copy(), sim.acf1(e)
    bl = [b for b in blocks if len(b) > 1]
    w = np.array([len(b) for b in bl], float)
    w /= w.sum()
    out, step = [], 0
    while len(out) < n:
        b = bl[rng.choice(len(bl), p=w)]
        i, j = rng.choice(b, 2, replace=False)
        if y[i] - ep[j] >= 0 and y[j] - ep[i] >= 0:
            ep[i], ep[j] = ep[j], ep[i]
            a2 = sim.acf1(ep)
            if kappa == 0 or rng.random() < np.exp(min(0.0, kappa * (a2 - a))):
                a = a2
            else:
                ep[i], ep[j] = ep[j], ep[i]
        step += 1
        if step > cfg.MECH_BURN and step % cfg.MECH_THIN == 0:
            out.append(ep.copy())
        if step > 200000:
            break
    return out


def move_largest(e, price, to_high):
    ep = e.copy()
    i = int(np.argmax(ep))
    tgt = np.where(price == (price.max() if to_high else price.min()))[0]
    j = tgt[np.argmin(np.abs(ep[tgt]))]
    ep[i], ep[j] = ep[j], ep[i]
    return ep


def weights_two_modes(P):
    """同一组扰动计划在现规则与开环下的逐时敏感度。"""
    y = P.y
    delta = 0.1 * max(y.mean(), 1e-6)
    base = P.plan(y)
    b = {m: P.cost(base, mode=m) for m in ("current", "open")}
    W = {m: (np.zeros(24), np.zeros(24)) for m in ("current", "open")}
    for t in range(24):
        up = y.copy(); up[t] += delta
        dn = y.copy(); dn[t] = max(dn[t] - delta, 0)
        pu, pd_ = P.plan(up), P.plan(dn)
        for m in ("current", "open"):
            W[m][1][t] = (P.cost(pu, mode=m) - b[m]) / delta
            W[m][0][t] = (P.cost(pd_, mode=m) - b[m]) / max(y[t] - dn[t], 1e-9)
    return W


def task(station, day, y, mean, source, scale, seed, full):
    kind = "bss" if station == cfg.BSS else "storage"
    price = D.price(station, day.month)
    grid, rep = (cfg.BSS_GRID, cfg.BSS_REP) if kind == "bss" else (cfg.ST_GRID, cfg.ST_REP)
    if not full:
        grid = rep
    rng = np.random.default_rng(seed)
    e = y - mean
    samples = {"orig": [e]}
    for lev, kap in cfg.MECH_KAPPA.items():
        samples[lev] = chain(e, y, tou_blocks(price), kap, cfg.MECH_N_PER, rng)
    neg = max(abs(sim.decl_cost(y - k, y, price, 0.67, 0.0).sum() - sim.decl_cost(y - e, y, price, 0.67, 0.0).sum())
              for lev in samples for k in samples[lev])
    sc = dict(scale)
    if kind == "bss":
        sc["tot_max"] = float(D.price("A7", day.month, "total").max())
    do_mpc = (day.dayofyear % cfg.MECH_MPC_EVERY == 0)
    rows, ctrl = [], []
    for par in grid:
        P = sim.Problem(kind, y, price, par, sc)
        ctrl.append((day, str(par), P.cost(P.plan(y)) - P.opt, P.cost(P.plan(y), mode="open") - P.opt, neg))
        W = weights_two_modes(P) if full else None
        fc = []
        if full:
            for B in ("none", "high", "low"):
                eb = e if B == "none" else move_largest(e, price, B == "high")
                for C in (-0.05, 0.0, 0.05):
                    for Dsc in (0.8, 1.0, 1.2):
                        f = y - Dsc * eb + C * y.mean()
                        if (f >= 0).all():
                            fc.append(("struct", "orig", B, C, Dsc, y - f))
        for lev, smp in samples.items():
            for k in smp:
                fc.append(("perm", lev, "none", 0.0, 1.0, k))
        mpc_left = {lev: (cfg.MECH_MPC_N if lev != "orig" else 1) for lev in samples}
        for treat, lev, B, C, Dsc, k in fc:
            f = y - k
            pl = P.plan(f)
            h, K = P.recourse(pl)
            r = dict(day=day, cfg=str(par), source=source, treat=treat, level=lev, B=B, C=C, D=Dsc, acf=sim.acf1(k),
                     dmax=sim.dmax(k), defrec_cur=sim.deficit_rec(k, h, K), defrec_open=sim.deficit_rec(k, np.zeros(24), K),
                     reg_current=P.cost(pl) - P.opt, reg_open=P.cost(pl, mode="open") - P.opt, reg_mpc=np.nan)
            if W is not None:
                r["dswe_cur"] = sim.dswe(k, *W["current"])
                r["dswe_open"] = sim.dswe(k, *W["open"])
                r["conc"] = sim.concentration(*W["current"])
            if do_mpc and par in rep and treat == "perm" and mpc_left[lev] > 0:
                r["reg_mpc"] = P.mpc(f) - P.opt
                mpc_left[lev] -= 1
            rows.append(r)
    tag = f"{station}_{source}_{day.date()}"
    save(pd.DataFrame(rows).assign(station=station), cfg.OUT / "mech" / "parts" / f"mech_{tag}.parquet")
    save(pd.DataFrame(ctrl, columns=["day", "cfg", "pc_current", "pc_open", "neg_ctrl"]).assign(station=station, source=source),
         cfg.OUT / "mech" / "parts" / f"ctrl_{tag}.parquet")
    return tag


def run():
    (cfg.OUT / "mech" / "parts").mkdir(parents=True, exist_ok=True)
    H = D.mp_hourly()
    sc_all = scales(H)
    days = D.days(cfg.MP_EVAL_START, cfg.MP_FC_END)
    tasks = []
    for source, stations, every, full in [(cfg.MECH_SOURCE, cfg.MECH_STATIONS, 1, True)] + \
            [(s, ["A3", cfg.BSS], 4, False) for s in cfg.MECH_SENS_SOURCES]:
        F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{source}.parquet")
        for st in stations:
            if st not in cfg.MP_SERIES:
                continue
            g = F[F.series == st].set_index("t")
            for i, d in enumerate(days[::every]):
                if done(cfg.OUT / "mech" / "parts" / f"mech_{st}_{source}_{d.date()}.parquet"):
                    continue
                idx = pd.date_range(d, periods=24, freq="h")
                y = H[st].reindex(idx).to_numpy()
                mean = np.maximum(g.loc[idx, "mean"].to_numpy(), 0)
                if not np.isfinite(y).all():
                    continue
                tasks.append((st, d, y, mean, source, sc_all[st], cfg.SEED + 1000 * i + sum(map(ord, st + source)), full))
    log(f"E4 任务 {len(tasks)} 个", "mech")
    pmap(task, tasks, desc="E4", name="mech", batch_size=1)
    for t in ("mech", "ctrl"):
        parts = sorted((cfg.OUT / "mech" / "parts").glob(f"{t}_*.parquet"))
        if parts:
            save(pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True), cfg.OUT / "mech" / f"{t}.parquet")
    log("E4 完成", "mech")


if __name__ == "__main__":
    run()
