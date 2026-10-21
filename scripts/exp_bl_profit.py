"""电池级仿真在利润口径下重算（A7，15 分钟、逐块电池、真实逐笔换电驱动；LP、仿真、规则与 exp_bl.py 相同）。
缺供按换电发生时所在小时的零售总价计少收收入（Asadi & Nurre Pinkley 2022；Widrick et al. 2018），
上海电价下取 A7 当月逐小时零售总价；其他电价形状下取“该电价 + A7 同小时服务费”。每次仿真分开记录电费（含日末估值）、缺供电量与缺供损失，
供服务水平口径（缺供 ≤ 需求的 1%，Sun, Tan & Tsang 2018）与权衡曲线使用。
模式：
  main    上海电价；库存 4/6 小时 × 单块功率 8/12/18/25/50 kW（总功率 2.5 倍平均需求）；规则 66 组 + 换下即充；
          LightGBM 均值（裕量 0/1/2 步）与 0.6—0.9 分位数、近期曲线（季节经验分位数的均值与分位数）、完全信息，各取最小/中点/最大（均值另含默认）。
  models  9 个模型的均值预测 × 最小/中点/最大/默认（6 小时 12/18/25 kW、4 小时 12 kW）。
  tariffs 广东、PG&E EV2-A、两段式电价（6 小时，12/25 kW）：规则、LightGBM、近期曲线、完全信息。
输出：结果/blp_{mode}_rule.parquet、blp_{mode}_fc.parquet。用法：python exp_bl_profit.py main|models|tariffs [天数]"""
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
from joblib import Parallel, delayed
import cfg
import data as D
from exp_bl import lp_bl, position, nsteps, load_events, NS
from exp_packs import TH
from exp_tariffs import price_of

OUT = ROOT / "results"
S = cfg.BSS
ETA = cfg.ETA
QL = (0.6, 0.7, 0.8, 0.9)
POOL = ["naive", "naive7", "seasonal_q", "lqr", "lgb_local", "lgb_global", "chronos2_cov", "timesfm3_cov", "combo5"]
TIEBREAK = 1 + 1e-5 * (23 - np.arange(24))
GRIDS = {"main": [((c, 2.5), pc) for pc in (8.0, 12.0, 18.0, 25.0, 50.0) for c in (4, 6)],
         "models": [((6, 2.5), 12.0), ((6, 2.5), 18.0), ((6, 2.5), 25.0), ((4, 2.5), 12.0)],
         "tariffs": [((6, 2.5), 12.0), ((6, 2.5), 25.0)]}
TARIFFS = ("guangdong", "pge_ev2a", "two_block")


def run_day_split(ev_step, ev_e, price, pen, K, C, pc, ebar, target):
    """与 exp_bl.run_day_bl 逐行相同，只是分开返回（电费 + 日末估值，未满足电量，缺供损失）。"""
    need = np.zeros(K)
    need[K // 2:] = ebar
    on = np.zeros(K, bool)
    add = ETA * pc * 0.25
    elec = unserved = pn = 0.0
    k = 0
    for step in range(NS):
        h = step // 4
        ready = need <= 1e-9
        on &= ~ready
        want = int(round(target[step])) - int(ready.sum())
        want = max(min(want, C, int((~ready).sum())), 0)
        n_on = int(on.sum())
        if want > n_on:
            idle = np.flatnonzero(~ready & ~on)
            on[idle[np.argsort(need[idle])[:want - n_on]]] = True
        elif want < n_on:
            act = np.flatnonzero(on)
            on[act[np.argsort(-need[act])[:n_on - want]]] = False
        e = np.minimum(need[on], add)
        need[on] -= e
        elec += price[h] * e.sum() / ETA
        while k < len(ev_step) and ev_step[k] == step:
            rd = np.flatnonzero(need <= 1e-9)
            if len(rd):
                need[rd[0]] = ev_e[k]
                on[rd[0]] = False
            else:
                unserved += ev_e[k]
                pn += pen[h] * ev_e[k]
            k += 1
    lo = price.min()
    gap = need.sum() - K * ebar / 2
    elec += gap * lo / ETA if gap > 0 else gap * lo * ETA
    return elec, unserved, pn


def task(mode, d, ev_step, ev_e, y, hmean, fcs, ebar, tariff="shanghai"):
    out_rule, rows = [], []
    if tariff == "shanghai":
        price = D.price(S, d.month)
        kap = D.price("A7", d.month, "total") * TIEBREAK
    else:
        price = price_of(tariff, d.month)
        kap = (price + D.price("A7", d.month, "total") - D.price("A7", d.month)) * TIEBREAK
    cls = np.repeat(np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6)), 4)
    ee = np.minimum(ev_e, 1.6 * ebar)
    cnt = np.bincount(ev_step, minlength=NS).astype(float)
    for (ch, pf), pc in GRIDS[mode]:
        cap, pmax = ch * hmean, pf * hmean
        K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)
        key = dict(tariff=tariff, day=d, cfg=f"({ch}, {pf})", pc=pc, n=n, dem=float(ee.sum()))
        sim = lambda tgt: run_day_split(ev_step, ee, price, kap, K, C, pc, ebar, tgt)
        if mode != "models":
            for i, (a, b) in enumerate(TH):
                e, u, p = sim(np.where(cls == 0, K, np.where(cls == 1, a * K, b * K)))
                out_rule.append(dict(key, th=i, E=e, S=u, Pen=p))
            e, u, p = sim(np.full(NS, K))
            out_rule.append(dict(key, th=-1, E=e, S=u, Pen=p))          # 换下即充
        if fcs is None:
            continue
        levels = {}
        for m, v in fcs.items():
            for nm, f in v.items():
                levels[(m, nm)] = np.repeat(np.maximum(f, 0) / ebar / 4, 4)
        if mode in ("main", "tariffs"):
            levels[("pi", "count")] = cnt
        for (m, nm), dd in levels.items():
            margins = (0, 1, 2) if (mode == "main" and nm in ("mean", "count")) else (0,)
            for mg in margins:
                npl = n + mg
                x0, f0 = lp_bl(dd, price, kap, K, C, npl, ebar)
                xl, _ = lp_bl(dd, price, kap, K, C, npl, ebar, 1.0, f0)
                xh, _ = lp_bl(dd, price, kap, K, C, npl, ebar, -1.0, f0)
                ties = [("min", xl), ("mid", (xl + xh) / 2), ("max", xh)] + ([("default", x0)] if nm in ("mean", "count") else [])
                for tie, x in ties:
                    e, u, p = sim(position(x, npl))
                    rows.append(dict(key, model=m, cand=f"{nm}@{mg}|{tie}", E=e, S=u, Pen=p,
                                     mae=float(np.abs(y - fcs[m]["mean"]).mean()) if m in fcs else np.nan))
    return out_rule, rows


def build(mode, ndays=None):
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    if mode == "main":
        pool, qs = ["lgb_local", "seasonal_q"], True
    elif mode == "models":
        pool, qs = POOL, False
    else:
        pool, qs = ["lgb_local", "seasonal_q"], True
    G = {}
    for m in pool:
        F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet")
        G[m] = F[F.series == S].set_index("t")
    start = cfg.MP_EVAL_START if mode == "models" else pd.Timestamp("2024-01-15")
    Tk = []
    for d in pd.date_range(start, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all() or d not in EV:
            continue
        fcs = None
        if d >= cfg.MP_EVAL_START:
            fcs = {}
            for m in pool:
                g = G[m].reindex(idx)
                v = {"mean": g["mean"].to_numpy()}
                if qs:
                    Q = g[cfg.QC].to_numpy()
                    v.update({f"q{q}": np.array([np.interp(q, cfg.QS, r) for r in Q]) for q in QL})
                if not all(np.isfinite(a).all() for a in v.values()):
                    fcs = None
                    break
                fcs[m] = v
            if fcs is None:
                continue
        for tf in (TARIFFS if mode == "tariffs" else ("shanghai",)):
            Tk.append((mode, d, *EV[d], y, hmean, fcs, ebar, tf))
    if ndays:
        Tk = [t for t in Tk if t[6] is not None][:ndays]
    return Tk


if __name__ == "__main__":
    mode = sys.argv[1]
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else None
    Tk = build(mode, nd)
    print(mode, len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=6, batch_size=1, verbose=2)(delayed(task)(*t) for t in Tk)
    suf = "_test" if nd else ""
    pd.DataFrame([r for b, _ in res for r in b]).to_parquet(OUT / f"blp_{mode}_rule{suf}.parquet")
    pd.DataFrame([r for _, rr in res for r in rr]).to_parquet(OUT / f"blp_{mode}_fc{suf}.parquet")
    print("saved", flush=True)
