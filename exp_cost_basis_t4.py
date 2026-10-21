"""利润口径下的计划选择补充（小时模型）：表 4 的对照计划、误差缩放、免赔额统计、最优面宽度与求解器默认解位置。
缺供按逐小时零售总价计（同 exp_cost_basis.py）；A8 以换电次数计，价格按次换算后比例不变，仍用 A8 自身的逐小时零售总价。
对照：求解器默认（HiGHS 对偶单纯形）、最小、中点、最大、内点法解、电价 ±1% 扰动后的唯一解、等成本面上充电功率峰值最小的计划。
误差缩放（A7、LightGBM）：f_k = y − k (y − f)，k = 0.8 / 0.5 / 0.2，默认与中点计划。
A7tight 为预先设定的 H4 检验网格（库存 2/4/8 小时 × 功率 1.1/1.3/1.6 倍）。
输出：结果/cb_t4_{A7|A8|A7tight}.parquet。用法：python exp_cost_basis_t4.py A7|A8|A7tight"""
import os
import pathlib
import sys
import zlib
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
import sim
from exp_cost_basis import POOL, CFGS, TIEBREAK, exec_split, mix
from exp_flat import plan_flat

OUT = ROOT / "results"
_LP = sim.linprog
SCALES = (0.8, 0.5, 0.2)


def plan_ipm(f, price, pen, cap, pmax):
    sim.linprog = lambda *a, **k: _LP(*a, **{**k, "method": "highs-ipm"})
    try:
        g, s, _, u = sim.bss_plan(f, price, pen, cap, pmax, "default")
    finally:
        sim.linprog = _LP
    return ("bss", g, s, f, u)


def plan(f, price, pen, cap, pmax, tie):
    g, s, _, u = sim.bss_plan(f, price, pen, cap, pmax, tie)
    return ("bss", g, s, f, u)


def buffers(pl, f):
    """计划在每小时满足预测需求后的余量 b_t = s_{t+1} − u_t（命题 3 的免赔额）。"""
    return pl[2][1:] - pl[4]


TIGHT = [(c, p) for c in (2, 4, 8) for p in (1.1, 1.3, 1.6)]     # 预先设定的 H4 检验网格（紧充电网格）


def task(station, d, y, price, pen, hmean, fcs, cfgs=None):
    rows = []
    z = np.random.default_rng(zlib.crc32(f"{station}{d.date()}".encode())).uniform(-1, 1, 24)
    for c, p in (cfgs or CFGS):
        cap, pmax = c * hmean, p * hmean
        _, _, obj, u0 = sim.bss_plan(y, price, pen, cap, pmax, None)
        opt = obj
        _, _, objn, _ = sim.bss_plan(y, price * (1 + 0.01 * z), pen, cap, pmax, None)
        P = sim.Problem("bss", y, price, (c, p, 1), dict(hmean=hmean, tot_max=1.0))
        P.pen = pen
        for m, f in fcs.items():
            pl = {t: plan(f, price, pen, cap, pmax, t) for t in ("default", "min", "max")}
            pl["mid"] = mix(pl["min"], pl["max"], 0.5)
            pl["ipm"] = plan_ipm(f, price, pen, cap, pmax)
            pl["flat"] = plan_flat(P, f)
            r = dict(station=station, day=d, cfg=f"({c}, {p})", model=m, dem=float(y.sum()))
            for t, x in pl.items():
                e, sh, pn = exec_split(x, y, price, pen, cap, pmax)
                r[f"reg_{t}"], r[f"S_{t}"] = e + pn - opt, sh
            # 电价扰动后的唯一解，在扰动电价下评价（与原分析相同）
            gn, sn, _, un = sim.bss_plan(f, price * (1 + 0.01 * z), pen, cap, pmax, "default")
            e, sh, pn = exec_split(("bss", gn, sn, f, un), y, price * (1 + 0.01 * z), pen, cap, pmax)
            r["reg_noise"] = e + pn - objn
            smin, smax = pl["min"][2].sum(), pl["max"][2].sum()
            r["width"] = (smax - smin) / (25 * cap)
            r["pos_default"] = (pl["default"][2].sum() - smin) / (smax - smin) if smax - smin > 1e-6 else np.nan
            ev = np.arange(24) >= 17
            for t in ("default", "mid"):
                b = buffers(pl[t], f)
                r[f"zero_eve_{t}"] = float((b[ev] <= 1e-6).mean())
                # 发生缺供的小时：逐小时执行
                s, short_h = cap / 2, np.zeros(24)
                gp, sp, dh_, up = pl[t][1], pl[t][2], pl[t][3], pl[t][4]
                for h in range(24):
                    g = float(np.clip((sp[h + 1] - s + max(dh_[h], 0) - up[h]) / sim.ETA, 0, min(pmax, (cap - s + max(dh_[h], 0)) / sim.ETA)))
                    avail = s + sim.ETA * g
                    served = min(y[h], avail)
                    short_h[h] = y[h] - served
                    s = min(avail - served, cap)
                sh_h = short_h > 1e-6
                r[f"short_hours_{t}"] = int(sh_h.sum())
                r[f"short_zero_{t}"] = int((sh_h & (b <= 1e-6)).sum())
            if m == "lgb_local" and station == cfg.BSS:
                for k in SCALES:
                    fk = np.maximum(y - k * (y - f), 0)
                    for t in ("default", "mid"):
                        if t == "mid":
                            x = mix(plan(fk, price, pen, cap, pmax, "min"), plan(fk, price, pen, cap, pmax, "max"), 0.5)
                        else:
                            x = plan(fk, price, pen, cap, pmax, "default")
                        e, sh, pn = exec_split(x, y, price, pen, cap, pmax)
                        r[f"sc{k}_{t}"] = e + pn - opt
            rows.append(r)
    return rows


def build(station, cfgs=None):
    H = D.mp_hourly()
    hmean = float(H[station].loc[:cfg.MP_DEV_END].mean())
    pst = cfg.PRICE_PROXY.get(station, station)
    G = {}
    for m in POOL:
        F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "mean"])
        G[m] = F[F.series == station].set_index("t")["mean"]
    end = cfg.MP_FC_END if station == cfg.BSS else pd.Timestamp("2024-10-31")
    T = []
    for d in pd.date_range(cfg.MP_EVAL_START, end, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[station].reindex(idx).to_numpy()
        fcs = {m: np.maximum(G[m].reindex(idx).to_numpy(), 0) for m in POOL}
        if np.isfinite(y).all() and all(np.isfinite(v).all() for v in fcs.values()):
            T.append((station, d, y, D.price(pst, d.month), D.price(pst, d.month, "total") * TIEBREAK, hmean, fcs, cfgs))
    return T


if __name__ == "__main__":
    tag = sys.argv[1]
    st = "A8" if tag == "A8" else cfg.BSS
    T = build(st, TIGHT if tag == "A7tight" else None)
    print(tag, len(T), "tasks", flush=True)
    res = Parallel(n_jobs=4, batch_size=2, verbose=2)(delayed(task)(*t) for t in T)
    pd.DataFrame([r for rr in res for r in rr]).to_parquet(OUT / f"cb_t4_{tag}.parquet")
    print("saved", flush=True)
