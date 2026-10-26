"""补充对照（对照方案在运行前写定，见 PAPER_MAP.md）：日前计划加固定安全库存 s_t ≥ k·平均需求。
口径与 exp_cost_basis_t4.py 相同（A7、小时模型、利润口径、均值预测、6 种配置 × 9 个模型、跟踪执行）。
每个 k 下记录默认解与中点解的遗憾、缺供电量和购电费；k = 0 应与 cb_t4_A7 的 reg_default / reg_mid 一致。
输出：results/safety_stock_A7.parquet（或 _pilot）。用法：python exp_safety_stock.py [pilot]"""
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
from scipy.sparse import csr_matrix
import cfg
import sim
from exp_cost_basis import CFGS, exec_split, mix
from exp_cost_basis_t4 import build

OUT = ROOT / "results"
KS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)        # 安全库存，单位：开发期小时平均需求


def plan_floor(dhat, price, pen, cap, pmax, floor, tie):
    """sim.bss_plan 加 s_t ≥ floor（t = 1..T），其余逐行相同。"""
    ETA = sim.ETA
    dhat = np.maximum(np.asarray(dhat, float), 0)
    T = len(dhat)
    s0, st, pv = cap / 2, cap / 2, price.min()
    pen = np.broadcast_to(np.asarray(pen, float), (T,))
    n = 3 * T + 3
    ig, isx, iu, idf, isr = 0, T, 2 * T + 1, 3 * T + 1, 3 * T + 2
    cost = np.zeros(n)
    cost[ig:ig + T], cost[iu:iu + T] = price, pen
    cost[idf], cost[isr] = pv / ETA, -pv * ETA
    rr, cc, vv = [], [], []
    for t in range(T):
        rr += [t] * 4; cc += [isx + t + 1, isx + t, ig + t, iu + t]; vv += [1, -1, -ETA, -1]
    rr += [T, T + 1, T + 1, T + 1]; cc += [isx, isx + T, idf, isr]; vv += [1, 1, 1, -1]
    A_eq = csr_matrix((vv, (rr, cc)), shape=(T + 2, n))
    b_eq = np.r_[-dhat, s0, st]
    fl = min(floor, cap)
    bounds = ([(0, pmax)] * T + [(0, cap)] + [(fl, cap)] * T + [(0, dh) for dh in dhat] + [(0, None), (0, None)])
    x, obj = sim._with_tie(cost, None, None, A_eq, b_eq, bounds, slice(isx, isx + T + 1), tie, "max")
    return ("bss", x[ig:ig + T], x[isx:isx + T + 1], dhat, x[iu:iu + T]), obj


def task(station, d, y, price, pen, hmean, fcs, cfgs=None):
    rows = []
    for c, p in CFGS:
        cap, pmax = c * hmean, p * hmean
        _, _, opt, _ = sim.bss_plan(y, price, pen, cap, pmax, None)
        for m, f in fcs.items():
            for k in KS:
                floor = k * hmean
                pl = {"default": plan_floor(f, price, pen, cap, pmax, floor, "default")[0]}
                lo = plan_floor(f, price, pen, cap, pmax, floor, "min")[0]
                hi = plan_floor(f, price, pen, cap, pmax, floor, "max")[0]
                pl["mid"] = mix(lo, hi, 0.5)
                for t, x in pl.items():
                    e, sh, pn = exec_split(x, y, price, pen, cap, pmax)
                    rows.append(dict(day=d, cfg=f"({c}, {p})", model=m, k=k, tie=t, reg=e + pn - opt, S=sh, E=e, dem=float(y.sum())))
    return rows


if __name__ == "__main__":
    pilot = "pilot" in sys.argv[1:]
    T = build(cfg.BSS)
    if pilot:
        T = T[:10]
    print(len(T), "day tasks", flush=True)
    res = Parallel(n_jobs=6, batch_size=1, verbose=5)(delayed(task)(*t) for t in T)
    pd.DataFrame([r for rr in res for r in rr]).to_parquet(OUT / f"safety_stock_A7{'_pilot' if pilot else ''}.parquet")
    print("saved", flush=True)
