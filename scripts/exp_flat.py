"""再加一个"工程上常见"的打破平局规则作对照：在等成本最优面上最小化充电功率峰值（削峰 / 平滑充电）。
问题：如果实务中本来就会加平滑项，求解器默认解是不是稻草人？中点解相对这种"平滑计划"还有多少优势？
A7，LightGBM，论文代表性 9 组合 + 锚定 6 组合（罚价×3），全部评估日。只读冻结流程（LP 结构照抄 sim.bss_plan）。"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack
from joblib import Parallel, delayed
import cfg
import data as D
import sim
from analysis import boot_mean
from exp_bss import mix

OUT = HERE.parents[1] / "results"
ETA = cfg.ETA
S = cfg.BSS
GRID = list(cfg.BSS_REP) + [(c, p, 3) for c in (4, 6, 8) for p in (2.5, 4.0)]


def plan_flat(P, dhat):
    """先最小成本，再在成本不变下最小化充电功率峰值 max_t g_t。变量顺序同 sim.bss_plan，末尾加 z。"""
    dhat = np.maximum(np.asarray(dhat, float), 0)
    T = 24
    price, pen, cap, pmax = P.price, np.broadcast_to(np.asarray(P.pen, float), (T,)), P.cap, P.pmax
    s0 = st = cap / 2
    pv = price.min()
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
    bounds = [(0, pmax)] * T + [(0, cap)] * (T + 1) + [(0, dh) for dh in dhat] + [(0, None), (0, None)]
    r = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    tol = 1e-9 * max(1.0, abs(r.fun)) + 1e-7
    A_eq2 = hstack([A_eq, csr_matrix((T + 2, 1))]).tocsr()
    A_ub = vstack([hstack([csr_matrix(cost.reshape(1, -1)), csr_matrix((1, 1))]),
                   hstack([csr_matrix((np.ones(T), (np.arange(T), ig + np.arange(T))), shape=(T, n)), csr_matrix(-np.ones((T, 1)))])]).tocsr()
    b_ub = np.r_[r.fun + tol, np.zeros(T)]
    c2 = np.zeros(n + 1); c2[-1] = 1.0
    r2 = linprog(c2, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq2, b_eq=b_eq, bounds=bounds + [(0, None)], method="highs")
    x = r2.x[:n] if r2.status == 0 else r.x
    return ("bss", x[ig:ig + T], x[isx:isx + T + 1], dhat, x[iu:iu + T])


def task(d, y, price, tot, hmean, f):
    rows = []
    for par in GRID:
        P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
        lo, hi = P.plan(f, "min"), P.plan(f, "max")
        pf = plan_flat(P, f)
        smin, smax = lo[2].sum(), hi[2].sum()
        rows.append(dict(day=d, cfg=str(par), reg_default=P.cost(P.plan(f, "default")) - P.opt, reg_mid=P.cost(mix(lo, hi, 0.5)) - P.opt,
                         reg_flat=P.cost(pf) - P.opt, pos_flat=(pf[2].sum() - smin) / (smax - smin) if smax - smin > 1e-6 else np.nan))
    return rows


if __name__ == "__main__":
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    T = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y, f = H[S].reindex(idx).to_numpy(), np.maximum(G.reindex(idx).to_numpy(), 0)
        if np.isfinite(y).all() and np.isfinite(f).all():
            T.append((d, y, D.price(S, d.month), float(D.price("A7", d.month, "total").max()), hmean, f))
    R = pd.DataFrame([r for rr in Parallel(n_jobs=6, batch_size=4)(delayed(task)(*t) for t in T) for r in rr])
    R.to_parquet(OUT / "bss_flat.parquet")
    R["set"] = np.where(R.cfg.map(lambda s: eval(s)[1]) > 2, "anchor", "paper")
    for st, g in R.groupby("set"):
        tot = g[["reg_default", "reg_flat", "reg_mid"]].sum()
        dd = g.groupby("day")[["reg_default", "reg_flat", "reg_mid"]].sum()
        m1, c1, _ = boot_mean(1 - dd.reg_mid / dd.reg_flat)
        m2, c2, _ = boot_mean(1 - dd.reg_flat / dd.reg_default)
        print(f"[{st}] 总遗憾/默认：平滑 {tot.reg_flat / tot.reg_default:.3f}，中点 {tot.reg_mid / tot.reg_default:.3f}；平滑解在最优面的位置均值 {g.pos_flat.mean():.3f}")
        print(f"      中点 vs 平滑 日度降幅 {m1:.3f} [{c1[0]:.3f}, {c1[1]:.3f}]；平滑 vs 默认 {m2:.3f} [{c2[0]:.3f}, {c2[1]:.3f}]")
