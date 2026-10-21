"""求解器默认解落在最优面的哪里：同一批日前计划问题（A7，LightGBM 均值预测），换求解器、换算法、换等价写法。
位置 = (Σ库存 − Σ库存_min) / (Σ库存_max − Σ库存_min)，0 = 最小库存解（格的底），1 = 最大库存解（格的顶）；min / max 用论文里的二阶段 LP 求。
求解器：SciPy-HiGHS（论文所用；对偶单纯形 / 内点+交叉）、HiGHS 原生接口（原始单纯形、内点不交叉、PDLP）、OR-Tools 的 GLOP / CLP / PDLP、CBC。
写法：(F1) 论文的小时库存 LP；(F1r) 同一 LP，变量按时间倒序排列；(F1e) 消去库存变量（只留充电量与缺口，库存写成累计和的不等式）；
      (F2) 电池级 LP（exp_bl）。只统计目标值与最优值相差 < 1e-6（相对）的解。用法：python exp_solvers.py"""
import os
import pathlib
import sys
import warnings
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, csc_matrix, vstack, lil_matrix
from joblib import Parallel, delayed
import highspy
from ortools.linear_solver import pywraplp
import pulp
import cfg
import data as D
from exp_bl import mats, nsteps, load_events, NS

warnings.filterwarnings("ignore")
OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
ETA = cfg.ETA
INF = float("inf")


# ------------------------------------------------------------------ 通用 LP：min c x, A_ub x ≤ b_ub, A_eq x = b_eq, lo ≤ x ≤ hi
def rows_of(A_ub, b_ub, A_eq, b_eq):
    mats_, lo, hi = [], [], []
    if A_ub is not None:
        mats_.append(csr_matrix(A_ub)); lo += [-INF] * len(b_ub); hi += list(b_ub)
    if A_eq is not None:
        mats_.append(csr_matrix(A_eq)); lo += list(b_eq); hi += list(b_eq)
    return vstack(mats_).tocsr(), np.array(lo, float), np.array(hi, float)


def s_scipy(method):
    def f(c, A_ub, b_ub, A_eq, b_eq, bounds):
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
        return r.x if r.status == 0 else None
    return f


def s_highspy(**opts):
    def f(c, A_ub, b_ub, A_eq, b_eq, bounds):
        A, rl, ru = rows_of(A_ub, b_ub, A_eq, b_eq)
        Ac = csc_matrix(A)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        for k, v in opts.items():
            h.setOptionValue(k, v)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = len(c), A.shape[0]
        lp.col_cost_ = np.asarray(c, float)
        lp.col_lower_ = np.array([b[0] if b[0] is not None else -INF for b in bounds], float)
        lp.col_upper_ = np.array([b[1] if b[1] is not None else INF for b in bounds], float)
        lp.row_lower_, lp.row_upper_ = rl, ru
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = Ac.indptr, Ac.indices, Ac.data
        h.passModel(lp)
        h.run()
        return np.array(h.getSolution().col_value)
    return f


def s_ortools(name):
    def f(c, A_ub, b_ub, A_eq, b_eq, bounds):
        A, rl, ru = rows_of(A_ub, b_ub, A_eq, b_eq)
        sv = pywraplp.Solver.CreateSolver(name)
        xs = [sv.NumVar(b[0] if b[0] is not None else -sv.infinity(), b[1] if b[1] is not None else sv.infinity(), f"x{j}") for j, b in enumerate(bounds)]
        for i in range(A.shape[0]):
            ct = sv.Constraint(rl[i] if np.isfinite(rl[i]) else -sv.infinity(), ru[i] if np.isfinite(ru[i]) else sv.infinity())
            for j, v in zip(A.indices[A.indptr[i]:A.indptr[i + 1]], A.data[A.indptr[i]:A.indptr[i + 1]]):
                ct.SetCoefficient(xs[j], float(v))
        ob = sv.Objective()
        for j, v in enumerate(c):
            if v:
                ob.SetCoefficient(xs[j], float(v))
        ob.SetMinimization()
        st = sv.Solve()
        return np.array([x.solution_value() for x in xs]) if st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None
    return f


def s_cbc(c, A_ub, b_ub, A_eq, b_eq, bounds):
    A, rl, ru = rows_of(A_ub, b_ub, A_eq, b_eq)
    pr = pulp.LpProblem("p", pulp.LpMinimize)
    xs = [pulp.LpVariable(f"x{j:04d}", lowBound=b[0], upBound=b[1]) for j, b in enumerate(bounds)]
    pr += pulp.lpSum(float(v) * xs[j] for j, v in enumerate(c) if v)
    for i in range(A.shape[0]):
        e = pulp.lpSum(float(v) * xs[j] for j, v in zip(A.indices[A.indptr[i]:A.indptr[i + 1]], A.data[A.indptr[i]:A.indptr[i + 1]]))
        if rl[i] == ru[i]:
            pr += e == rl[i]
        else:
            pr += e <= ru[i]
    pr.solve(pulp.PULP_CBC_CMD(msg=0))
    return np.array([x.value() if x.value() is not None else 0.0 for x in xs]) if pr.status == 1 else None


SOLVERS = {
    "HiGHS 对偶单纯形（SciPy，论文所用）": s_scipy("highs"),
    "HiGHS 对偶单纯形（显式）": s_scipy("highs-ds"),
    "HiGHS 内点 + 交叉": s_scipy("highs-ipm"),
    "HiGHS 原始单纯形": s_highspy(solver="simplex", simplex_strategy=4),
    "HiGHS 内点，不交叉": s_highspy(solver="ipm", run_crossover="off"),
    "HiGHS 不预处理的对偶单纯形": s_highspy(solver="simplex", simplex_strategy=1, presolve="off"),
    "GLOP（OR-Tools）": s_ortools("GLOP"),
    "CLP（OR-Tools）": s_ortools("CLP"),
    "PDLP（一阶法）": s_ortools("PDLP"),
    "CBC（PuLP）": s_cbc,
}


# ------------------------------------------------------------------ 写法
def f1(dhat, price, pen, cap, pmax):
    """论文的小时库存 LP（sim.bss_plan 的矩阵）。返回 (c, A_ub, b_ub, A_eq, b_eq, bounds, 库存下标, 库存恢复函数)。"""
    T = 24
    n = 3 * T + 3
    ig, isx, iu, idf, isr = 0, T, 2 * T + 1, 3 * T + 1, 3 * T + 2
    pv = price.min()
    c = np.zeros(n)
    c[ig:ig + T], c[iu:iu + T] = price, pen
    c[idf], c[isr] = pv / ETA, -pv * ETA
    rr, cc, vv = [], [], []
    for t in range(T):
        rr += [t] * 4; cc += [isx + t + 1, isx + t, ig + t, iu + t]; vv += [1, -1, -ETA, -1]
    rr += [T, T + 1, T + 1, T + 1]; cc += [isx, isx + T, idf, isr]; vv += [1, 1, 1, -1]
    A_eq = csr_matrix((vv, (rr, cc)), shape=(T + 2, n))
    b_eq = np.r_[-dhat, cap / 2, cap / 2]
    bounds = [(0, pmax)] * T + [(0, cap)] * (T + 1) + [(0, dh) for dh in dhat] + [(0, None), (0, None)]
    return c, None, None, A_eq, b_eq, bounds, (lambda x: x[isx:isx + T + 1])


def f1r(dhat, price, pen, cap, pmax):
    """同一 LP，列按时间倒序排列（g_23…g_0, s_24…s_0, u_23…u_0, df, sr）。"""
    c, _, _, A_eq, b_eq, bounds, stock = f1(dhat, price, pen, cap, pmax)
    T = 24
    perm = np.r_[np.arange(T)[::-1], T + np.arange(T + 1)[::-1], 2 * T + 1 + np.arange(T)[::-1], 3 * T + 1, 3 * T + 2]
    inv = np.argsort(perm)
    return c[perm], None, None, A_eq[:, perm], b_eq, [bounds[j] for j in perm], (lambda x: stock(np.asarray(x)[inv]))


def f1e(dhat, price, pen, cap, pmax):
    """消去库存：变量 g(24) u(24) df sr；0 ≤ N/2 + Σ_{k<t}(η g_k − d_k + u_k) ≤ N；日末库存 + df − sr = N/2。"""
    T = 24
    n = 2 * T + 2
    pv = price.min()
    c = np.zeros(n)
    c[:T], c[T:2 * T] = price, pen
    c[2 * T], c[2 * T + 1] = pv / ETA, -pv * ETA
    L = np.tril(np.ones((T, T)))
    cum = np.hstack([ETA * L, L, np.zeros((T, 2))])               # s_{t+1} − N/2 + Σ d = 这一行 · x
    cd = np.cumsum(dhat)
    A_ub = np.vstack([cum, -cum])
    b_ub = np.r_[cap / 2 + cd, cap / 2 - cd]                       # s ≤ N；s ≥ 0
    A_eq = np.zeros((1, n))
    A_eq[0, :2 * T] = cum[-1, :2 * T]
    A_eq[0, 2 * T], A_eq[0, 2 * T + 1] = 1, -1
    b_eq = np.array([cd[-1]])
    bounds = [(0, pmax)] * T + [(0, dh) for dh in dhat] + [(0, None), (0, None)]
    return c, csr_matrix(A_ub), b_ub, csr_matrix(A_eq), b_eq, bounds, (lambda x: np.r_[cap / 2, cap / 2 + cum @ np.asarray(x) - cd])


def f2(dstep, price, pen, K, C, n, ebar):
    A, G, (ix, iR, iW, iu, idf, isr), nv = mats(n)
    ps = np.repeat(price, 4)
    c = np.zeros(nv)
    c[ix:ix + NS] = (ebar / ETA) * np.array([np.mean([ps[(s + k) % NS] for k in range(n)]) for s in range(NS)])
    c[iu:iu + NS] = np.repeat(pen, 4) * ebar
    lo = price.min()
    c[idf], c[isr] = lo / ETA * ebar, -lo * ETA * ebar
    b = np.zeros(2 * NS + 3)
    b[:NS], b[NS:2 * NS] = -dstep, dstep
    b[2 * NS], b[2 * NS + 1], b[2 * NS + 2] = K // 2, K - K // 2, K / 2
    bounds = [(0, None)] * NS + [(0, K)] * (2 * NS + 2) + [(0, max(v, 0.0)) for v in dstep] + [(0, None), (0, None)]
    return c, G, np.full(NS, float(C)), A, b, bounds, (lambda x: x[iR:iR + NS + 1])


def face(c, A_ub, b_ub, A_eq, b_eq, bounds, stock):
    """最优值，以及最优面上 Σ库存 的最小、最大值（HiGHS，两阶段）。"""
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    tol = 1e-9 * max(1.0, abs(r.fun)) + 1e-7
    n = len(c)
    # 库存是 x 的仿射函数：Σ库存 = w·x + 常数，用有限差分取 w
    base = stock(np.zeros(n)).sum()
    w = np.array([stock(np.eye(n)[j]).sum() - base for j in range(n)])
    row = csr_matrix(c.reshape(1, -1))
    A2 = vstack([csr_matrix(A_ub), row]) if A_ub is not None else row
    b2 = np.r_[b_ub, r.fun + tol] if b_ub is not None else np.array([r.fun + tol])
    lo = linprog(w, A_ub=A2, b_ub=b2, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    hi = linprog(-w, A_ub=A2, b_ub=b2, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return r.fun, stock(lo.x).sum(), stock(hi.x).sum()


def task(d, f, price, tot, hmean, ebar):
    rows = []
    insts = []
    for par in ((4, 2.5, 3), (6, 2.5, 3), (8, 4.0, 3)):
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        for nm, fn in (("F1 小时库存 LP", f1), ("F1r 同上，变量倒序", f1r), ("F1e 同上，消去库存变量", f1e)):
            insts.append((nm, str(par), fn(np.maximum(f, 0), price, pen, cap, pmax)))
    for par, pc in (((6, 2.5, 3), 12.0), ((6, 2.5, 3), 25.0)):
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)
        insts.append(("F2 电池级 LP", f"{par} {pc:.0f}kW", f2(np.repeat(np.maximum(f, 0) / ebar / 4, 4), price, pen, K, C, n, ebar)))
    for nm, cf, (c, A_ub, b_ub, A_eq, b_eq, bounds, stock) in insts:
        fstar, smin, smax = face(c, A_ub, b_ub, A_eq, b_eq, bounds, stock)
        if smax - smin < 1e-6 * max(1.0, abs(smax)):
            continue
        for sn, sv in SOLVERS.items():
            try:
                x = sv(c, A_ub, b_ub, A_eq, b_eq, bounds)
            except Exception:
                x = None
            if x is None:
                rows.append((d, nm, cf, sn, np.nan, np.nan))
                continue
            gap = (float(c @ x) - fstar) / max(1.0, abs(fstar))
            rows.append((d, nm, cf, sn, (stock(x).sum() - smin) / (smax - smin), gap))
    return rows


if __name__ == "__main__":
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="3D"):
        f = G.reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
        if np.isfinite(f).all():
            Tk.append((d, f, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, ebar))
    if len(sys.argv) > 1:
        Tk = Tk[:int(sys.argv[1])]
    res = Parallel(n_jobs=int(os.environ.get("NJ", 2)), batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk)
    R = pd.DataFrame([r for rr in res for r in rr], columns=["day", "form", "cfg", "solver", "pos", "gap"])
    R.to_parquet(OUT / "exp_solvers.parquet")
    ok = R[R.gap.abs() < 1e-6]
    lines = [f"{R.day.nunique()} 天（评估期每 3 天取 1 天）；位置 0 = 最小库存解，1 = 最大库存解；只统计达到最优值的解（相对差 < 1e-6）"]
    for form, g in R.groupby("form", sort=False):
        lines.append(f"== {form}（{g.cfg.nunique()} 组）")
        for sn in SOLVERS:
            a, b = g[g.solver == sn], ok[(ok.form == form) & (ok.solver == sn)]
            if len(b) == 0:
                lines.append(f"  {sn}：无达到最优值的解（{len(a)} 例，相对差中位 {a.gap.abs().median():.1e}）")
                continue
            lines.append(f"  {sn}：位置中位 {b.pos.median():.2f}，均值 {b.pos.mean():.2f}，四分位 [{b.pos.quantile(.25):.2f}, {b.pos.quantile(.75):.2f}]；在底部（<0.05）{100 * (b.pos < 0.05).mean():.0f}%，在顶部（>0.95）{100 * (b.pos > 0.95).mean():.0f}%；达到最优 {len(b)}/{len(a)}")
    txt = "\n".join(lines)
    (OUT / "exp_solvers.txt").write_text(txt, encoding="utf-8")
    print(txt)
