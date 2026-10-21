"""电价不是分段平的时候，"免费缓冲"还在不在？（A7 小时模型，LightGBM 均值预测，锚定 18 组，评估期全部天）
分时电价在时段内是平的，所以日前计划有一整个等成本面；若段内电价略有起伏（或是逐时变化的现货价），最优解唯一，面消失。
这里把"在最优面上选库存最多的解"推广成"在计划成本不超过最优值 (1+ε) 的近似最优集合里选库存最多的解"（ε-最大库存解），
以及它与最小库存最优解的中点（ε-中点解），考察三种电价：
  tou      原分时电价；
  slope5   每小时电价 = 分时电价 × (1 + ξ_t)，ξ_t ~ U(−5%, 5%)，逐日独立抽取（计划与结算都用这套电价）；
  slope20  同上，±20%。
报告：各解的总遗憾 / 该电价下默认解的总遗憾；实际付出的计划成本溢价；执行成本相对默认解的变化；最优面（或 ε-面）宽度。用法：python exp_eps.py"""
import os
import pathlib
import sys
import zlib
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + str(HERE.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack
from joblib import Parallel, delayed
import cfg
import data as D
import sim
from exp_packs_value import rel_ci

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
ETA = cfg.ETA
GRID = [(c, p, m) for c in (4, 6, 8) for p in (2.5, 4.0) for m in (1, 3, 10)]
EPS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
REG = {"tou": 0.0, "slope5": 0.05, "slope20": 0.20}
T = 24


def lp_parts(dhat, price, pen, cap, pmax):
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
    return c, A_eq, b_eq, bounds, (ig, isx, iu)


def task(d, y, f, price0, tot, hmean):
    rows = []
    rng = np.random.default_rng(zlib.crc32(str(d.date()).encode()))
    xi = rng.uniform(-1, 1, T)
    f = np.maximum(f, 0)
    for reg, amp in REG.items():
        price = price0 * (1 + amp * xi)
        for par in GRID:
            P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
            c, A_eq, b_eq, bounds, (ig, isx, iu) = lp_parts(f, price, P.pen, P.cap, P.pmax)
            r0 = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            fstar = r0.fun
            sec = np.zeros(len(c))
            sec[isx:isx + T + 1] = 1.0
            row = csr_matrix(c.reshape(1, -1))

            def edge(sign, eps):
                tol = 1e-9 * max(1.0, abs(fstar)) + 1e-7
                r = linprog(sign * sec, A_ub=row, b_ub=np.array([fstar + eps * abs(fstar) + tol]), A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
                return r.x if r.status == 0 else r0.x

            def ev(x):
                plan = ("bss", x[ig:ig + T], x[isx:isx + T + 1], f, x[iu:iu + T])
                cost = P.cost(plan)
                return cost - P.opt, cost, float(c @ x) / fstar - 1, float(x[isx:isx + T + 1].mean() / P.cap)
            xmin, xmax = edge(1.0, 0.0), edge(-1.0, 0.0)
            key = (d, reg, str(par))
            rows.append((*key, "default", *ev(r0.x)))
            rows.append((*key, "min", *ev(xmin)))
            rows.append((*key, "mid", *ev((xmin + xmax) / 2)))
            rows.append((*key, "max", *ev(xmax)))
            for e in EPS:
                xe = edge(-1.0, e)
                rows.append((*key, f"emax{e}", *ev(xe)))
                rows.append((*key, f"emid{e}", *ev((xmin + xe) / 2)))
            rows.append((*key, "opt", 0.0, P.opt, 0.0, 0.0))
    return rows


def analyse():
    R = pd.read_parquet(OUT / "exp_eps.parquet")
    R["mult"] = R.cfg.map(lambda s: eval(s)[2])
    lines = [f"{R.day.nunique()} 天 × {R.cfg.nunique()} 组；总遗憾 / 该电价下默认解的总遗憾；括号内：实际计划成本溢价（均值），执行成本相对默认解的变化"]
    tab = []
    for reg in REG:
        g = R[R.reg == reg]
        W = g.pivot_table(index=["day", "cfg"], columns="plan", values="stock")
        lines.append(f"== {reg}：最优面宽度（最大 − 最小库存解的平均库存，占库容）中位 {100 * (W['max'] - W['min']).median():.1f}%；默认解遗憾 / 完全信息成本 = "
                     f"{g[g.plan == 'default'].regret.sum() / g[g.plan == 'opt'].cost.sum():.3f}")
        for mult in (1, 3, 10, "all"):
            h = g if mult == "all" else g[g.mult == mult]
            M = h.groupby(["day", "plan"]).regret.sum().unstack("plan")
            C = h.groupby(["day", "plan"]).cost.sum().unstack("plan")
            Pm = h.groupby("plan").prem.mean()
            s = f"  罚价 {'合计' if mult == 'all' else '×' + str(mult)}："
            for pl in ["min", "mid", "max"] + [f"emid{e}" for e in EPS] + [f"emax{e}" for e in EPS]:
                ratio = M[pl].sum() / M["default"].sum()
                s += f" {pl} {ratio:.3f}（{100 * Pm[pl]:.2f}%，{100 * (C[pl].sum() / C['default'].sum() - 1):+.1f}%）"
                tab.append(dict(reg=reg, mult=mult, plan=pl, ratio=ratio, prem=100 * Pm[pl], dcost=100 * (C[pl].sum() / C["default"].sum() - 1)))
            lines.append(s)
        M = g.groupby(["day", "plan"]).regret.sum().unstack("plan")
        for a, b in (("mid", "default"), ("emid0.005", "default"), ("emid0.01", "default"), ("emax0.005", "default"), ("emid0.005", "mid")):
            m, lo, hi = rel_ci(M[a].to_numpy(), M[b].to_numpy())
            lines.append(f"  {a} vs {b}（全部罚价）：{100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    txt = "\n".join(lines)
    (OUT / "exp_eps.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(tab).to_csv(OUT / "exp_eps.csv", index=False, encoding="utf-8-sig")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y, f = H[S].reindex(idx).to_numpy(), G.reindex(idx).to_numpy()
        if np.isfinite(y).all() and np.isfinite(f).all():
            Tk.append((d, y, f, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean))
    if len(sys.argv) > 1:
        Tk = Tk[:int(sys.argv[1])]
    res = Parallel(n_jobs=int(os.environ.get("NJ", 2)), batch_size=2, verbose=1)(delayed(task)(*t) for t in Tk)
    pd.DataFrame([r for rr in res for r in rr], columns=["day", "reg", "cfg", "plan", "regret", "cost", "prem", "stock"]).to_parquet(OUT / "exp_eps.parquet")
    analyse()
