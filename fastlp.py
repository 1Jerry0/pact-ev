"""小时换电库存 LP 的快速求值：用 HiGHS 原生接口逐次冷启动（每次新建实例、默认选项、行列顺序与 SciPy 的调用相同），
复现冻结流程 sim.bss_plan 的四种选解（默认 / 最小 / 最大 / 中点库存），省掉 SciPy 的输入检查与包装开销。
为什么不用热启动：预测被放大到无法全部满足时计划缺口 u > 0，二阶段 LP 的最优解不再唯一，热启动与冷启动会落在不同的顶点上；
以遗憾为损失的训练会去"利用"求解路径的这种差别，所以训练用的求值必须与部署的管线逐例一致（见 __main__ 的核对，含放大的预测）。"""
import numpy as np
import highspy
from scipy.sparse import csc_matrix
import cfg
import sim

ETA = cfg.ETA
T = 24
INF = highspy.kHighsInf


def _run(c, Ac, rl, ru, cl, cu):
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(c), len(rl)
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, cl, cu
    lp.row_lower_, lp.row_upper_ = rl, ru
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = Ac.indptr, Ac.indices, Ac.data
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.run()
    ok = h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return np.array(h.getSolution().col_value), h.getInfo().objective_function_value, ok


class FastBss:
    def __init__(self, P):
        self.P = P
        n = 3 * T + 3
        self.n, self.isx, self.iu = n, T, 2 * T + 1
        idf, isr = 3 * T + 1, 3 * T + 2
        pv = P.price.min()
        c = np.zeros(n)
        c[:T], c[self.iu:self.iu + T] = P.price, P.pen
        c[idf], c[isr] = pv / ETA, -pv * ETA
        self.c = c
        A = np.zeros((T + 2, n))
        for t in range(T):
            A[t, self.isx + t + 1], A[t, self.isx + t], A[t, t], A[t, self.iu + t] = 1, -1, -ETA, -1
        A[T, self.isx] = 1
        A[T + 1, self.isx + T], A[T + 1, idf], A[T + 1, isr] = 1, 1, -1
        self.A1 = csc_matrix(A)                                   # 一阶段：只有等式行
        self.A2 = csc_matrix(np.vstack([c, A]))                   # 二阶段：成本上限行在前（SciPy 把 A_ub 排在 A_eq 之前）
        self.sec = np.zeros(n)
        self.sec[self.isx:self.isx + T + 1] = 1.0

    def plan(self, f, tie="mid"):
        P = self.P
        f = np.maximum(np.asarray(f, float), 0)
        cl = np.zeros(self.n)
        cu = np.r_[np.full(T, P.pmax), np.full(T + 1, P.cap), f, INF, INF]
        rb = np.r_[-f, P.cap / 2, P.cap / 2]
        x0, fstar, _ = _run(self.c, self.A1, rb, rb, cl, cu)
        if tie == "default":
            x = x0
        else:
            tol = 1e-9 * max(1.0, abs(fstar)) + 1e-7
            rl, ru = np.r_[-INF, rb], np.r_[fstar + tol, rb]
            xs = []
            for sign in ((-1.0, 1.0) if tie == "mid" else (-1.0,) if tie == "max" else (1.0,)):
                x2, _, ok = _run(sign * self.sec, self.A2, rl, ru, cl, cu)
                xs.append(x2 if ok else x0)
            x = np.mean(xs, axis=0)
        return x[:T], x[self.isx:self.isx + T + 1], x[self.iu:self.iu + T], f

    def regret(self, f, tie="mid"):
        g, s, u, fc = self.plan(f, tie)
        P = self.P
        return sim.bss_exec(g, s, fc, P.y, P.price, P.pen, P.cap, P.pmax, "current", u)[0] - P.opt


if __name__ == "__main__":
    import time
    import pandas as pd
    import data as D
    S = cfg.BSS
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    rng = np.random.default_rng(1)
    dif, t_fast, t_ref, k, nu = [], 0.0, 0.0, 0, 0
    for d in pd.date_range("2024-04-01", "2024-12-31", freq="9D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y, f = H[S].reindex(idx).to_numpy(), G.reindex(idx).to_numpy()
        if not (np.isfinite(y).all() and np.isfinite(f).all()):
            continue
        for par in ((4, 2.5, 1), (6, 2.5, 3), (8, 4.0, 10), (6, 4.0, 3), (4, 2.5, 10)):
            P = sim.Problem("bss", y, D.price(S, d.month), par, dict(hmean=hmean, tot_max=float(D.price("A7", d.month, "total").max())))
            Fb = FastBss(P)
            for j in range(8):
                ff = np.maximum(f * np.exp(rng.normal(0.0 if j < 3 else 0.6, 0.3 if j < 3 else 0.5, 24)), 0)     # 后 5 个把预测大幅放大，逼出计划缺口
                for tie in ("mid", "max", "min", "default"):
                    t0 = time.perf_counter(); a = Fb.regret(ff, tie); t1 = time.perf_counter(); b = P.regret(ff, tie); t2 = time.perf_counter()
                    dif.append(abs(a - b) / max(1.0, abs(b)))
                    t_fast += t1 - t0; t_ref += t2 - t1; k += 1
                nu += float(Fb.plan(ff, "max")[2].sum() > 1e-6)
    dif = np.array(dif)
    print(f"{k} 例（其中计划缺口 > 0 的预测 {int(nu)} 个）：相对差最大 {dif.max():.2e}，中位 {np.median(dif):.2e}，> 1e-6 的占比 {100 * (dif > 1e-6).mean():.2f}%；每例用时 快 {1000 * t_fast / k:.2f} ms，冻结 {1000 * t_ref / k:.2f} ms")
