"""决策层：申报成本、储能与换电库存的计划 LP、选解规则、三种执行方式、评分。
口径：遗憾 = 执行后的实际成本 − 完全信息 LP 最优值；规划与执行共用日末估值，正对照严格为 0。"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack
import cfg

ETA = cfg.ETA


# ---------------------------------------------------------------- 评分与申报
def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def crps_q(y, Q):
    """19 个分位数上平均分位损失的 2 倍（CRPS 的分位数近似），逐行返回。"""
    return 2 * np.mean([pinball(y, Q[:, i], t) for i, t in enumerate(cfg.QS)], axis=0)


def qinterp(Q, tau):
    return np.array([np.interp(tau, cfg.QS, r) for r in Q])


def decl_cost(q, y, p, tau, b):
    """C = α_u p [y − (1+b)q]^+ + α_o p [(1−b)q − y]^+，α_u + α_o = 2，τ = α_u / 2。"""
    return 2 * tau * p * np.maximum(y - (1 + b) * q, 0) + 2 * (1 - tau) * p * np.maximum((1 - b) * q - y, 0)


def _knots(Q):
    lo = np.maximum(Q[:, :1] - (Q[:, 1:2] - Q[:, :1]), 0)
    hi = Q[:, -1:] + (Q[:, -1:] - Q[:, -2:-1])
    return np.hstack([lo, Q, hi]), np.r_[0.0, cfg.QS, 1.0]


def _cdf(x, K, P):
    i = np.clip((K <= x[:, None]).sum(1) - 1, 0, K.shape[1] - 2)
    r = np.arange(len(x))
    k0, k1 = K[r, i], K[r, i + 1]
    w = np.where(k1 > k0, (x - k0) / np.where(k1 > k0, k1 - k0, 1), 1.0)
    F = P[i] + np.clip(w, 0, 1) * (P[i + 1] - P[i])
    return np.where(x < K[:, 0], 0.0, np.where(x >= K[:, -1], 1.0, F))


def declare_prob(Q, tau, b):
    """分段线性预测分布下的最优申报量（一阶条件二分求解，向量化）。b = 0 时即 τ 分位数。"""
    K, P = _knots(np.sort(np.maximum(Q, 0), axis=1))
    au, ao = 2 * tau, 2 * (1 - tau)
    lo, hi = np.zeros(len(Q)), K[:, -1] / max(1 - b, 1e-9) + 1e-9
    for _ in range(50):
        mid = (lo + hi) / 2
        g = ao * (1 - b) * _cdf((1 - b) * mid, K, P) - au * (1 + b) * (1 - _cdf((1 + b) * mid, K, P))
        hi, lo = np.where(g > 0, mid, hi), np.where(g > 0, lo, mid)
    return (lo + hi) / 2


def declare_samples(S, tau, b):
    """经验分布（每行一组样本）下的最优申报量。"""
    out = np.empty(len(S))
    for i, ys in enumerate(S):
        ys = np.maximum(ys, 0)
        cand = np.unique(np.r_[ys / (1 + b), ys / max(1 - b, 1e-9)])
        out[i] = cand[int(np.argmin(decl_cost(cand[:, None], ys[None, :], 1.0, tau, b).mean(1)))]
    return out


# ---------------------------------------------------------------- LP 求解与选解规则
def _solve(cost, A_ub, b_ub, A_eq, b_eq, bounds, sec=None):
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    if sec is None:
        return r.x, r.fun
    tol = 1e-9 * max(1.0, abs(r.fun)) + 1e-7
    row = csr_matrix(cost.reshape(1, -1))
    A2 = vstack([A_ub, row]) if A_ub is not None else row
    b2 = np.r_[b_ub, r.fun + tol] if b_ub is not None else np.array([r.fun + tol])
    r2 = linprog(sec, A_ub=A2, b_ub=b2, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return (r.x, r.fun) if r2.status != 0 else (r2.x, float(cost @ r2.x))


def _with_tie(cost, A_ub, b_ub, A_eq, b_eq, bounds, sl, tie, maximize_name):
    """tie：default 求解器默认解；'max' / 'min' 在最优面上最大化 / 最小化 Σ 状态（库存或 SOC）；
    'mid' 取二者中点（仍为最优解，位于最优面内部，主分析规则）。"""
    if tie in (None, "default"):
        return _solve(cost, A_ub, b_ub, A_eq, b_eq, bounds)

    def sec(sign):
        s = np.zeros(len(cost))
        s[sl] = sign
        return s
    if tie == "mid":
        x1, _ = _solve(cost, A_ub, b_ub, A_eq, b_eq, bounds, sec(-1.0))
        x2, _ = _solve(cost, A_ub, b_ub, A_eq, b_eq, bounds, sec(1.0))
        x = (x1 + x2) / 2
        return x, float(cost @ x)
    return _solve(cost, A_ub, b_ub, A_eq, b_eq, bounds, sec(-1.0 if tie == maximize_name else 1.0))


def terminal_value(s_end, s0, pv):
    """日末状态相对目标的估值：缺额按次日谷价补充（÷η），盈余按谷价折算（×η）。规划与执行共用。"""
    return (s0 - s_end) * pv / ETA if s_end < s0 else (s0 - s_end) * pv * ETA


# ---------------------------------------------------------------- 储能辅助充电站
def st_plan(yhat, price, E, L, lam, tie="mid", s0=None, s_target=None, pv=None):
    """视野长度 = len(yhat)。变量 c d s x def sur。返回 (c, d, LP 目标值不含常数项)。"""
    yhat = np.maximum(np.asarray(yhat, float), 0)
    T, P = len(yhat), E / 2
    s0 = E / 2 if s0 is None else s0
    st = E / 2 if s_target is None else s_target
    pv = price.min() if pv is None else pv
    n = 4 * T + 3
    ic, idd, isx, ix, idf, isr = 0, T, 2 * T, 3 * T + 1, 4 * T + 1, 4 * T + 2
    cost = np.zeros(n)
    cost[ic:ic + T], cost[idd:idd + T], cost[ix:ix + T] = price, -price, lam
    cost[idf], cost[isr] = pv / ETA, -pv * ETA
    r = np.repeat(np.arange(T), 3)
    c = np.column_stack([ic + np.arange(T), idd + np.arange(T), ix + np.arange(T)]).ravel()
    A_ub = csr_matrix((np.tile([1, -1, -1], T), (r, c)), shape=(T, n))
    b_ub = L - yhat
    rr, cc, vv = [], [], []
    for t in range(T):
        rr += [t] * 4; cc += [isx + t + 1, isx + t, ic + t, idd + t]; vv += [1, -1, -ETA, 1 / ETA]
    rr += [T, T + 1, T + 1, T + 1]; cc += [isx, isx + T, idf, isr]; vv += [1, 1, 1, -1]
    A_eq = csr_matrix((vv, (rr, cc)), shape=(T + 2, n))
    b_eq = np.r_[np.zeros(T), s0, st]
    bounds = ([(0, P)] * T + [(0, min(P, yh)) for yh in yhat] + [(0, E)] * (T + 1) + [(0, None)] * T
              + [(0, None), (0, None)])
    x, obj = _with_tie(cost, A_ub, b_ub, A_eq, b_eq, bounds, slice(isx, isx + T + 1), tie, "max")
    return x[ic:ic + T], x[idd:idd + T], obj


def _st_step(c_p, d_p, y, soc, E, L, recourse):
    P = E / 2
    c = min(c_p, (E - soc) / ETA)
    d = min(d_p, y, soc * ETA)
    net = y + c - d
    if recourse and net > L:
        if c > 0:
            cut = min(c, net - L)
            c -= cut
            net -= cut
        d += max(min(net - L, P - d, y - d, soc * ETA - d), 0)
        net = y + c - d
    return net, soc + ETA * c - d / ETA


def st_exec(c_plan, d_plan, y, price, E, L, lam, mode="current"):
    """执行成本（含与决策无关的 Σp·y）。mode：current 逐时纠偏；open 开环（只做物理截断）。"""
    soc, tot = E / 2, 0.0
    for t in range(24):
        net, soc = _st_step(c_plan[t], d_plan[t], y[t], soc, E, L, mode != "open")
        tot += price[t] * net + lam * max(net - L, 0)
    return tot + terminal_value(soc, E / 2, price.min())


def st_mpc(yhat, y, price, E, L, lam, tie="mid"):
    """滚动重优化：每小时用当前 SOC 与剩余小时的日前预测重新求解，执行第一小时决策（带逐时纠偏）。"""
    soc, tot, pv = E / 2, 0.0, price.min()
    for t in range(24):
        c, d, _ = st_plan(yhat[t:], price[t:], E, L, lam, tie, s0=soc, pv=pv)
        net, soc = _st_step(c[0], d[0], y[t], soc, E, L, True)
        tot += price[t] * net + lam * max(net - L, 0)
    return tot + terminal_value(soc, E / 2, pv)


def st_opt(y, price, E, L, lam):
    return st_plan(y, price, E, L, lam, None)[2] + float(np.dot(price, y))


# ---------------------------------------------------------------- 换电站电池库存
def bss_plan(dhat, price, pen, cap, pmax, tie="mid", s0=None, s_target=None, pv=None):
    """变量 g s u def sur；min Σ p g + Σ pen u + 日末估值；s_{t+1} = s_t + η g − (D − u)；0 ≤ u ≤ D。
    返回 (g, s, LP 目标值, 计划缺口 u)。"""
    dhat = np.maximum(np.asarray(dhat, float), 0)
    T = len(dhat)
    s0 = cap / 2 if s0 is None else s0
    st = cap / 2 if s_target is None else s_target
    pv = price.min() if pv is None else pv
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
    bounds = [(0, pmax)] * T + [(0, cap)] * (T + 1) + [(0, dh) for dh in dhat] + [(0, None), (0, None)]
    x, obj = _with_tie(cost, None, None, A_eq, b_eq, bounds, slice(isx, isx + T + 1), tie, "max")
    return x[ig:ig + T], x[isx:isx + T + 1], obj, x[iu:iu + T]


def bss_exec(g_plan, s_plan, dhat, d, price, pen, cap, pmax, mode="current", u_plan=None):
    """mode：current 状态反馈（按"计划库存轨迹 + 本小时计划净出库量（预测需求 − 计划缺口）"决定充电，
    缺口罚价低于尖峰充电成本时计划会主动缺货，执行规则据此不补充电）；open 按计划充电量执行。返回 (成本, 缺口)。"""
    s0 = cap / 2
    u_plan = np.zeros(24) if u_plan is None else u_plan
    s, tot, short = s0, 0.0, 0.0
    for t in range(24):
        dh = max(dhat[t], 0)
        g = g_plan[t] if mode == "open" else (s_plan[t + 1] - s + dh - u_plan[t]) / ETA
        g = float(np.clip(g, 0, min(pmax, (cap - s + dh) / ETA)))
        avail = s + ETA * g
        served = min(d[t], avail)
        u = d[t] - served
        s = min(avail - served, cap)
        tot += price[t] * g + pen_at(pen, t) * u
        short += u
    return tot + terminal_value(s, s0, price.min()), short


def bss_mpc(dhat, d, price, pen, cap, pmax, tie="mid"):
    s0, pv = cap / 2, price.min()
    s, tot = s0, 0.0
    for t in range(24):
        g = bss_plan(dhat[t:], price[t:], np.broadcast_to(np.asarray(pen, float), (24,))[t:], cap, pmax, tie, s0=s, s_target=s0, pv=pv)[0][0]
        dh = max(dhat[t], 0)
        g = float(np.clip(g, 0, min(pmax, (cap - s + dh) / ETA)))
        avail = s + ETA * g
        served = min(d[t], avail)
        s = min(avail - served, cap)
        tot += price[t] * g + pen_at(pen, t) * (d[t] - served)
    return tot + terminal_value(s, s0, pv)


def pen_at(pen, t):
    return float(np.broadcast_to(np.asarray(pen, float), (24,))[t])


def bss_opt(d, price, pen, cap, pmax):
    return bss_plan(d, price, pen, cap, pmax, None)[2]


# ---------------------------------------------------------------- 统一接口
class Problem:
    """把一个站点 × 日 × 资源组合封装成统一接口：plan(f, tie) / cost(plan, y, mode) / mpc(f) / opt。"""

    def __init__(self, kind, y, price, par, scale):
        self.kind, self.y, self.price = kind, np.asarray(y, float), price
        if kind == "storage":
            fe, fl, lam = par
            self.E, self.L, self.lam = fe * scale["dmean"], scale["peaks"][fl], lam
        else:
            fc_, fp, mult = par
            self.cap, self.pmax = fc_ * scale["hmean"], fp * scale["hmean"]
            # 缺口罚价 = 当日最高总价 × 倍数（不低于任何时段的补电成本，换电站有电池时不会主动拒绝服务）；
            # 加每小时 1e-5 的微小递减，使无法避免的缺口排在后面，与"有电池就先服务"的执行顺序一致
            self.pen = mult * scale["tot_max"] * (1 + 1e-5 * (23 - np.arange(24)))
        self.opt = self.optimum(self.y)

    def optimum(self, y):
        if self.kind == "storage":
            return st_opt(y, self.price, self.E, self.L, self.lam)
        return bss_opt(y, self.price, self.pen, self.cap, self.pmax)

    def plan(self, f, tie="mid"):
        f = np.maximum(np.asarray(f, float), 0)
        if self.kind == "storage":
            c, d, _ = st_plan(f, self.price, self.E, self.L, self.lam, tie)
            return ("st", c, d, f)
        g, s, _, u = bss_plan(f, self.price, self.pen, self.cap, self.pmax, tie)
        return ("bss", g, s, f, u)

    def cost(self, plan, y=None, mode="current"):
        y = self.y if y is None else np.asarray(y, float)
        if plan[0] == "st":
            return st_exec(plan[1], plan[2], y, self.price, self.E, self.L, self.lam, mode)
        return bss_exec(plan[1], plan[2], plan[3], y, self.price, self.pen, self.cap, self.pmax, mode, plan[4])[0]

    def mpc(self, f, tie="mid"):
        f = np.maximum(np.asarray(f, float), 0)
        if self.kind == "storage":
            return st_mpc(f, self.y, self.price, self.E, self.L, self.lam, tie)
        return bss_mpc(f, self.y, self.price, self.pen, self.cap, self.pmax, tie)

    def regret(self, f, tie="mid", mode="current"):
        if mode == "mpc":
            return self.mpc(f, tie) - self.opt
        return self.cost(self.plan(f, tie), mode=mode) - self.opt

    def recourse(self, plan):
        """逐时纠偏余量 h_t 与计划缓冲 K_t（扣除纠偏余量的累计缺口用）。"""
        if plan[0] == "st":
            c, d, f = plan[1], plan[2], plan[3]
            return (self.E / 2 - d) + c, np.maximum(self.L - (f + c - d), 0)
        g, s = plan[1], plan[2]
        return ETA * (self.pmax - g), s[1:]

    def dswe_weights(self, mode="current", tie="mid", rel=0.1):
        """完全信息问题上的逐时有限差分：低估、高估一个单位时遗憾的增量。"""
        y = self.y
        delta = rel * max(y.mean(), 1e-6)
        base = self.cost(self.plan(y, tie), mode=mode)
        wu, wo = np.zeros(24), np.zeros(24)
        for t in range(24):
            up = y.copy(); up[t] += delta
            dn = y.copy(); dn[t] = max(dn[t] - delta, 0)
            wo[t] = (self.cost(self.plan(up, tie), mode=mode) - base) / delta
            wu[t] = (self.cost(self.plan(dn, tie), mode=mode) - base) / max(y[t] - dn[t], 1e-9)
        return wu, wo


# ---------------------------------------------------------------- 路径特征
def dswe(e, wu, wo):
    return float((wu * np.maximum(e, 0) + wo * np.maximum(-e, 0)).sum())


def concentration(wu, wo, share=0.25):
    """权重集中度：权重最大的 25% 小时承担的权重比例。"""
    w = np.sort(np.maximum(wu, 0) + np.maximum(wo, 0))[::-1]
    return float(w[:max(1, int(round(share * len(w))))].sum() / max(w.sum(), 1e-12))


def dmax(e):
    best = cur = 0.0
    for v in e:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best


def deficit_rec(e, h, K):
    """扣除纠偏余量的累计缺口：x_t = max(x_{t−1} − h_t, 0) + e_t，取 max_t (x_t − K_t)^+。"""
    x, worst = 0.0, 0.0
    for t in range(len(e)):
        x = max(x - h[t], 0.0) + e[t]
        worst = max(worst, x - K[t])
    return max(worst, 0.0)


def acf1(e):
    return float(np.corrcoef(e[:-1], e[1:])[0, 1]) if np.std(e) > 1e-9 else 0.0
