"""精读手册用的 4 小时玩具换电站：验证等成本最优面、求解器默认计划、误差价格及其分解。
设定：电价 [0.3, 0.3, 0.9, 0.9] 元/kWh；预测需求 [0, 0, 10, 10] kWh；库存容量 N=10，功率 P=10，η=1；
期初库存 0，期末库存 ≥ 0 且不计终值；缺供代价 κ=1.8 元/kWh（利润口径：峰段少收的零售价，取 A7 七月最高零售价 1.8）。第八稿版本。"""
import numpy as np
from scipy.optimize import linprog

p = np.array([0.3, 0.3, 0.9, 0.9])
N, P, KAPPA = 10.0, 10.0, 1.8
T = 4


def plan(dhat, method="highs-ds", tie=None):
    """变量 g(4) s(4 个期末库存 s1..s4) u(4)；min Σ p g + Σ κ u；s_t = s_{t-1} + g_t − (D̂_t − u_t)，s_0 = 0。"""
    n = 3 * T
    c = np.r_[p, np.zeros(T), np.full(T, KAPPA)]
    A_eq, b_eq = [], []
    for t in range(T):
        row = np.zeros(n)
        row[T + t] = 1                       # s_t
        if t > 0:
            row[T + t - 1] = -1              # −s_{t−1}
        row[t] = -1                          # −g_t
        row[2 * T + t] = -1                  # −u_t
        A_eq.append(row)
        b_eq.append(-dhat[t])
    bounds = [(0, P)] * T + [(0, N)] * T + [(0, d) for d in dhat]
    r = linprog(c, A_eq=np.array(A_eq), b_eq=b_eq, bounds=bounds, method=method)
    if tie is None:
        return r.x, r.fun
    # 字典序：先最优成本，再最小/最大总库存
    A2 = np.vstack([A_eq, c])
    b2 = np.r_[b_eq, r.fun + 1e-9]
    sec = np.r_[np.zeros(T), np.ones(T) * (1 if tie == "min" else -1), np.zeros(T)]
    r2 = linprog(sec, A_ub=c[None, :], b_ub=[r.fun + 1e-9], A_eq=np.array(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
    return r2.x, c @ r2.x


def execute(x, dhat, d):
    """跟踪执行：每小时充电使可用库存达到目标 A_t = s_t(计划期末) + D̂_t − u_t。"""
    g_p, s_p, u_p = x[:T], x[T:2 * T], x[2 * T:]
    s, cost, short = 0.0, 0.0, 0.0
    for t in range(T):
        target = s_p[t] + dhat[t] - u_p[t]
        g = float(np.clip(target - s, 0, min(P, N - s + dhat[t])))
        avail = s + g
        served = min(d[t], avail)
        u = d[t] - served
        s = min(avail - served, N)
        cost += p[t] * g + KAPPA * u
        short += u
    return cost, short


dhat = np.array([0, 0, 10, 10.0])
print("求解器默认（HiGHS 对偶单纯形）、内点法、最小缓冲、最大缓冲：")
for name, kw in [("对偶单纯形", dict(method="highs-ds")), ("内点法", dict(method="highs-ipm")),
                 ("最小缓冲", dict(tie="min")), ("最大缓冲", dict(tie="max"))]:
    x, f = plan(dhat, **kw)
    print(f"  {name}: 充电 g={np.round(x[:T], 2)}  期末库存 s={np.round(x[T:2*T], 2)}  计划成本={f:.2f}")
xs = {}
for a in (0.0, 0.5, 1.0):
    xmin, _ = plan(dhat, tie="min")
    xmax, _ = plan(dhat, tie="max")
    xs[a] = (1 - a) * xmin + a * xmax
d_true = np.array([0, 0, 12, 10.0])
opt_true = plan(d_true)[1]
opt_fc = plan(dhat)[1]
print(f"\n实际需求 {d_true}；完全信息最优成本 C*(y) = {opt_true:.2f}；按预测需求的最优成本 C*(y−δ) = {opt_fc:.2f}")
lam = (opt_true - opt_fc) / 2
for a, x in xs.items():
    c, sh = execute(x, dhat, d_true)
    regret = c - opt_true
    r_u = (c - opt_fc) / 2
    print(f"  α={a}: 计划 g={np.round(x[:T], 1)}，第 3 小时后计划库存 {x[T+2]:.1f}；执行成本 {c:.2f}，缺供 {sh:.1f}，"
          f"遗憾 {regret:.2f}；低估价格 π={regret/2:.2f} = 补救成本 r={r_u:.2f} − 影子价格 λ*={lam:.2f}")
# 高估：实际 [0,0,8,10]
d_over = np.array([0, 0, 8, 10.0])
opt_over = plan(d_over)[1]
for a, x in xs.items():
    c, sh = execute(x, dhat, d_over)
    print(f"  高估 2 kWh，α={a}: 执行成本 {c:.2f}，完全信息 {opt_over:.2f}，遗憾 {c-opt_over:.2f}，高估价格 {(c-opt_over)/2:.2f}")
