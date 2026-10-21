"""稳健性：充电需要时间。聚合模型假设"本小时充进去的电本小时就能换出"；实际一块电池要充 1–2 小时以上。
这里把模型改成带提前期 L 的库存模型：第 t 小时开始充的电量 g_t 在第 t+L 小时才成为可换出的库存，
电费按充电期间 L 个小时的平均电价计，同时在充功率 Σ g/L ≤ P，库存 + 在充电量 ≤ 库容。
  计划：min Σ c_t g_t + Σ κ_t u_t + 日末估值（库存 + 在充电量 相对 N/2），等成本面上取 默认 / 最小 / 中点 / 最大库存解；
  执行：按"到 t+L 时的预计可用量"跟踪计划（用预测扣除提前期内的需求）——带提前期的补货到目标水平；
  规则：谷段把库存位置（库存 + 在充）补到满，平段 / 峰段补到阈值（阈值在 1–3 月选定），不用预测；
  基准：同一模型下的完全信息 LP 最优。
A7，LightGBM 单站，锚定组合（罚价 ×3 与 ×10），L ∈ {0, 1, 2}（L = 0 复现原模型的时序）。用法：python exp_leadtime.py"""
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
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from joblib import Parallel, delayed
import cfg
import data as D
from analysis import boot_mean

OUT = HERE.parents[1] / "results"
S = cfg.BSS
ETA = cfg.ETA
T = 24
GRID = [(c, p, m) for c in (4, 6, 8) for p in (2.5, 4.0) for m in (3, 10)]
LS = [int(x) for x in os.environ.get("LS", "0,1,2").split(",")]
SUF = os.environ.get("SUF", "")
LV = np.round(np.arange(0, 1.01, 0.1), 2)
TH = np.array([(a, b) for a in LV for b in LV if b <= a])


def coef(price, L):
    """第 t 小时开始充的 1 kWh 的电费：充电期间（L 小时，L=0 视同 1 小时）的平均电价（跨日用同小时电价）。"""
    w = max(L, 1)
    return np.array([np.mean([price[(t + k) % T] for k in range(w)]) for t in range(T)])


def lp(dhat, price, pen, cap, pmax, L, sec=None, fstar=None):
    """变量 g(24) s(25) u(24) def sur。返回 (x, 目标值)。sec=±1：在成本 ≤ fstar 的面上最小化 / 最大化 Σ s。"""
    n = 3 * T + 3
    ig, isx, iu, idf, isr = 0, T, 2 * T + 1, 3 * T + 1, 3 * T + 2
    pv = price.min()
    c = np.zeros(n)
    c[ig:ig + T], c[iu:iu + T] = coef(price, L), pen
    c[idf], c[isr] = pv / ETA, -pv * ETA
    Aeq, beq = lil_matrix((T + 2, n)), np.zeros(T + 2)
    for t in range(T):                                       # s_{t+1} = s_t + η g_{t−L} − (d_t − u_t)
        Aeq[t, isx + t + 1], Aeq[t, isx + t], Aeq[t, iu + t] = 1, -1, -1
        if t - L >= 0:
            Aeq[t, ig + t - L] = -ETA
        beq[t] = -dhat[t]
    Aeq[T, isx], beq[T] = 1, cap / 2
    Aeq[T + 1, isx + T], Aeq[T + 1, idf], Aeq[T + 1, isr] = 1, 1, -1     # 日末：库存 + 在充电量 + def − sur = cap/2
    for k in range(1, L + 1):
        Aeq[T + 1, ig + T - k] = ETA
    beq[T + 1] = cap / 2
    rows, b = [], []
    w = max(L, 1)
    for t in range(T):
        r = np.zeros(n)                                      # 在充功率
        for k in range(w):
            if t - k >= 0:
                r[ig + t - k] = 1.0 / w
        rows.append(r); b.append(pmax)
        if L > 0:
            r = np.zeros(n)                                  # 库存 + 本小时到货 + 在充电量 ≤ 库容（与执行时的上限一致）
            r[isx + t] = 1
            for k in range(0, L + 1):
                if t - k >= 0:
                    r[ig + t - k] = ETA
            rows.append(r); b.append(cap)
    bounds = [(0, None)] * T + [(0, cap)] * (T + 1) + [(0, dh) for dh in dhat] + [(0, None), (0, None)]
    Aub, bub = np.array(rows), np.array(b)
    if sec is None:
        r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
        return r.x, r.fun
    tol = 1e-9 * max(1.0, abs(fstar)) + 1e-7
    c2 = np.zeros(n); c2[isx:isx + T + 1] = sec
    r = linprog(c2, A_ub=np.vstack([Aub, c]), b_ub=np.r_[bub, fstar + tol], A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    return r.x, float(c @ r.x)


def unpack(x):
    return x[:T], x[T:2 * T + 1], x[2 * T + 1:3 * T + 1]


def execute(plan, dhat, y, price, pen, cap, pmax, L, rule=None):
    """按计划跟踪（rule=None）或阈值规则执行。返回 (成本, 缺口)。"""
    g_p, s_p, u_p = plan if plan is not None else (None, None, None)
    dhat = np.zeros(T) if dhat is None else dhat
    cg = coef(price, L)
    w = max(L, 1)
    cls = np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6))
    s, tot, short = cap / 2, 0.0, 0.0
    hist = []                                               # 已开始充电的 g
    for t in range(T):
        pipe = sum(hist[max(0, t - L + 1):t]) if L > 1 else 0.0          # 之前开始、尚未到货的（不含本小时到货的 g_{t−L}）
        arriving = hist[t - L] if (L > 0 and t - L >= 0) else 0.0
        draw_prev = sum(hist[max(0, t - w + 1):t]) / w
        gmax = max(min((pmax - draw_prev) * w, (cap - s - ETA * (pipe + arriving)) / ETA + (max(dhat[t], 0) if L == 0 else 0.0)), 0.0)
        if rule is not None:
            tgt = cap if cls[t] == 0 else (rule[0] * cap if cls[t] == 1 else rule[1] * cap)
            g = (tgt - s - ETA * (pipe + arriving)) / ETA
        elif L == 0:
            g = (s_p[t + 1] - s + max(dhat[t], 0) - u_p[t]) / ETA
        else:
            # 到 t+L 小时初的预计库存 = s + 到货 + 在途 − 期间预测净需求；目标 = 计划在 t+L 小时初的库存 + 该小时到货（即 η g_t 的去向）
            if t + L <= T - 1:
                target = s_p[t + L] + ETA * g_p[t]
                proj = s + ETA * (arriving + pipe) - sum(max(dhat[j], 0) - u_p[j] for j in range(t, t + L))
            else:                                           # 提前期跨过日末：按计划的日末库存位置
                target = s_p[T] + ETA * sum(g_p[max(T - L, 0):t + 1])
                proj = s + ETA * (arriving + pipe) - sum(max(dhat[j], 0) - u_p[j] for j in range(t, T))
            g = (target - proj) / ETA
        g = float(np.clip(g, 0.0, gmax))
        hist.append(g)
        avail = s + ETA * (g if L == 0 else arriving)
        served = min(y[t], avail)
        short += y[t] - served
        s = min(avail - served, cap)
        tot += cg[t] * g + pen[t] * (y[t] - served)
    pos = s + ETA * sum(hist[T - L:]) if L > 0 else s
    lo = price.min()
    term = (cap / 2 - pos) * lo / ETA if pos < cap / 2 else (cap / 2 - pos) * lo * ETA
    return tot + term, short


def task(d, y, price, tot_max, hmean, f, Q=None, Hres=None):
    rows, rule = [], []
    for par in GRID:
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot_max * (1 + 1e-5 * (23 - np.arange(T)))
        for L in LS:
            _, opt = lp(y, price, pen, cap, pmax, L)
            rc = [execute(None, None, y, price, pen, cap, pmax, L, th) for th in TH]
            rule.append((d, str(par), L, [c - opt for c, _ in rc], [u for _, u in rc]))
            if f is None:
                continue
            x0, f0 = lp(f, price, pen, cap, pmax, L)
            xl, _ = lp(f, price, pen, cap, pmax, L, 1.0, f0)
            xh, _ = lp(f, price, pen, cap, pmax, L, -1.0, f0)
            r = dict(day=d, cfg=str(par), L=L, opt=opt, dem=y.sum(), width=(xh[T:2 * T + 1].sum() - xl[T:2 * T + 1].sum()) / (25 * cap))
            for name, x in (("default", x0), ("min", xl), ("mid", (xl + xh) / 2), ("max", xh)):
                c, u = execute(unpack(x), f, y, price, pen, cap, pmax, L)
                r[f"reg_{name}"], r[f"short_{name}"] = c - opt, u
            xp, _ = lp(y, price, pen, cap, pmax, L)
            r["reg_pi"] = execute(unpack(xp), y, y, price, pen, cap, pmax, L)[0] - opt
            if Hres is not None and len(Hres) >= 7:          # 情景优化：候选 = 分位数 × 选解规则，按提前期执行规则在残差情景上模拟后选最优
                Ys = np.maximum(f[None, :] + Hres, 0)
                cands = {("mean", "default"): (x0, f), ("mean", "mid"): ((xl + xh) / 2, f), ("mean", "max"): (xh, f)}
                for q in (0.6, 0.7, 0.8, 0.9):
                    fq = np.maximum(np.array([np.interp(q, cfg.QS, row) for row in Q]), 0)
                    y0, g0 = lp(fq, price, pen, cap, pmax, L)
                    yl, _ = lp(fq, price, pen, cap, pmax, L, 1.0, g0)
                    yh, _ = lp(fq, price, pen, cap, pmax, L, -1.0, g0)
                    cands[(q, "default")], cands[(q, "mid")], cands[(q, "max")] = (y0, fq), ((yl + yh) / 2, fq), (yh, fq)
                sc = {k: np.mean([execute(unpack(x), fc, ys, price, pen, cap, pmax, L)[0] for ys in Ys]) for k, (x, fc) in cands.items()}
                kb = min(sc, key=sc.get)
                r["reg_saa"] = execute(unpack(cands[kb][0]), cands[kb][1], y, price, pen, cap, pmax, L)[0] - opt
                r["saa_q"], r["saa_tie"] = str(kb[0]), kb[1]
            rows.append(r)
    return rows, rule


if __name__ == "__main__":
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    Rr = FQ.y - FQ["mean"]
    t0 = Rr.index.min().normalize()
    Ar = Rr.reindex(pd.date_range(t0, Rr.index.max().normalize() + pd.Timedelta(hours=23), freq="h")).to_numpy().reshape(-1, 24)
    okr = np.isfinite(Ar).all(1)
    Tk = []
    for d in pd.date_range("2024-01-08", cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all():
            continue
        f = None
        if d >= cfg.MP_EVAL_START:
            f = np.maximum(G.reindex(idx).to_numpy(), 0)
            if not np.isfinite(f).all():
                continue
        Qd = Hd = None
        if f is not None:
            i = (d - t0).days
            hid = [j for j in range(max(0, i - 2 - cfg.SCEN_WINDOW + 1), i - 1) if okr[j]]
            Qd, Hd = FQ[cfg.QC].reindex(idx).to_numpy(), (Ar[hid] if hid else None)
        Tk.append((d, y, D.price(S, d.month), float(D.price("A7", d.month, "total").max()), hmean, f, Qd, Hd))
    if len(sys.argv) > 1:
        Tk = [t for t in Tk if t[5] is not None][40:40 + int(sys.argv[1])]
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=7, batch_size=2, verbose=2)(delayed(task)(*t) for t in Tk)
    R = pd.DataFrame([r for a, _ in res for r in a])
    RU = pd.DataFrame([(d, c, L, i, reg[i], sh[i]) for _, b in res for (d, c, L, reg, sh) in b for i in range(len(TH))], columns=["day", "cfg", "L", "th", "regret", "short"])
    dev = RU[RU.day < cfg.MP_EVAL_START]
    if len(dev):
        best = dev.groupby(["cfg", "L", "th"]).regret.sum().unstack("th").idxmin(axis=1).rename("best").reset_index()
        ev = RU[RU.day >= cfg.MP_EVAL_START].merge(best, on=["cfg", "L"])
        ev = ev[ev.th == ev.best][["day", "cfg", "L", "regret", "short"]].rename(columns={"regret": "rule", "short": "short_rule"})
        R = R.merge(ev, on=["day", "cfg", "L"])
    R["mult"] = R.cfg.map(lambda s: eval(s)[2])
    R.to_parquet(OUT / f"bss_leadtime{SUF}.parquet")
    lines = ["正对照（完全信息计划执行后的遗憾）最大值：" + str(R.reg_pi.abs().max())]
    R = R.dropna(subset=["reg_saa"])
    cols = [c for c in ("reg_default", "reg_min", "reg_mid", "reg_max", "reg_saa", "rule") if c in R]
    for (L, m), g in R.groupby(["L", "mult"]):
        tot = g[cols].sum()
        dd = g.groupby("day")[cols].sum()
        mm, ci, _ = boot_mean(1 - dd.reg_mid / dd.reg_default)
        s = f"L={L} 罚价×{m}: 总遗憾/默认 " + (tot / tot.reg_default).round(3).to_dict().__repr__() + f"；中点 vs 默认 {mm:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]；最优面宽度中位数 {g.width.median():.3f}"
        s += f"；未满足占比 默认 {g.short_default.sum() / g.dem.sum():.4f} 中点 {g.short_mid.sum() / g.dem.sum():.4f}"
        if "rule" in g:
            m2, c2, _ = boot_mean(1 - dd.reg_mid / dd.rule)
            m3, c3, _ = boot_mean(1 - dd.reg_saa / dd.rule)
            s += f"；情景优化 vs 规则 {m3:.3f} [{c3[0]:.3f}, {c3[1]:.3f}]"
            s += f" 规则 {g.short_rule.sum() / g.dem.sum():.4f}；中点 vs 规则 {m2:.3f} [{c2[0]:.3f}, {c2[1]:.3f}]；遗憾占成本 默认 {100 * g.reg_default.sum() / (g.opt.sum() + g.reg_default.sum()):.1f}% 中点 {100 * g.reg_mid.sum() / (g.opt.sum() + g.reg_mid.sum()):.1f}% 规则 {100 * g.rule.sum() / (g.opt.sum() + g.rule.sum()):.1f}%"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / f"exp_leadtime{SUF}.txt").write_text(txt, encoding="utf-8")
    print(txt)
