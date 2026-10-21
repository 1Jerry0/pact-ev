"""梯度式的决策导向训练（A7 小时模型，锚定 6 组 × 罚价 MULT，默认解 / 中点解各做一遍）：
在冻结的 LightGBM 均值预测之上继续做提升（boosting），损失就是下游遗憾本身：ŷ_t = f_t · exp(g(x_t))，g 为若干棵浅树之和。
每一轮：对训练窗内随机抽的若干天，逐小时用中心差分算 ∂遗憾/∂ŷ_t（穿过日前 LP 与跟踪执行规则，δ = 5% 平均需求），
链式法则换成对 g 的梯度，拟合一棵回归树（深度 3、叶子 ≥ 24 个样本），再在整个训练窗上对步长做线搜索；训练遗憾不降就停。
需求出现在约束里，SPO+ 之类针对目标系数的替代损失不适用；有限差分梯度对任何"预测 → 计划 → 执行"的管线都可用。
训练窗：每月初之前的 D−85…D−2（84 天，样本外的日前预测）；评估：当月各天。特征只用发布时刻可得的量。
对照（同一批天与组合）：均值预测、滚动选的统一分位、滚动选步长的价格分位（dfl_x*.parquet）、8 个分块偏差的直接搜索（dfl_learn_x*.parquet）。
用法：python exp_dfl_grad.py [pilot]；python exp_dfl_grad.py ana"""
import os
import pathlib
import sys
import time
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
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeRegressor
import cfg
import data as D
import sim
from exp_packs_value import pick, rel_ci
from exp_dfl import QU, LAM
from fastlp import FastBss

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
MULT = int(os.environ.get("MULT", 3))
GRID = [(c, p, MULT) for c in (4, 6, 8) for p in (2.5, 4.0)]
WIN, ROUNDS, NSUB, DEPTH, LEAF = 84, 20, 28, 3, 24
STEPS = (0.2, 0.1, 0.05, 0.025)
DREL = 0.05


def grad_day(P, f, tie, delta):
    g = np.zeros(24)
    for t in range(24):
        up, dn = f.copy(), f.copy()
        up[t] += delta
        dn[t] = max(dn[t] - delta, 0.0)
        g[t] = (P.regret(up, tie) - P.regret(dn, tie)) / (up[t] - dn[t])
    return g


def task(par, month, tie, train, test, hmean, verbose=False):
    """train / test: [(day, y, price, tot, f(24), X(24×k))]。返回 (cfg, month, tie, 轮数, [(day, 遗憾_DFL, 遗憾_均值)], 训练轨迹)。"""
    rng = np.random.default_rng(zlib.crc32(f"{par}|{month}|{tie}".encode()))
    mk = lambda r: sim.Problem("bss", r[1], r[2], par, dict(hmean=hmean, tot_max=r[3]))
    Ptr = [FastBss(mk(r)) for r in train]                       # 训练时用快速求值（与冻结流程逐例一致，见 fastlp.py）；评估仍用冻结的 sim.Problem
    Ftr = np.array([r[4] for r in train])
    Xtr = np.vstack([r[5] for r in train])
    n = len(train)
    delta = DREL * hmean
    g = np.zeros((n, 24))
    reg = lambda gg: np.array([Ptr[i].regret(Ftr[i] * np.exp(gg[i]), tie) for i in range(n)])
    cur = reg(g).sum()
    trace, trees, scale = [cur], [], None
    for r in range(ROUNDS):
        sub = rng.choice(n, size=min(NSUB, n), replace=False)
        G = np.zeros((len(sub), 24))
        for j, i in enumerate(sub):
            fc = Ftr[i] * np.exp(g[i])
            G[j] = grad_day(Ptr[i], fc, tie, delta) * fc
        if scale is None:
            scale = max(float(np.abs(G).mean()), 1e-9)
        z = -G / scale
        rows = (sub[:, None] * 24 + np.arange(24)[None, :]).ravel()
        tr = DecisionTreeRegressor(max_depth=DEPTH, min_samples_leaf=LEAF, random_state=int(rng.integers(1 << 30))).fit(Xtr[rows], z.ravel())
        step = np.clip(tr.predict(Xtr), -2.0, 2.0).reshape(n, 24)
        best, bnu = cur, None
        for nu in STEPS:
            v = reg(g + nu * step).sum()
            if v < best - 1e-9:
                best, bnu = v, nu
        if bnu is None:
            trace.append(cur)
            continue
        g += bnu * step
        cur = best
        trees.append((tr, bnu))
        trace.append(cur)
    out = []
    for rtest in test:
        P = mk(rtest)
        gt = np.zeros(24)
        for tr, nu in trees:
            gt += nu * np.clip(tr.predict(rtest[5]), -2.0, 2.0)
        fd = rtest[4] * np.exp(gt)
        out.append((rtest[0], P.regret(fd, tie), P.regret(rtest[4], tie), float(fd.sum() / max(rtest[4].sum(), 1e-9))))
    if verbose:
        print(par, month, tie, "训练遗憾轨迹 / 初值:", np.round(np.array(trace) / max(trace[0], 1e-9), 3).tolist(), "树数", len(trees), flush=True)
    return str(par), month, tie, len(trees), out, [float(v) for v in trace]


def build():
    H = D.mp_hourly()
    y = H[S]
    hmean = float(y.loc[:cfg.MP_DEV_END].mean())
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    FE = D.features(y)
    rec = {}
    for d in pd.date_range(cfg.MP_FC_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        yy, f = y.reindex(idx).to_numpy(), FQ["mean"].reindex(idx).to_numpy()
        q10, q50, q90 = (FQ[c].reindex(idx).to_numpy() for c in ("q10", "q50", "q90"))
        if not (np.isfinite(yy).all() and np.isfinite(f).all() and np.isfinite(q90).all()):
            continue
        f = np.maximum(f, 0)
        den = np.maximum(f, 0.05 * hmean)
        fe = FE.reindex(idx)
        X = np.column_stack([np.arange(24), fe.dow.to_numpy(), fe.workday.to_numpy(), fe.holiday.to_numpy(), f / hmean, (q90 - q10) / den, (q50 - f) / den,
                             np.full(24, f.sum() / (24 * hmean)), np.nan_to_num(fe.lag_asof.to_numpy() / hmean), np.nan_to_num(fe.mean_d2_d8.to_numpy() / hmean)])
        rec[d] = (d, yy, D.price(S, d.month), HK(D.price("A7", d.month, "total")), f, X)
    return rec, hmean


def analyse():
    Gd = pd.read_parquet(OUT / f"dfl_grad_x{MULT}.parquet")
    L = pd.read_parquet(OUT / f"dfl_learn_x{MULT}.parquet")
    R = pd.read_parquet(OUT / f"dfl_x{MULT}.parquet")
    days = sorted(set(Gd.day.unique()) & set(L.day.unique()))
    base = L[L.tie == "default"].groupby("day").reg_mean.sum().loc[days]
    b = base.sum()
    lines = [f"罚价 ×{MULT}；{len(days)} 天 × {Gd.cfg.nunique()} 组；总遗憾 /（均值预测 + 默认解）"]
    for tie in ("default", "mid"):
        gg = Gd[Gd.tie == tie].groupby("day")[["reg_dfl", "reg_mean"]].sum().loc[days]
        ll = L[L.tie == tie].groupby("day")[["reg_learn", "reg_mean"]].sum().loc[days]
        M = R[R.tie == tie].groupby(["day", "cand"]).regret.sum().unstack("cand")
        uni, pri = ["mean"] + [f"q{q}" for q in QU], ["mean"] + [f"p{l}" for l in LAM]
        tu = np.array([M.loc[d, pick(M[uni], d, "mean")] for d in days])
        tp = np.array([M.loc[d, pick(M[pri], d, "mean")] for d in days])
        s = (f"[{tie}] 均值 {gg.reg_mean.sum() / b:.3f}（核对 {ll.reg_mean.sum() / b:.3f}）；梯度式 DFL {gg.reg_dfl.sum() / b:.3f}；分块偏差直接搜索 {ll.reg_learn.sum() / b:.3f}；"
             f"滚动统一分位 {tu.sum() / b:.3f}；滚动价格分位 {tp.sum() / b:.3f}")
        for nm, x in (("DFL vs 均值", gg.reg_mean.to_numpy()), ("DFL vs 分块偏差", ll.reg_learn.to_numpy()), ("DFL vs 价格分位", tp), ("DFL vs 统一分位", tu)):
            m, lo, hi = rel_ci(gg.reg_dfl.to_numpy(), x)
            s += f"；{nm} {100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        k = Gd[Gd.tie == tie].drop_duplicates(["cfg", "month"])
        s += f"；平均树数 {k.ntree.mean():.1f}；预测总量 / 均值预测总量 {Gd[Gd.tie == tie].ratio.mean():.3f}；训练遗憾末值 / 初值（中位）{k.tr_ratio.median():.3f}"
        lines.append(s)
    # 默认解 + DFL 与 中点解 + 均值 的对比：替代关系
    gd = Gd[Gd.tie == "default"].groupby("day").reg_dfl.sum().loc[days].to_numpy()
    mm = Gd[Gd.tie == "mid"].groupby("day").reg_mean.sum().loc[days].to_numpy()
    gm = Gd[Gd.tie == "mid"].groupby("day").reg_dfl.sum().loc[days].to_numpy()
    m, lo, hi = rel_ci(gd, mm)
    lines.append(f"默认解 + 梯度式 DFL vs 中点解 + 均值预测：{100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
                 f"DFL 去掉的遗憾：对默认解 {100 * (1 - gd.sum() / b):.0f}%，对中点解 {100 * (1 - gm.sum() / mm.sum()):.0f}%；中点解（均值预测）去掉 {100 * (1 - mm.sum() / b):.0f}%")
    txt = "\n".join(lines)
    (OUT / f"exp_dfl_grad_x{MULT}.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    rec, hmean = build()
    Tk = []
    for mo in range(5, 13):
        m0 = pd.Timestamp(2024, mo, 1)
        train = [rec[d] for d in pd.date_range(m0 - pd.Timedelta(days=WIN + 1), m0 - pd.Timedelta(days=2), freq="D") if d in rec]
        test = [rec[d] for d in pd.date_range(m0, m0 + pd.offsets.MonthEnd(0), freq="D") if d in rec]
        for par in GRID:
            for tie in ("default", "mid"):
                Tk.append((par, mo, tie, train, test, hmean))
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        for t in [t for t in Tk if t[0] == (6, 2.5, MULT) and t[1] == 6]:
            t0 = time.perf_counter()
            c, mo, tie, nt, out, trace = task(*t, verbose=True)
            a = np.array([(o[1], o[2]) for o in out])
            print(tie, "用时 s", round(time.perf_counter() - t0, 1), "训练天数", len(t[3]), "测试遗憾 DFL / 均值", round(a[:, 0].sum() / a[:, 1].sum(), 3), flush=True)
        sys.exit()
    Tk.sort(key=lambda t: t[2] != "mid")                       # 中点解的任务更慢，先排
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 7)), batch_size=1, verbose=5)(delayed(task)(*t) for t in Tk)
    rows = [(c, mo, tie, nt, tr[-1] / max(tr[0], 1e-9), d, rd, rm, ratio) for c, mo, tie, nt, out, tr in res for d, rd, rm, ratio in out]
    pd.DataFrame(rows, columns=["cfg", "month", "tie", "ntree", "tr_ratio", "day", "reg_dfl", "reg_mean", "ratio"]).to_parquet(OUT / f"dfl_grad_x{MULT}.parquet")
    analyse()
