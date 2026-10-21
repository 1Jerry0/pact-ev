"""端到端的决策导向对照：直接以下游遗憾为损失"学"一个预测修正（A7 小时模型，LightGBM 均值预测，锚定 6 组，罚价由 MULT 给定）。
修正形式：ŷ_t = f_t · (1 + b_{块(t)})，一天分 8 个 3 小时块，b ∈ {−0.1, 0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}；
训练：每月初用此前 D−29…D−2 的 28 天，以这些天的总遗憾为目标做坐标下降（两轮，从晚间块开始），逐组合、逐选解规则（默认 / 中点）分别学；
评估：当月各天用学到的 b。这是黑箱式的"直接损失最小化"，不需要对 LP 求导。
对照（同一批天、同一批组合，取自 dfl_x{MULT}.parquet）：均值预测；滚动选的统一分位；滚动选步长的价格分位。用法：python exp_dfl_learn.py"""
import os
import pathlib
import sys
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
import cfg
import data as D
import sim
from exp_packs_value import pick, rel_ci
from exp_dfl import QU, LAM

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
MULT = int(os.environ.get("MULT", 3))
GRID = [(c, p, MULT) for c in (4, 6, 8) for p in (2.5, 4.0)]
BG = (-0.1, 0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
BLK = np.arange(24) // 3


def task(par, month, tie, train, test, hmean):
    """train / test: [(day, y, price, tot, f)]。返回 (学到的 b, [(day, 遗憾)])。"""
    mk = lambda r: sim.Problem("bss", r[1], r[2], par, dict(hmean=hmean, tot_max=r[3]))
    Ptr = [(mk(r), r[4]) for r in train]
    obj = lambda b: sum(P.regret(np.maximum(f * (1 + b[BLK]), 0), tie) for P, f in Ptr)
    b = np.zeros(8)
    best = obj(b)
    for _ in range(2):
        for k in range(7, -1, -1):
            for v in BG:
                if v == b[k]:
                    continue
                c = b.copy()
                c[k] = v
                o = obj(c)
                if o < best - 1e-9:
                    best, b = o, c
    out = []
    for r in test:
        P = mk(r)
        out.append((r[0], P.regret(np.maximum(r[4] * (1 + b[BLK]), 0), tie), P.regret(r[4], tie), P.regret(r[4], "default")))
    return str(par), month, tie, b.tolist(), out


def analyse():
    L = pd.read_parquet(OUT / f"dfl_learn_x{MULT}.parquet")
    R = pd.read_parquet(OUT / f"dfl_x{MULT}.parquet")
    days = sorted(L.day.unique())
    base = L[L.tie == "default"].groupby("day").reg_mean.sum()
    lines = [f"罚价 ×{MULT}；{len(days)} 天 × {L.cfg.nunique()} 组；总遗憾 /（均值预测 + 默认解）"]
    for tie in ("default", "mid"):
        g = L[L.tie == tie].groupby("day")[["reg_learn", "reg_mean"]].sum()
        M = R[R.tie == tie].groupby(["day", "cand"]).regret.sum().unstack("cand")
        uni, pri = ["mean"] + [f"q{q}" for q in QU], ["mean"] + [f"p{l}" for l in LAM]
        tu = np.array([M.loc[d, pick(M[uni], d, "mean")] for d in days])
        tp = np.array([M.loc[d, pick(M[pri], d, "mean")] for d in days])
        b = base.loc[days].sum()
        s = f"[{tie}] 均值 {g.reg_mean.sum() / b:.3f}；学到的分块偏差 {g.reg_learn.sum() / b:.3f}；滚动选统一分位 {tu.sum() / b:.3f}；滚动选价格分位 {tp.sum() / b:.3f}"
        for nm, x in (("学到的偏差 vs 价格分位", tp), ("学到的偏差 vs 统一分位", tu), ("学到的偏差 vs 均值", g.reg_mean.loc[days].to_numpy())):
            m, lo, hi = rel_ci(g.reg_learn.loc[days].to_numpy(), x)
            s += f"；{nm} {100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        bb = np.array(L[L.tie == tie].drop_duplicates(["cfg", "month"]).b.to_list())
        s += "；学到的各块偏差均值（0–2 时 … 21–23 时）" + np.round(bb.mean(0), 2).tolist().__repr__()
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / f"exp_dfl_learn_x{MULT}.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    rec = {}
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y, f = H[S].reindex(idx).to_numpy(), G.reindex(idx).to_numpy()
        if np.isfinite(y).all() and np.isfinite(f).all():
            rec[d] = (d, y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), np.maximum(f, 0))
    Tk = []
    for mo in range(5, 13):
        m0 = pd.Timestamp(2024, mo, 1)
        train = [rec[d] for d in pd.date_range(m0 - pd.Timedelta(days=29), m0 - pd.Timedelta(days=2), freq="D") if d in rec]
        test = [rec[d] for d in pd.date_range(m0, m0 + pd.offsets.MonthEnd(0), freq="D") if d in rec]
        for par in GRID:
            for tie in ("default", "mid"):
                Tk.append((par, mo, tie, train, test, hmean))
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 7)), batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk)
    rows = [(c, mo, tie, b, d, rl, rm, rd) for c, mo, tie, b, out in res for d, rl, rm, rd in out]
    pd.DataFrame(rows, columns=["cfg", "month", "tie", "b", "day", "reg_learn", "reg_mean", "reg_mean_default"]).to_parquet(OUT / f"dfl_learn_x{MULT}.parquet")
    analyse()
