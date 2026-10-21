"""命题 3（计划库存是每小时的"免赔额"）的数值核对：小时换电模型、跟踪规则、锚定 18 组、LightGBM 均值预测、评估期全部天。
计划给出每小时充电后的目标可用量 A_t = s^p_{t+1} + D̂_t − u^p_t，免赔额 b_t = A_t − D̂_t = s^p_{t+1} − u^p_t。
命题：若第 t 小时充电功率与库容都不约束（A_t ≤ s_t + ηP 且 A_t ≤ N + D̂_t），则执行缺口 v_t ≤ (e_t − b_t)^+，e_t = y_t − D̂_t；当 s_t ≤ A_t 时取等号。
报告：约束不绑定的小时占比、不等式成立占比、等号成立占比；默认 / 中点 / 最大库存解下免赔额为零的小时占比（全天、晚间 17–23 时）；
缺口发生的小时里免赔额为零的占比。用法：python exp_prop3.py"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
import sim

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
ETA = cfg.ETA
GRID = [(c, p, m) for c in (4, 6, 8) for p in (2.5, 4.0) for m in (3,)]   # 利润口径下罚价档被忽略，只算一档


def task(d, y, f, price, tot, hmean):
    rows = []
    for par in GRID:
        P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
        for tie in ("default", "mid", "max"):
            g, s, _, u = sim.bss_plan(f, price, P.pen, P.cap, P.pmax, tie)
            A = s[1:] + f - u
            b = A - f
            st = P.cap / 2
            for t in range(24):
                dh = f[t]
                gg = float(np.clip((s[t + 1] - st + dh - u[t]) / ETA, 0, min(P.pmax, (P.cap - st + dh) / ETA)))
                avail = st + ETA * gg
                served = min(y[t], avail)
                v = y[t] - served
                slack = (A[t] <= st + ETA * P.pmax + 1e-9) and (A[t] <= P.cap + dh + 1e-9)
                bound = max(y[t] - dh - b[t], 0.0)
                rows.append((d, str(par), tie, t, slack, st <= A[t] + 1e-9, v, bound, b[t], b[t] / P.cap))
                st = min(avail - served, P.cap)
    return rows


if __name__ == "__main__":
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y, f = H[S].reindex(idx).to_numpy(), G.reindex(idx).to_numpy()
        if np.isfinite(y).all() and np.isfinite(f).all():
            Tk.append((d, y, np.maximum(f, 0), D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean))
    res = Parallel(n_jobs=int(os.environ.get("NJ", 4)), batch_size=4)(delayed(task)(*t) for t in Tk)
    R = pd.DataFrame([r for rr in res for r in rr], columns=["day", "cfg", "tie", "t", "slack", "below", "v", "bound", "b", "b_rel"])
    R.to_parquet(OUT / "prop3.parquet")
    tol = 1e-6 * hmean
    sl = R[R.slack]
    lines = [f"{R.day.nunique()} 天 × {R.cfg.nunique()} 组 × 3 种选解 × 24 小时 = {len(R)} 个小时；充电功率与库容不约束的小时占 {100 * R.slack.mean():.1f}%",
             f"其中 v ≤ (e − b)^+ 成立 {100 * (sl.v <= sl.bound + tol).mean():.3f}%；s_t ≤ A_t 的小时里等号成立 {100 * (np.abs(sl[sl.below].v - sl[sl.below].bound) <= tol).mean():.3f}%"]
    for tie, g in R.groupby("tie"):
        ev = g[g.t >= 17]
        sh = g[g.v > tol]
        lines.append(f"[{tie}] 免赔额为零（< 0.1% 库容）的小时：全天 {100 * (g.b_rel < 1e-3).mean():.0f}%，17–23 时 {100 * (ev.b_rel < 1e-3).mean():.0f}%；平均免赔额 {100 * g.b_rel.mean():.1f}% 库容；"
                     f"发生缺口的小时占 {100 * (g.v > tol).mean():.2f}%，其中免赔额为零的占 {100 * (sh.b_rel < 1e-3).mean():.0f}%；缺口总量 {g.v.sum() / 1e3:.0f} MWh")
    txt = "\n".join(lines)
    (OUT / "exp_prop3.txt").write_text(txt, encoding="utf-8")
    print(txt)
