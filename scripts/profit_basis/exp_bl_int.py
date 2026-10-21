"""电池级计划的"整块电池"检验。exp_bl.py 的 LP 是线性松弛：预测到达量是小数（逐小时预测 / 4 / 平均电量），计划里会出现半块电池。
试过直接加整数约束（exp_bl_milp.py）：到达量为小数时 MILP 30 秒内常解不完；而到达量为整数时（完全信息）LP 松弛本身就是整数解。
所以这里把预测到达量按累计取整变成整数（总量不变、逐步到达数为整数），再解同一个 LP——得到的就是整块电池的计划，
与小数计划比较执行成本，并重算"滚动调参管线 vs 滚动规则"（候选只含均值预测 × 裕量 {0,1,2} × 4 种选解，两种计划各自滚动选）。
配置：6 h × {12, 18, 25} kW 与 4 h × 12 kW，罚价 ×3。用法：python exp_bl_int.py"""
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
from exp_bl import lp_bl, position, run_day_bl, nsteps, load_events, NS
from exp_packs_value import rel_ci, pick, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((6, 2.5, 3), 12.0), ((6, 2.5, 3), 18.0), ((6, 2.5, 3), 25.0), ((4, 2.5, 3), 12.0)]


def task(d, ev_step, ev_e, price, tot, hmean, f, ebar):
    rows = []
    ee = np.minimum(ev_e, 1.6 * ebar)
    dd = np.repeat(np.maximum(f, 0) / ebar / 4, 4)
    di = np.diff(np.r_[0.0, np.round(np.cumsum(dd))])
    for par, pc in GRID:
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)
        for mg in (0, 1, 2):
            x0, f0 = lp_bl(di, price, pen, K, C, n + mg, ebar)
            xl, _ = lp_bl(di, price, pen, K, C, n + mg, ebar, 1.0, f0)
            xh, _ = lp_bl(di, price, pen, K, C, n + mg, ebar, -1.0, f0)
            frac = float(max(np.abs(x[:NS] - np.round(x[:NS])).max() for x in (x0, xl, xh)))
            for tie, x in (("min", xl), ("mid", (xl + xh) / 2), ("max", xh), ("default", x0)):
                c, u = run_day_bl(ev_step, ee, price, pen, K, C, pc, ebar, position(x, n + mg))
                rows.append((d, str(par), pc, f"mean@{mg}|{tie}", c, u, frac))
    return rows


if __name__ == "__main__":
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        f = G.reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
        if np.isfinite(f).all() and d in EV:
            Tk.append((d, *EV[d], D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, f, ebar))
    print(len(Tk), "tasks", flush=True)
    I = pd.DataFrame([r for rr in Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk) for r in rr], columns=["day", "cfg", "pc", "cand", "cost", "short", "frac"])
    I.to_parquet(OUT / "bl_int.parquet")
    FR = pd.read_parquet(OUT / "bl_fc.parquet")
    RU = pd.read_parquet(OUT / "bl_rule.parquet")
    lines = [f"整数到达量下 LP 解的最大小数部分（x 与最近整数之差）：中位 {I.frac.median():.2e}，最大 {I.frac.max():.2e}"]
    for (c, pc), g in I.groupby(["cfg", "pc"], sort=False):
        A = g.pivot(index="day", columns="cand", values="cost").sort_index()
        B = FR[(FR.cfg == c) & (FR.pc == pc) & FR.cand.str.startswith("mean@")].pivot(index="day", columns="cand", values="cost").sort_index()
        Mc = RU[(RU.cfg == c) & (RU.pc == pc)].pivot(index="day", columns="th", values="cost").sort_index()
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        days = [d for d in A.index if d >= START and d in B.index and d in Mc.index]
        ref = np.array([Mc.loc[d, pick(Mc, d, k0)] for d in days])
        s = f"{c} {pc:.0f} kW（{len(days)} 天）：整块计划 vs 小数计划的执行成本 %："
        for k in ("mean@0|default", "mean@0|max", "mean@1|max", "mean@0|min"):
            m, lo, hi = rel_ci(A.loc[days, k].to_numpy(), B.loc[days, k].to_numpy())
            s += f" {k} {100 * m:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
        ta = np.array([A.loc[d, pick(A, d, "mean@0|mid")] for d in days])
        tb = np.array([B.loc[d, pick(B, d, "mean@0|mid")] for d in days])
        for nm, x in (("整块·滚动调参 vs 规则", ta), ("小数·滚动调参 vs 规则", tb)):
            m, lo, hi = rel_ci(x, ref)
            s += f"；{nm} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        m, lo, hi = rel_ci(A.loc[days, "mean@0|max"].to_numpy(), A.loc[days, "mean@0|min"].to_numpy())
        s += f"；整块计划 最大 vs 最小 {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_int.txt").write_text(txt, encoding="utf-8")
    print(txt)
