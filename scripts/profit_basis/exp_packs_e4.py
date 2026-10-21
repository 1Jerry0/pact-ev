"""电池级的 E4：同一组误差只在同一电价时段内交换（MAE、RMSE、日偏差严格不变），看电池级仿真里当天成本还变不变。
抽样方式沿用冻结的 mech.chain（低 / 中 / 高持续性三档，每档 30 个排列，来源 = 5 成员组合预测）；
每个排列分别用默认解与中点解做计划、在电池级仿真里执行。配置：6 h、2.5×、×3（12 kW / L=2 与 25 kW / L=1），4 h、2.5×、×3（12 kW）。
统计：同一天同一档内成本的 p10–p90 极差，相对"误差的成本"（计划用预测 − 计划用真实需求，同为默认解）与相对总成本；高/低持续性；真实排列 vs 随机排列。
用法：python exp_packs_e4.py"""
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
import sim as S0
from mech import chain, tou_blocks
from analysis import boot_mean
from exp_leadtime import lp, unpack
from exp_packs import run_day, positions, RAW, T

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((6, 2.5, 3), 12.0, 2), ((6, 2.5, 3), 25.0, 1), ((4, 2.5, 3), 12.0, 2)]


def task(d, ev_step, ev_e, y, f, price, tot, hmean, ebar, seed):
    rng = np.random.default_rng(seed)
    e = y - f
    samples = {"orig": [e]}
    for lev, kap in cfg.MECH_KAPPA.items():
        samples[lev] = chain(e, y, tou_blocks(price), kap, cfg.MECH_N_PER, rng)
    ee = np.minimum(ev_e, 1.6 * ebar)
    rows = []
    for par, pc, L in GRID:
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(T)))
        sim = lambda x, fc: run_day(ev_step, ee, price, pen, cap, pmax, pc, ebar, positions(unpack(x), fc, L) / ebar)[0]
        cpi = sim(lp(y, price, pen, cap, pmax, L)[0], y)
        for lev, smp in samples.items():
            for i, k in enumerate(smp):
                fk = np.maximum(y - k, 0)
                x0, f0 = lp(fk, price, pen, cap, pmax, L)
                xl, _ = lp(fk, price, pen, cap, pmax, L, 1.0, f0)
                xh, _ = lp(fk, price, pen, cap, pmax, L, -1.0, f0)
                rows.append((d, str(par), pc, lev, i, float(S0.acf1(k)), cpi, sim(x0, fk), sim((xl + xh) / 2, fk)))
    return rows


def ana():
    R = pd.read_parquet(OUT / "bss_packs_e4.parquet")
    lines = []
    for (c, pc), g in R.groupby(["cfg", "pc"], sort=False):
        P = g[g.level != "orig"]
        s = f"{c} {pc:.0f} kW（{g.day.nunique()} 天）："
        for col, nm in (("c_def", "默认解"), ("c_mid", "中点解")):
            q = P.groupby(["day", "level"])[col]
            rng_ = (q.quantile(0.9) - q.quantile(0.1)).groupby("day").mean()
            ex = (P[col] - P.cpi).groupby(P.day).mean()
            tot = P.groupby("day")[col].mean()
            m, ci, _ = boot_mean(rng_)
            ratio = rng_.sum() / ex.sum()
            bs = [rng_.sample(frac=1, replace=True, random_state=i) for i in range(500)]
            rr = np.array([b.sum() / ex.loc[b.index].sum() for b in bs])
            s += (f" [{nm}] 同日同档成本 p10–p90 极差 日均 {m:.0f} 元 [{ci[0]:.0f}, {ci[1]:.0f}]，= 误差成本（日均 {ex.mean():.0f} 元）的 {ratio:.2f} 倍 [{np.quantile(rr, .025):.2f}, {np.quantile(rr, .975):.2f}]，"
                  f"= 总成本的 {100 * rng_.sum() / tot.sum():.1f}%；")
            lv = (P[col] - P.cpi).groupby([P.day, P.level]).mean().unstack()
            o = (g[g.level == "orig"].set_index("day")[col] - g[g.level == "orig"].set_index("day").cpi)
            dlt = (o - lv["mid"]).dropna()
            m2, c2, _ = boot_mean(dlt)
            s += f"高/低持续性的误差成本 {lv['high'].mean():.0f} / {lv['low'].mean():.0f} 元（比 {lv['high'].mean() / lv['low'].mean():.2f}）；真实排列 {o.mean():.0f} vs 随机 {lv['mid'].mean():.0f}（差 {m2:.0f} [{c2[0]:.0f}, {c2[1]:.0f}]）；"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_packs_e4.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        ana()
        sys.exit()
    d0 = pd.read_excel(RAW / "charging_session.xlsx", sheet_name="A7")
    d0.columns = ["conn", "disc", "energy"]
    d0["energy"] = pd.to_numeric(d0.energy, errors="coerce")
    d0["disc"] = pd.to_datetime(d0.disc, errors="coerce")
    a = d0[d0.energy.gt(0) & d0.disc.notna()].copy()
    onhour = (a.disc.dt.minute == 0) & (a.disc.dt.second == 0)
    mass = (a.disc >= "2024-07-26 10:00") & (a.disc < "2024-07-26 11:00")
    a = a[~onhour & ~mass].sort_values("disc")
    ebar = float(a.energy.mean())
    a["day"], a["step"] = a.disc.dt.normalize(), a.disc.dt.hour * 4 + a.disc.dt.minute // 15
    EV = {d: (g.step.to_numpy(), g.energy.to_numpy()) for d, g in a.groupby("day")}
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{cfg.MECH_SOURCE}.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    Tk = []
    for i, d in enumerate(pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D")):
        idx = pd.date_range(d, periods=24, freq="h")
        y, f = H[S].reindex(idx).to_numpy(), np.maximum(G.reindex(idx).to_numpy(), 0)
        if np.isfinite(y).all() and np.isfinite(f).all() and d in EV:
            Tk.append((d, *EV[d], y, f, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, ebar, cfg.SEED + 1000 * i))
    if len(sys.argv) > 1:
        Tk = Tk[:int(sys.argv[1])]
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk)
    pd.DataFrame([r for rr in res for r in rr], columns=["day", "cfg", "pc", "level", "i", "acf", "cpi", "c_def", "c_mid"]).to_parquet(OUT / "bss_packs_e4.parquet")
    ana()
