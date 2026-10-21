"""研究假设 H2（误差顺序的作用）在利润口径下重算：与正式机制实验相同的做法（5 成员组合的样本外残差，只在同一电价时段内交换，
低 / 中 / 高三档持续性各 30 个排列，误差指标严格不变），只是缺供与超限按利润口径计价：
换电站 A7 缺供按逐小时零售总价；储能站 A1、A3 超限按服务费（零售总价 − 电度电价）。
统计量同原分析：同一天、同一配置、同一持续性档内遗憾的第 10—90 百分位极差 / 平均遗憾，先在档与配置上平均，再对日自助。
配置：A7 紧充电网格（库存 2/4/8 小时 × 功率 1.1/1.3/1.6 倍，预先设定的 H2 检验网格）与数据锚定网格；A1、A3 为 9 种储能情景。
输出：结果/h2_profit.parquet；结果/h2_profit.txt。用法：python exp_h2_profit.py [run|ana]"""
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
from joblib import Parallel, delayed
import cfg
import data as D
import sim
from mech import chain, tou_blocks
from exp_cost_basis import TIEBREAK, exec_split as bss_exec_split, mix
from exp_cost_basis_st import exec_split as st_exec_split, fee, CFGS as ST_CFGS

OUT = ROOT / "results"
TIGHT = [(c, p) for c in (2, 4, 8) for p in (1.1, 1.3, 1.6)]
ANCHOR = [(c, p) for c in (4, 6, 8) for p in (2.5, 4.0)]


def bss_regret(f, y, price, pen, cap, pmax, opt):
    lo = sim.bss_plan(f, price, pen, cap, pmax, "min")
    hi = sim.bss_plan(f, price, pen, cap, pmax, "max")
    pl = mix(("bss", lo[0], lo[1], f, lo[3]), ("bss", hi[0], hi[1], f, hi[3]), 0.5)
    e, _, pn = bss_exec_split(pl, y, price, pen, cap, pmax)
    return e + pn - opt


def st_regret(f, y, price, phi, E, L, opt):
    c, d, _ = sim.st_plan(f, price, E, L, phi, "mid")
    el, _, pn = st_exec_split(c, d, y, price, E, L, phi)
    return el + pn - opt


def task(station, day, y, mean, scale, seed):
    price = D.price(station, day.month)
    rng = np.random.default_rng(seed)
    e = y - mean
    samples = {lev: chain(e, y, tou_blocks(price), kap, cfg.MECH_N_PER, rng) for lev, kap in cfg.MECH_KAPPA.items()}
    rows = []
    if station == cfg.BSS:
        pen = D.price("A7", day.month, "total") * TIEBREAK
        for grid_name, grid in (("tight", TIGHT), ("anchor", ANCHOR)):
            for c, p in grid:
                cap, pmax = c * scale["hmean"], p * scale["hmean"]
                opt = sim.bss_plan(y, price, pen, cap, pmax, None)[2]
                for lev, smp in samples.items():
                    for k in smp:
                        rows.append((station, day, grid_name, f"({c}, {p})", lev, bss_regret(np.maximum(y - k, 0), y, price, pen, cap, pmax, opt)))
    else:
        phi = fee(station, day.month)
        for fe, fq in ST_CFGS:
            E, L = fe * scale["dmean"], scale["peaks"][fq]
            c0, d0, obj = sim.st_plan(y, price, E, L, phi, None)
            opt = obj + float(price @ y)
            for lev, smp in samples.items():
                for k in smp:
                    rows.append((station, day, "storage", f"({fe}, {fq})", lev, st_regret(np.maximum(y - k, 0), y, price, phi, E, L, opt)))
    return rows


def run():
    from dec_mp import scales
    H = D.mp_hourly()
    sc = scales(H)
    F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{cfg.MECH_SOURCE}.parquet")
    T = []
    for st in cfg.MECH_STATIONS:
        g = F[F.series == st].set_index("t")
        for i, d in enumerate(pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D")):
            idx = pd.date_range(d, periods=24, freq="h")
            y = H[st].reindex(idx).to_numpy()
            mean = np.maximum(g.reindex(idx)["mean"].to_numpy(), 0)
            if np.isfinite(y).all() and np.isfinite(mean).all():
                T.append((st, d, y, mean, sc[st], cfg.SEED + 1000 * i + sum(map(ord, st + cfg.MECH_SOURCE))))
    print(len(T), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 6)), batch_size=2, verbose=2)(delayed(task)(*t) for t in T)
    pd.DataFrame([r for rr in res for r in rr], columns=["station", "day", "grid", "cfg", "level", "regret"]).to_parquet(OUT / "h2_profit.parquet")


def ana():
    from analysis import boot_mean
    M = pd.read_parquet(OUT / "h2_profit.parquet")
    g = M.groupby(["station", "grid", "cfg", "level", "day"]).regret
    disp = ((g.quantile(0.9) - g.quantile(0.1)) / g.mean().clip(lower=1e-9)).reset_index(name="disp")
    lines = []
    for (st, gr), h in disp.groupby(["station", "grid"]):
        dd = h.groupby("day").disp.mean()
        m, ci, means = boot_mean(dd)
        p = float((means <= cfg.SESOI_REGRET).mean())
        lines.append(f"{st} {gr}：离散度均值 {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]，中位数 {h.groupby('day').disp.mean().median():.3f}；相对 5% 的单侧 p = {p:.4f}")
    lev = M.groupby(["station", "grid", "level", "day"]).regret.mean().unstack("level")
    for (st, gr), h in lev.groupby(level=[0, 1]):
        lines.append(f"{st} {gr}：高 / 低持续性的平均遗憾之比（日中位数）{(h['high'] / h['low']).median():.2f}；均值 {h['high'].mean():.0f} 对 {h['low'].mean():.0f}")
    txt = "\n".join(lines)
    (OUT / "h2_profit.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    (ana if (sys.argv[1:] or ["run"])[0] == "ana" else run)()
