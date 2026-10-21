"""成本口径重算（储能辅助充电站，7 站）：超过并网限值的负荷按削减充电、少售电量计价。
依据：国办发〔2023〕19 号规定 2030 年前对实行两部制电价的集中式充换电设施免收需量（容量）电费，
国内充电站超限不产生需量电费，只能削减充电负荷；按利润口径，每削减 1 kWh 少收零售总价 r_t、少买电 p_t，
净损失为服务费 r_t − p_t（数据集按站、按月给出；A4、A9 无价格，沿用 A3、A1 的价格，与主分析一致）。
在原模型（超限电量照买并另付 φ）中，这等价于 φ_t = r_t − p_t。
资源：E = 日均电量 10%/25%/50% × L = 开发期日峰值第 30/50/70 百分位（9 种情景）。9 个模型的均值预测，默认 / 中点计划，逐时纠偏执行。
输出：结果/cb_storage_sched.parquet。用法：python exp_cost_basis_st.py [check]"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
import sim

OUT = ROOT / "results"
ETA = cfg.ETA
POOL = ["naive", "naive7", "seasonal_q", "lqr", "lgb_local", "lgb_global", "chronos2_cov", "timesfm3_cov", "combo5"]
CFGS = [(e, q) for e in (0.1, 0.25, 0.5) for q in (0.3, 0.5, 0.7)]


def fee(station, month):
    return D.price(station, month, "total") - D.price(station, month)


def exec_split(c_plan, d_plan, y, price, E, L, lam):
    """与 sim.st_exec（current 模式）逐行相同；lam 可为逐小时向量；分开返回电费（含 Σp·y 与日末估值）、超限电量、超限损失。"""
    lam = np.broadcast_to(np.asarray(lam, float), (24,))
    soc, El, X, Pn = E / 2, 0.0, 0.0, 0.0
    for t in range(24):
        net, soc = sim._st_step(c_plan[t], d_plan[t], y[t], soc, E, L, True)
        x = max(net - L, 0)
        El += price[t] * net
        X += x
        Pn += lam[t] * x
    return El + sim.terminal_value(soc, E / 2, price.min()), X, Pn


def task(station, d, y, price, phi, scale, fcs, check):
    rows = []
    for fe, fq in CFGS:
        E, L = fe * scale["dmean"], scale["peaks"][fq]
        c0, d0, obj = sim.st_plan(y, price, E, L, phi, None)
        e0, x0, p0 = exec_split(c0, d0, y, price, E, L, phi)
        base = dict(station=station, day=d, cfg=f"({fe}, {fq})", dem=float(y.sum()), cost_opt=obj + float(price @ y), E_opt=e0, X_opt=x0)
        for m, f in fcs.items():
            mae = float(np.abs(y - f).mean())
            for tie in ("default", "mid"):
                c, dd, _ = sim.st_plan(f, price, E, L, phi, tie)
                el, x, pn = exec_split(c, dd, y, price, E, L, phi)
                rows.append(dict(base, model=m, tie=tie, E=el, X=x, Pen=pn, mae=mae))
            if check and m == "lgb_local":
                P = sim.Problem("storage", y, price, (fe, fq, 3.0), scale)
                rows.append(dict(base, model="check_phi3", tie="mid", E=P.cost(P.plan(f, "mid")) - P.opt, X=np.nan, Pen=np.nan, mae=np.nan))
    return rows


def build(check=False):
    H = D.mp_hourly()
    dev = H.loc[:cfg.MP_DEV_END]
    tasks = []
    for s in cfg.POWER:
        daily, peaks = dev[s].resample("D").sum(), dev[s].resample("D").max()
        scale = dict(dmean=float(daily.mean()), hmean=float(dev[s].mean()), peaks={q: float(peaks.quantile(q)) for q in (0.3, 0.5, 0.7)})
        G = {}
        for m in POOL:
            F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "mean"])
            G[m] = F[F.series == s].set_index("t")["mean"]
        days = pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D")
        if check:
            days = days[:5]
        for d in days:
            idx = pd.date_range(d, periods=24, freq="h")
            y = H[s].reindex(idx).to_numpy()
            fcs = {m: np.maximum(G[m].reindex(idx).to_numpy(), 0) for m in POOL}
            if not np.isfinite(y).all() or not all(np.isfinite(v).all() for v in fcs.values()):
                continue
            tasks.append((s, d, y, D.price(s, d.month), fee(s, d.month), scale, fcs, check))
    return tasks


if __name__ == "__main__":
    check = "check" in sys.argv[1:]
    T = build(check)
    print(len(T), "station-day tasks", flush=True)
    res = Parallel(n_jobs=6, batch_size=4, verbose=3)(delayed(task)(*t) for t in T)
    out = pd.DataFrame([r for a in res for r in a])
    out.to_parquet(OUT / f"cb_storage_sched{'_check' if check else ''}.parquet")
    print("saved", len(out), "rows", flush=True)
