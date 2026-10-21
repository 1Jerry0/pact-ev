"""成本口径重算（小时模型，A7）：缺供按顶刊口径计价，不再使用自定的惩罚倍数。
- 利润口径：缺供按该小时零售总价计少收的换电收入（Asadi & Nurre Pinkley 2022；Widrick et al. 2018），
  κ_t = 当月 A7 第 t 小时零售总价 ×（1 + 1e-5 (23 − t)），微小递减只用于在同价时段内把无法避免的缺供后移。
- 每个计划同时记录电费（含日末估值）E、缺供电量 S 和缺供损失 Pen，供服务水平口径（缺供不超过需求的 1%，
  Sun, Tan & Tsang 2018）和成本—缺货权衡曲线（Gardner 1990；Kourentzes et al. 2020）使用。
资源：库存 4/6/8 小时 × 功率 2.5/4 倍平均需求（数据锚定网格）。
策略族：9 个模型 ×（均值 + 校准分位数 0.5—0.95）× 选解（默认/最小/中点/最大）；66 组分时阈值规则；完全信息。
输出：结果/cb_hourly_plans.parquet、结果/cb_hourly_rule.parquet。用法：python exp_cost_basis.py [check]"""
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
S = cfg.BSS
POOL = ["naive", "naive7", "seasonal_q", "lqr", "lgb_local", "lgb_global", "chronos2_cov", "timesfm3_cov", "combo5"]
INPUTS = ["mean", "q50", "q60", "q70", "q80", "q90", "q95"]
CFGS = [(c, p) for c in (4, 6, 8) for p in (2.5, 4.0)]
LV = np.round(np.arange(0, 1.01, 0.1), 2)
TH = np.array([(a, b) for a in LV for b in LV if b <= a])          # (平段阈值, 峰/尖段阈值)，66 组
TIEBREAK = 1 + 1e-5 * (23 - np.arange(24))


def kappa_profit(month):
    return D.price("A7", month, "total") * TIEBREAK


def exec_split(plan, y, price, pen, cap, pmax):
    """与 sim.bss_exec（current 模式）逐行相同，只是把电费、缺供电量和缺供损失分开返回。"""
    g_plan, s_plan, dhat, u_plan = plan[1], plan[2], plan[3], plan[4]
    s0 = cap / 2
    s, E, Sh, Pn = s0, 0.0, 0.0, 0.0
    for t in range(24):
        dh = max(dhat[t], 0)
        g = (s_plan[t + 1] - s + dh - u_plan[t]) / ETA
        g = float(np.clip(g, 0, min(pmax, (cap - s + dh) / ETA)))
        avail = s + ETA * g
        served = min(y[t], avail)
        u = y[t] - served
        s = min(avail - served, cap)
        E += price[t] * g
        Sh += u
        Pn += pen[t] * u
    return E + sim.terminal_value(s, s0, price.min()), Sh, Pn


def rule_split(y, price, pen, cap, pmax):
    """不用预测的分时阈值规则（66 组向量化）：谷段充满，平段保持 a·cap，峰/尖段保持 b·cap。"""
    cls = np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6))
    s0 = cap / 2
    s = np.full(len(TH), s0)
    E, Sh, Pn = np.zeros(len(TH)), np.zeros(len(TH)), np.zeros(len(TH))
    for t in range(24):
        tgt = np.full(len(TH), cap) if cls[t] == 0 else (TH[:, 0] * cap if cls[t] == 1 else TH[:, 1] * cap)
        g = np.clip((tgt - s) / ETA, 0, pmax)
        avail = s + ETA * g
        served = np.minimum(y[t], avail)
        u = y[t] - served
        s = np.minimum(avail - served, cap)
        E += price[t] * g
        Sh += u
        Pn += pen[t] * u
    lo = price.min()
    E += np.where(s < s0, (s0 - s) * lo / ETA, (s0 - s) * lo * ETA)
    return E, Sh, Pn


def mix(lo, hi, a):
    return ("bss", (1 - a) * lo[1] + a * hi[1], (1 - a) * lo[2] + a * hi[2], lo[3], (1 - a) * lo[4] + a * hi[4])


def task(d, y, price, hmean, fcs, check):
    pen = kappa_profit(d.month)
    rows, rule = [], []
    for c, p in CFGS:
        cap, pmax = c * hmean, p * hmean
        g0, s0_, obj, u0 = sim.bss_plan(y, price, pen, cap, pmax, None)
        Pn_opt = float(pen @ u0)
        base = dict(day=d, cfg=f"({c}, {p})", dem=float(y.sum()), E_opt=obj - Pn_opt, S_opt=float(u0.sum()), Pen_opt=Pn_opt)
        E, Sh, Pn = rule_split(y, price, pen, cap, pmax)
        rule += [dict(base, th=i, E=float(E[i]), S=float(Sh[i]), Pen=float(Pn[i])) for i in range(len(TH))]
        for m, F in fcs.items():
            mae = float(np.abs(y - F["mean"]).mean())
            for k in INPUTS:
                f = np.maximum(F[k], 0)
                pl = {}
                for tie in ("default", "min", "max"):
                    gg, ss, _, uu = sim.bss_plan(f, price, pen, cap, pmax, tie)
                    pl[tie] = ("bss", gg, ss, f, uu)
                pl["mid"] = mix(pl["min"], pl["max"], 0.5)
                for tie, plan in pl.items():
                    e, sh, pn = exec_split(plan, y, price, pen, cap, pmax)
                    rows.append(dict(base, model=m, inp=k, tie=tie, E=e, S=sh, Pen=pn, mae=mae))
        if check and "lgb_local" in fcs:          # 复现检验：旧口径（×1 当日最高零售价）下应与 bss_anchor_plans 一致
            tot = float(D.price("A7", d.month, "total").max())
            P = sim.Problem("bss", y, price, (c, p, 1), dict(hmean=hmean, tot_max=tot))
            f = np.maximum(fcs["lgb_local"]["mean"], 0)
            rows.append(dict(base, model="check_x1max", inp="mean", tie="default", E=P.cost(P.plan(f, "default")) - P.opt, S=np.nan, Pen=np.nan, mae=np.nan))
    return rows, rule


def build(check=False):
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    G = {}
    for m in POOL:
        F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}_cal.parquet")
        G[m] = F[F.series == S].set_index("t")
    days = pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D")
    if check:
        days = days[days >= cfg.MP_EVAL_START][:10]
    tasks = []
    for d in days:
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all():
            continue
        fcs = {}
        if d >= cfg.MP_EVAL_START:
            for m in POOL:
                g = G[m].reindex(idx)
                v = {k: g[k].to_numpy() for k in INPUTS}
                if all(np.isfinite(a).all() for a in v.values()):
                    fcs[m] = v
            if len(fcs) < len(POOL):
                fcs = {}
        tasks.append((d, y, D.price(S, d.month), hmean, fcs, check))
    return tasks


if __name__ == "__main__":
    check = "check" in sys.argv[1:]
    T = build(check)
    print(len(T), "day tasks", flush=True)
    res = Parallel(n_jobs=6, batch_size=2, verbose=3)(delayed(task)(*t) for t in T)
    rows = pd.DataFrame([r for a, _ in res for r in a])
    rule = pd.DataFrame([r for _, b in res for r in b])
    suf = "_check" if check else ""
    rows.to_parquet(OUT / f"cb_hourly_plans{suf}.parquet")
    rule.to_parquet(OUT / f"cb_hourly_rule{suf}.parquet")
    print("saved", len(rows), "plan rows,", len(rule), "rule rows", flush=True)
