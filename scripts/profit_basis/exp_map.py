"""A7 换电站"资源—价值"图：在（库容 N，充电功率 P）平面上比较 预测+计划（默认/中点/最大缓冲/各 α）与不用预测的阈值规则，
同时记录：各 α 的缺口（检验"缺口随缓冲比例单调不增"是否普遍成立）、执行后的平均库存（用于估算缓冲的持有/老化代价的盈亏平衡点）。
只读冻结流程。模型：LightGBM 单站与 5 成员组合；罚价 ×3 与 ×10；全部评估日。用法：python exp_map.py"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
import sim
from exp_bss import rule_costs, TH, mix

OUT = HERE.parents[2] / "results" / "profit_basis"
ETA = cfg.ETA
S = cfg.BSS
MODELS = ["lgb_local", "combo5"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GRID = [(c, p, m) for c in (2, 4, 6, 8) for p in (1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0) for m in (3,)]   # 利润口径下罚价档被忽略，只算一档


def exec_path(P, pl, y):
    """与 sim.bss_exec（现规则）逐步一致，另返回缺口与执行后的小时末库存均值。"""
    g_plan, s_plan, dhat, u_plan = pl[1], pl[2], pl[3], pl[4]
    cap, pmax, price = P.cap, P.pmax, P.price
    s0 = cap / 2
    s, tot, short, inv = s0, 0.0, 0.0, 0.0
    for t in range(24):
        dh = max(dhat[t], 0)
        g = (s_plan[t + 1] - s + dh - u_plan[t]) / ETA
        g = float(np.clip(g, 0, min(pmax, (cap - s + dh) / ETA)))
        avail = s + ETA * g
        served = min(y[t], avail)
        u = y[t] - served
        s = min(avail - served, cap)
        tot += price[t] * g + sim.pen_at(P.pen, t) * u
        short += u
        inv += s
    return tot + sim.terminal_value(s, s0, price.min()), short, inv / 24


def task(d, y, price, tot_max, hmean, fcs):
    sc = dict(hmean=hmean, tot_max=tot_max)
    rows, rule = [], []
    for par in GRID:
        P = sim.Problem("bss", y, price, par, sc)
        rc, rs = rule_costs(y, price, P.pen, P.cap, P.pmax)
        rule.append((d, str(par), rc - P.opt, rs))
        for m, f in fcs.items():
            pdef, pmin, pmax_ = P.plan(f, "default"), P.plan(f, "min"), P.plan(f, "max")
            r = dict(day=d, cfg=str(par), model=m, opt=P.opt, dem=y.sum(), cap_kwh=P.cap)
            c, sh, iv = exec_path(P, pdef, y)
            chk = P.cost(pdef)
            r["reg_default"], r["short_default"], r["inv_default"], r["chk"] = c - P.opt, sh, iv, abs(c - chk)
            for a in ALPHAS:
                c, sh, iv = exec_path(P, mix(pmin, pmax_, a), y)
                r[f"reg_a{a}"], r[f"short_a{a}"], r[f"inv_a{a}"] = c - P.opt, sh, iv
            rows.append(r)
    return rows, rule


if __name__ == "__main__":
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = {m: pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "mean"]) for m in MODELS}
    G = {m: F[m][F[m].series == S].set_index("t")["mean"] for m in MODELS}
    T = []
    for d in pd.date_range("2024-01-08", cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all():
            continue
        fcs = {}
        if d >= cfg.MP_EVAL_START:
            fcs = {m: np.maximum(G[m].reindex(idx).to_numpy(), 0) for m in MODELS}
            if not all(np.isfinite(v).all() for v in fcs.values()):
                fcs = {}
        T.append((d, y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, fcs))
    print(len(T), "day tasks,", len(GRID), "configs", flush=True)
    res = Parallel(n_jobs=6, batch_size=2, verbose=3)(delayed(task)(*t) for t in T)
    pd.DataFrame([r for a, _ in res for r in a]).to_parquet(OUT / "bss_map_plans.parquet")
    rule = [(d, c, i, float(reg[i]), float(sh[i])) for _, b in res for (d, c, reg, sh) in b for i in range(len(TH))]
    pd.DataFrame(rule, columns=["day", "cfg", "th", "regret", "short"]).to_parquet(OUT / "bss_map_rule.parquet")
    print("saved", flush=True)
