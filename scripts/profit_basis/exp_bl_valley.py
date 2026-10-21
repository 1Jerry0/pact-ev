"""电池级仿真中各策略的谷时段充电电量占比（投稿版补充：对应“试点城市 60% 以上充电电量集中在低谷”的政策目标）。
与 exp_bl.py 相同的计划、规则与仿真，只在仿真里按电价档记录从电网购入的电量。配置：6 小时库存、2.5 倍功率、12 / 25 kW。
另核对成本与 exp_bl.csv 一致。用法：python exp_bl_valley.py [天数]；python exp_bl_valley.py ana"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(HERE.parent))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + str(HERE.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
from profit_kappa import HK
from exp_bl import lp_bl, position, nsteps, load_events, NS, QL, ETA, S
from exp_packs import TH
from exp_packs_value import pick, rel_ci, START

OUT = ROOT / "results" / "profit_basis"
GRID = [((6, 2.5, 3), 12.0), ((6, 2.5, 3), 25.0)]


def run_day_v(ev_step, ev_e, price, pen, K, C, pc, ebar, target, cls):
    """exp_bl.run_day_bl 的副本，另按电价档（0 = 谷）累计从电网购入的电量（kWh）。"""
    need = np.zeros(K)
    need[K // 2:] = ebar
    on = np.zeros(K, bool)
    add = ETA * pc * 0.25
    cost = unserved = 0.0
    grid = np.zeros(int(cls.max()) + 1)
    k = 0
    for step in range(NS):
        h = step // 4
        ready = need <= 1e-9
        on &= ~ready
        want = int(round(target[step])) - int(ready.sum())
        want = max(min(want, C, int((~ready).sum())), 0)
        n_on = int(on.sum())
        if want > n_on:
            idle = np.flatnonzero(~ready & ~on)
            on[idle[np.argsort(need[idle])[:want - n_on]]] = True
        elif want < n_on:
            act = np.flatnonzero(on)
            on[act[np.argsort(-need[act])[:n_on - want]]] = False
        e = np.minimum(need[on], add)
        need[on] -= e
        cost += price[h] * e.sum() / ETA
        grid[cls[h]] += e.sum() / ETA
        while k < len(ev_step) and ev_step[k] == step:
            rd = np.flatnonzero(need <= 1e-9)
            if len(rd):
                need[rd[0]] = ev_e[k]
                on[rd[0]] = False
            else:
                unserved += ev_e[k]
                cost += pen[h] * ev_e[k]
            k += 1
    lo = price.min()
    gap = need.sum() - K * ebar / 2
    cost += gap * lo / ETA if gap > 0 else gap * lo * ETA
    return cost, unserved, grid


def task(d, ev_step, ev_e, y, price, tot, hmean, fcs, ebar):
    rows = []
    cls_h = np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6))
    cls = np.repeat(cls_h, 4)
    top = int(cls_h.max())
    ee = np.minimum(ev_e, 1.6 * ebar)
    cnt = np.bincount(ev_step, minlength=NS).astype(float)
    for par, pc in GRID:
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)

        def rec(cand, tgt):
            c, u, g = run_day_v(ev_step, ee, price, pen, K, C, pc, ebar, tgt, cls_h)
            rows.append(dict(day=d, cfg=str(par), pc=pc, cand=cand, cost=c, short=u, e_tot=g.sum(), e_valley=g[0], e_top=g[top],
                             e_peakplus=g[2:].sum() if len(g) > 2 else g[-1]))
        for i, (a, b) in enumerate(TH):
            rec(f"rule{i}", np.where(cls == 0, K, np.where(cls == 1, a * K, b * K)))
        rec("full", np.full(NS, K))
        levels = {nm: np.repeat(np.maximum(f, 0) / ebar / 4, 4) for nm, f in fcs.items()}
        levels["pi"] = cnt
        for nm, dd in levels.items():
            for mg in ((0, 1, 2) if nm in ("mean", "pi") else (0,)):
                npl = n + mg
                x0, f0 = lp_bl(dd, price, pen, K, C, npl, ebar)
                xl, _ = lp_bl(dd, price, pen, K, C, npl, ebar, 1.0, f0)
                xh, _ = lp_bl(dd, price, pen, K, C, npl, ebar, -1.0, f0)
                ties = [("min", xl), ("mid", (xl + xh) / 2), ("max", xh)] + ([("default", x0)] if nm in ("mean", "pi") else [])
                for tie, x in ties:
                    rec(f"{nm}@{mg}|{tie}", position(x, npl))
    return rows


def analyse():
    R = pd.read_parquet(OUT / "bl_valley.parquet")
    lines = []
    for (c, pc), g in R.groupby(["cfg", "pc"], sort=False):
        M = g.pivot(index="day", columns="cand", values="cost").sort_index()
        V = {k: g.pivot(index="day", columns="cand", values=k).sort_index() for k in ("e_tot", "e_valley", "e_top", "e_peakplus", "short")}
        days = M.index[M.index >= START]
        rules = [k for k in M.columns if k.startswith("rule")]
        lgb = [k for k in M.columns if not k.startswith(("rule", "pi@", "full"))]
        pis = [k for k in M.columns if k.startswith("pi@")]
        k0 = rules[0]
        sel = {}
        for nm, cols, fb in (("rule_roll", rules, k0), ("fc_roll", lgb, "mean@0|mid"), ("pi_roll", pis, "pi@0|mid")):
            sel[nm] = [pick(M[cols], d_, fb) for d_ in days]
        for nm, col in (("full", "full"), ("fc_default", "mean@0|default"), ("fc_min", "mean@0|min"), ("fc_mid", "mean@0|mid"), ("fc_max", "mean@0|max")):
            sel[nm] = [col] * len(days)
        cost = {nm: np.array([M.loc[d_, k] for d_, k in zip(days, ks)]) for nm, ks in sel.items()}
        s = f"{c} {pc:.0f} kW（{len(days)} 天）"
        for nm, ks in sel.items():
            tot = sum(V["e_tot"].loc[d_, k] for d_, k in zip(days, ks))
            val = sum(V["e_valley"].loc[d_, k] for d_, k in zip(days, ks))
            topv = sum(V["e_top"].loc[d_, k] for d_, k in zip(days, ks))
            pk = sum(V["e_peakplus"].loc[d_, k] for d_, k in zip(days, ks))
            m = rel_ci(cost[nm], cost["rule_roll"])[0]
            s += f"\n  {nm}: 谷时段电量占比 {100 * val / tot:.1f}%，峰及尖峰 {100 * pk / tot:.1f}%，最高价档 {100 * topv / tot:.1f}%，日均购电 {tot / len(days):.0f} kWh，成本相对滚动规则 {100 * m:+.1f}%"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_valley.txt").write_text(txt, encoding="utf-8")
    print(txt)
    ref = pd.read_csv(OUT / "exp_bl.csv")
    print(ref[(ref.cfg == "(6, 2.5, 3)") & ref.pc.isin([12.0, 25.0])][["pc", "cost_rule", "fc_default", "fc_min", "fc_mid", "fc_max", "fc_roll", "full"]].to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all() or d not in EV:
            continue
        f = FQ["mean"].reindex(idx).to_numpy()
        Q = FQ[cfg.QC].reindex(idx).to_numpy()
        if not (np.isfinite(f).all() and np.isfinite(Q).all()):
            continue
        fcs = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        Tk.append((d, *EV[d], y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, fcs, ebar))
    if len(sys.argv) > 1:
        Tk = Tk[:int(sys.argv[1])]
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk)
    R = pd.DataFrame([r for rr in res for r in rr])
    R.to_parquet(OUT / ("bl_valley.parquet" if len(sys.argv) == 1 else "bl_valley_test.parquet"))
    if len(sys.argv) == 1:
        analyse()
    else:
        print(R.groupby(["pc", "cand"]).cost.mean().loc[(slice(None), ["full", "mean@0|default", "mean@0|mid", "mean@0|max"])].round(0))
        print((R.groupby(["pc", "cand"]).e_valley.sum() / R.groupby(["pc", "cand"]).e_tot.sum()).loc[(slice(None), ["full", "rule0", "mean@0|default", "mean@0|mid", "mean@0|max"])].round(3))
