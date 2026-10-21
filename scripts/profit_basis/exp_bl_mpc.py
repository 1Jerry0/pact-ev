"""电池级的滚动重排（MPC）与日内更新预测：检验"预测管线只用日前信息、规则却实时反应"是否让比较对预测不公平。
每个整点（每 4 个 15 分钟步）从仿真器的当前状态（满电块数、在充各块的剩余步数、待充块数）出发，用电池级 LP 重排当天剩余各步，
随后 4 步按新计划隐含的库存位置执行。三种信息：
  da  日前 LightGBM 预测（只增加状态反馈）；
  id  日内滚动更新的预测（fc_intraday.py；每个整点用截至上一小时的实际需求更新剩余各小时）；
  pih 实际逐小时需求（上界）。
候选：均值 × 充电步数裕量 {0,1,2} × {最大库存, 中点, 默认}；分位 0.6–0.9 × 最大库存。与规则相同的办法逐日滚动选候选（D−29…D−2）。
对照取自 bl_rule.parquet（滚动阈值规则）与 bl_fc.parquet（日前计划、不重排的调参管线）。
用法：python exp_bl_mpc.py [天数]；python exp_bl_mpc.py ana"""
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
from scipy.optimize import linprog
from scipy.sparse import vstack, csr_matrix
from joblib import Parallel, delayed
import cfg
import data as D
from exp_bl import mats, nsteps, load_events, run_day_bl, lp_bl, position, NS, ETA
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((6, 2.5, 3), pc) for pc in (8.0, 12.0, 18.0, 25.0, 50.0)] + [((6, 2.5, 10), 12.0), ((6, 2.5, 10), 25.0)]
QL = (0.6, 0.7, 0.8, 0.9)
QM = (0.7, 0.8, 0.9)
CANDS = [("mean", mg, "max") for mg in (0, 1, 2)] + [("mean", mg, "mid") for mg in (1, 2)] + [(f"q{q}", 0, "max") for q in QM]
CANDS_PI = [("mean", mg, "max") for mg in (0, 1, 2)] + [("mean", mg, "mid") for mg in (1, 2)]
SAME = ["mean@0|max", "mean@1|max", "mean@2|max", "mean@1|mid", "mean@2|mid", "q0.7@0|max", "q0.8@0|max", "q0.9@0|max"]   # 日前计划里的同一批候选
_C = {}


def replan(s0, need, on, d, price, pen, K, C, n, ebar, add, tie):
    """从第 s0 步初的状态重排到日末。need / on：仿真器状态；d：96 步需求预测（块，s0 之前的不用）。返回 96 步库存位置目标。"""
    A, G, (ix, iR, iW, iu, idf, isr), nv = mats(n)
    ready = need <= 1e-9
    rem = np.ceil(need[on & ~ready] / add - 1e-9).astype(int)            # 在充各块还要占充电位的步数
    arr, occ = np.zeros(NS), np.zeros(NS)
    endmiss = 0.0
    for r, nd in zip(rem, need[on & ~ready]):
        fin = s0 + r - 1
        occ[s0:min(fin, NS - 1) + 1] += 1
        if fin <= NS - 1:
            arr[fin] += 1
        else:
            endmiss += (nd - (NS - s0) * add) / ebar
    dd = np.where(np.arange(NS) >= s0, np.maximum(d, 0), 0.0)
    ck = (n, price.tobytes(), pen.tobytes())
    if ck not in _C:
        ps = np.repeat(price, 4)
        c = np.zeros(nv)
        c[ix:ix + NS] = (ebar / ETA) * np.array([np.mean([ps[(s + k) % NS] for k in range(n)]) for s in range(NS)])
        c[iu:iu + NS] = np.repeat(pen, 4) * ebar
        lo = price.min()
        c[idf], c[isr] = lo / ETA * ebar, -lo * ETA * ebar
        _C[ck] = c
    c = _C[ck]
    b = np.zeros(2 * NS + 3)
    b[:NS], b[NS:2 * NS] = -dd + arr, dd
    R0, W0 = int(ready.sum()), int((~ready & ~on).sum())
    b[2 * NS], b[2 * NS + 1], b[2 * NS + 2] = R0, W0, K / 2 - endmiss
    bounds = [(0, 0) if s < s0 else (0, None) for s in range(NS)] + [(0, K)] * (2 * NS + 2) + [(0, v) for v in dd] + [(0, None), (0, None)]
    h = np.maximum(C - occ, 0.0)
    r0 = linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if r0.status != 0:
        return None
    if tie == "default":
        x = r0.x
    else:
        tol = 1e-9 * max(1.0, abs(r0.fun)) + 1e-7
        A2, b2 = vstack([G, csr_matrix(c)]), np.r_[h, r0.fun + tol]
        c2 = np.zeros(nv)
        xs = []
        for sec in ((-1.0,) if tie == "max" else (1.0,) if tie == "min" else (1.0, -1.0)):
            c2[iR:iR + NS + 1] = sec
            r2 = linprog(c2, A_ub=A2, b_ub=b2, A_eq=A, b_eq=b, bounds=bounds, method="highs")
            xs.append(r2.x if r2.status == 0 else r0.x)
        x = np.mean(xs, axis=0)
    return position(x, n) + occ


def run_day_mpc(ev_step, ev_e, price, pen, K, C, pc, ebar, npl, fc, tie, every=4):
    """fc(h) → 96 步需求预测（块）。其余与 run_day_bl 相同。"""
    need = np.zeros(K)
    need[K // 2:] = ebar
    on = np.zeros(K, bool)
    add = ETA * pc * 0.25
    cost = unserved = 0.0
    k = 0
    target = np.full(NS, float(K))
    for step in range(NS):
        h = step // 4
        ready = need <= 1e-9
        on &= ~ready
        if step % every == 0:
            t = replan(step, need, on, fc(h), price, pen, K, C, npl, ebar, add, tie)
            if t is not None:
                target = t
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
    return cost, unserved


def task(d, ev_step, ev_e, y, price, tot, hmean, DA, ID, ebar, grid=None, check=False):
    """DA：{水平: 24 小时日前预测}；ID：{水平: 24×24 矩阵，[发布整点, 目标小时]，发布整点之前的目标小时为 nan}。"""
    rows = []
    ee = np.minimum(ev_e, 1.6 * ebar)
    to_steps = lambda f: np.repeat(np.maximum(np.nan_to_num(f), 0) / ebar / 4, 4)
    for par, pc in (grid or GRID):
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)
        key = (d, str(par), pc)
        for lv, mg, tie in CANDS:
            f_da = to_steps(DA[lv])
            c, u = run_day_mpc(ev_step, ee, price, pen, K, C, pc, ebar, n + mg, lambda h: f_da, tie)
            rows.append((*key, f"da:{lv}@{mg}|{tie}", c, u))
            M = ID[lv]
            c, u = run_day_mpc(ev_step, ee, price, pen, K, C, pc, ebar, n + mg, lambda h: to_steps(M[h]), tie)
            rows.append((*key, f"id:{lv}@{mg}|{tie}", c, u))
            if check:                         # 只在 0 点排一次时应与日前计划（exp_bl）逐元相同
                c1, _ = run_day_mpc(ev_step, ee, price, pen, K, C, pc, ebar, n + mg, lambda h: f_da, tie, every=NS)
                x0, f0 = lp_bl(f_da, price, pen, K, C, n + mg, ebar)
                xl, _ = lp_bl(f_da, price, pen, K, C, n + mg, ebar, 1.0, f0)
                xh, _ = lp_bl(f_da, price, pen, K, C, n + mg, ebar, -1.0, f0)
                c2, _ = run_day_bl(ev_step, ee, price, pen, K, C, pc, ebar, position(xh if tie == "max" else (xl + xh) / 2, n + mg))
                rows.append((*key, f"chk:{lv}@{mg}|{tie}", c1 - c2, 0.0))
        f_pi = to_steps(y)
        for lv, mg, tie in CANDS_PI:
            c, u = run_day_mpc(ev_step, ee, price, pen, K, C, pc, ebar, n + mg, lambda h: f_pi, tie)
            rows.append((*key, f"pih:{lv}@{mg}|{tie}", c, u))
    return rows


def analyse(suf=""):
    R = pd.read_parquet(OUT / f"bl_mpc{suf}.parquet")
    RU = pd.read_parquet(OUT / "bl_rule.parquet")
    FC = pd.read_parquet(OUT / "bl_fc.parquet")
    lines, tab = [], []
    for (c, pc), g in R.groupby(["cfg", "pc"], sort=False):
        M = g.pivot(index="day", columns="cand", values="cost").sort_index()
        U = g.pivot(index="day", columns="cand", values="short").sort_index()
        ru = RU[(RU.cfg == c) & (RU.pc == pc)]
        Mc = ru.pivot(index="day", columns="th", values="cost").sort_index()
        dem = ru.drop_duplicates("day").set_index("day").dem
        h = FC[(FC.cfg == c) & (FC.pc == pc)]
        Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
        days = M.index[M.index >= START]
        if len(days) < 30:
            continue
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        rule = np.array([Mc.loc[d_, pick(Mc, d_, k0)] for d_ in days])
        lgbc = [k for k in Fc.columns if not k.startswith("pi@")]
        ol = np.array([Fc.loc[d_, pick(Fc[lgbc], d_, "mean@0|mid")] for d_ in days])
        pol, pk = {"open_loop": ol, "open_same": np.array([Fc.loc[d_, pick(Fc[SAME], d_, "mean@1|mid")] for d_ in days])}, {}
        for typ in ("da", "id", "pih"):
            cols = [k for k in M.columns if k.startswith(typ + ":")]
            ks = [pick(M[cols], d_, f"{typ}:mean@1|mid") for d_ in days]
            pol[typ] = np.array([M.loc[d_, k] for d_, k in zip(days, ks)])
            pk[typ] = pd.Series(ks).value_counts().head(3).to_dict()
        both = [k for k in M.columns if k.startswith(("da:", "id:"))]
        row = dict(cfg=c, pc=pc, days=len(days), rule=rule.mean())
        s = f"{c} {pc:.0f} kW（{len(days)} 天）：滚动规则 {rule.mean():.0f} 元/日；相对规则 %："
        for nm, lab in (("open_loop", "日前计划不重排（调参，24 个候选）"), ("open_same", "日前计划不重排（调参，同 8 个候选）"), ("da", "日前预测 + 整点重排（调参）"), ("id", "日内预测 + 整点重排（调参）"), ("pih", "实际逐时需求 + 整点重排（调参）")):
            m, lo, hi = rel_ci(pol[nm], rule)
            s += f" {lab} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
            row[nm], row[nm + "_lo"], row[nm + "_hi"] = 100 * m, 100 * lo, 100 * hi
        for a_, b_, lab in (("da", "open_same", "重排 vs 不重排（日前预测，同候选）"), ("id", "da", "日内预测 vs 日前预测（都重排）"), ("id", "open_loop", "日内预测 + 重排 vs 日前计划不重排（24 个候选）")):
            m, lo, hi = rel_ci(pol[a_], pol[b_])
            s += f" {lab} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
            row[a_ + "_vs_" + b_], row[a_ + "_vs_" + b_ + "_lo"], row[a_ + "_vs_" + b_ + "_hi"] = 100 * m, 100 * lo, 100 * hi
        fixed = {k: 100 * (M.loc[days, k].sum() / rule.sum() - 1) for k in ("da:mean@0|max", "id:mean@0|max", "pih:mean@0|max", "da:mean@1|max", "id:mean@1|max", "pih:mean@1|max", "da:q0.8@0|max", "id:q0.8@0|max")}
        for k in ("mean@0|max", "mean@1|max", "q0.8@0|max"):
            fixed["不重排 " + k] = 100 * (Fc.loc[days, k].sum() / rule.sum() - 1)
        s += " 固定候选：" + "，".join(f"{k} {v:+.1f}" for k, v in fixed.items())
        s += "；未满足 % " + "，".join(f"{typ} {100 * np.array([U.loc[d_, pick(M[[k for k in M.columns if k.startswith(typ + ':')]], d_, typ + ':mean@1|mid')] for d_ in days]).sum() / dem.loc[days].sum():.2f}" for typ in ("da", "id"))
        s += f"；常选 da {pk['da']}，id {pk['id']}"
        lines.append(s)
        tab.append(row)
    txt = "\n".join(lines)
    (OUT / f"exp_bl_mpc{suf}.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(tab).to_csv(OUT / f"exp_bl_mpc{suf}.csv", index=False, encoding="utf-8-sig")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    IDF = pd.read_parquet(OUT / "fc_intraday_a7.parquet")
    IDF["day"], IDF["hour"] = IDF.t.dt.normalize(), IDF.t.dt.hour
    IDF["origin"] = IDF.hour - IDF.lead
    idg = {d: g for d, g in IDF.groupby("day")}
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        f = FQ["mean"].reindex(idx).to_numpy()
        Q = FQ[cfg.QC].reindex(idx).to_numpy()
        if not (np.isfinite(y).all() and np.isfinite(f).all() and np.isfinite(Q).all()) or d not in EV or d not in idg or len(idg[d]) != 300:
            continue
        DA = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        g = idg[d]
        ID = {}
        for lv in ["mean"] + [f"q{q}" for q in QL]:
            M = np.full((24, 24), np.nan)
            M[g.origin.to_numpy(), g.hour.to_numpy()] = g[lv].to_numpy()
            ID[lv] = M
        Tk.append((d, *EV[d], y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, DA, ID, ebar))
    test = len(sys.argv) > 1
    if test:
        Tk = Tk[40:40 + int(sys.argv[1])]
        res = Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t, grid=GRID[1:4:2], check=True) for t in Tk)
        T = pd.DataFrame([r for rr in res for r in rr], columns=["day", "cfg", "pc", "cand", "cost", "short"])
        chk = T[T.cand.str.startswith("chk:")]
        print("一致性检查（只在 0 点排一次 − 日前计划）最大绝对差：", float(chk.cost.abs().max()))
        print(T[~T.cand.str.startswith("chk:")].groupby(["pc", "cand"]).cost.mean().unstack("pc").round(0).to_string())
        sys.exit()
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 7)), batch_size=1, verbose=5)(delayed(task)(*t) for t in Tk)
    pd.DataFrame([r for rr in res for r in rr], columns=["day", "cfg", "pc", "cand", "cost", "short"]).to_parquet(OUT / "bl_mpc.parquet")
    analyse()
