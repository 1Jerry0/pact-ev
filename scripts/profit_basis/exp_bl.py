"""电池级日前计划：不再用小时聚合 LP 的计划去驱动电池级仿真，而是直接在"15 分钟 × 电池块"上做日前 LP。
  变量（96 个 15 分钟步）：x_s 本步开始充电的块数；R_s 步初满电块数；W_s 步初待充（未接入）块数；u_s 计划缺口（块）。
  一块电池充满要占用充电位 n 步（n = 平均单次电量 / (η·单块功率) / 0.25 h，取整），第 s 步开始充的块在第 s+n−1 步可换出（与仿真器一致）；
  同时在充块数 Σ_{k<n} x_{s−k} ≤ 充电位数 C；电费按充电各步电价的平均计；日末按站内缺电量相对半满估值。
  等成本面上按 ΣR（满电块数）取最小 / 最大库存解，中点 = 二者平均；默认 = 求解器直接返回的解。
执行与规则同一机制：每 15 分钟把"满电 + 在充"的块数补到计划隐含的位置 R_s + 在充_s（规则则补到 θ·K）。
对照：滚动阈值规则、换下即充；完全预测（真实逐 15 分钟换电次数）下的同一计划；以及原来的聚合 LP 计划（若该配置已算过）。
配置：单块充电功率 {8, 12, 18, 25, 50} kW（充满约 3.3 / 2.2 / 1.5 / 1.0 / 0.5 小时）× 库容 {4, 6} 小时，功率 2.5×，罚价 ×3；另加 12、25 kW 的罚价 ×10。
用法：python exp_bl.py [天数]；python exp_bl.py ana"""
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
from scipy.sparse import lil_matrix, vstack, csr_matrix
from joblib import Parallel, delayed
import cfg
import data as D
from exp_packs import RAW, TH
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
ETA = cfg.ETA
NS = 96
QL = (0.6, 0.7, 0.8, 0.9)
GRID = [((c, 2.5, 3), pc) for pc in (8.0, 12.0, 18.0, 25.0, 50.0) for c in (4, 6)] + [((6, 2.5, 10), 12.0), ((6, 2.5, 10), 25.0), ((4, 2.5, 10), 12.0)]
_M = {}


def nsteps(pc, ebar):
    return max(int(round(ebar / (ETA * pc) / 0.25)), 1)


def mats(n):
    """与需求、K、C 无关的约束矩阵。变量：x(96) R(97) W(97) u(96) df sr。"""
    if n in _M:
        return _M[n]
    nv = 4 * NS + 4
    ix, iR, iW, iu, idf, isr = 0, NS, 2 * NS + 1, 3 * NS + 2, 4 * NS + 2, 4 * NS + 3
    m = n - 1
    A = lil_matrix((2 * NS + 3, nv))
    for s in range(NS):
        A[s, iR + s + 1], A[s, iR + s], A[s, iu + s] = 1, -1, -1          # R_{s+1} = R_s + x_{s−m} − (d_s − u_s)
        if s - m >= 0:
            A[s, ix + s - m] = -1
        A[NS + s, iW + s + 1], A[NS + s, iW + s], A[NS + s, ix + s], A[NS + s, iu + s] = 1, -1, 1, 1   # W_{s+1} = W_s + (d_s − u_s) − x_s
    A[2 * NS, iR], A[2 * NS + 1, iW] = 1, 1
    A[2 * NS + 2, iW + NS], A[2 * NS + 2, idf], A[2 * NS + 2, isr] = 1, -1, 1   # 日末缺电（块）= W_96 + 在充未满部分 = K/2 + df − sr
    for s in range(NS - n + 1, NS):
        A[2 * NS + 2, ix + s] = (s + n - NS) / n
    G = lil_matrix((NS, nv))
    for s in range(NS):
        for k in range(n):
            if s - k >= 0:
                G[s, ix + s - k] = 1
    _M[n] = (A.tocsr(), G.tocsr(), (ix, iR, iW, iu, idf, isr), nv)
    return _M[n]


def lp_bl(d, price, pen, K, C, n, ebar, sec=None, fstar=None):
    """d：逐 15 分钟需求（块）；price、pen：24 小时。返回 (x, 目标值)。sec=±1：在成本 ≤ fstar 的面上最小化 / 最大化 ΣR。"""
    A, G, (ix, iR, iW, iu, idf, isr), nv = mats(n)
    ps = np.repeat(price, 4)
    c = np.zeros(nv)
    c[ix:ix + NS] = (ebar / ETA) * np.array([np.mean([ps[(s + k) % NS] for k in range(n)]) for s in range(NS)])
    c[iu:iu + NS] = np.repeat(pen, 4) * ebar
    lo = price.min()
    c[idf], c[isr] = lo / ETA * ebar, -lo * ETA * ebar
    b = np.zeros(2 * NS + 3)
    b[:NS], b[NS:2 * NS] = -d, d
    b[2 * NS], b[2 * NS + 1], b[2 * NS + 2] = K // 2, K - K // 2, K / 2
    bounds = [(0, None)] * NS + [(0, K)] * (2 * NS + 2) + [(0, max(v, 0.0)) for v in d] + [(0, None), (0, None)]
    h = np.full(NS, float(C))
    if sec is None:
        r = linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=b, bounds=bounds, method="highs")
        return r.x, r.fun
    tol = 1e-9 * max(1.0, abs(fstar)) + 1e-7
    c2 = np.zeros(nv)
    c2[iR:iR + NS + 1] = sec
    r = linprog(c2, A_ub=vstack([G, csr_matrix(c)]), b_ub=np.r_[h, fstar + tol], A_eq=A, b_eq=b, bounds=bounds, method="highs")
    return r.x, float(c @ r.x)


def position(x, n):
    """计划隐含的逐步库存位置（满电 + 在充，块）。"""
    xs, R = x[:NS], x[NS:2 * NS + 1]
    I = np.array([xs[max(0, s - n + 1):s + 1].sum() for s in range(NS)])
    return R[:NS] + I


def run_day_bl(ev_step, ev_e, price, pen, K, C, pc, ebar, target):
    """target：96 个 15 分钟步的库存位置目标（块）。与 exp_packs.run_day 相同的仿真，只是目标逐步给。返回 (成本, 未满足电量)。"""
    need = np.zeros(K)
    need[K // 2:] = ebar
    on = np.zeros(K, bool)
    add = ETA * pc * 0.25
    cost = unserved = 0.0
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


def task(d, ev_step, ev_e, y, price, tot, hmean, fcs, ebar, grid=None, rule=True, pi=True):
    """fcs：{名称: 逐小时预测（kWh）}；返回 (规则结果, 计划结果)。"""
    out_rule, rows = [], []
    cls = np.repeat(np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6)), 4)
    ee = np.minimum(ev_e, 1.6 * ebar)
    cnt = np.bincount(ev_step, minlength=NS).astype(float)
    for par, pc in (grid or GRID):
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)
        key = (d, str(par), pc, n)
        sim = lambda tgt: run_day_bl(ev_step, ee, price, pen, K, C, pc, ebar, tgt)
        if rule:
            rc = [sim(np.where(cls == 0, K, np.where(cls == 1, a * K, b * K))) for a, b in TH]
            cf = sim(np.full(NS, K))
            out_rule.append((*key, [c for c, _ in rc], [u for _, u in rc], cf[0], cf[1], float(ee.sum())))
        if fcs is None:
            continue
        levels = {nm: np.repeat(np.maximum(f, 0) / ebar / 4, 4) for nm, f in fcs.items()}
        if pi:
            levels["pi"] = cnt
        for nm, dd in levels.items():
            for mg in ((0, 1, 2) if nm in ("mean", "pi") else (0,)):        # 计划里的充电步数裕量：单块电量有大有小，按平均电量算的步数偏乐观
                npl = n + mg
                x0, f0 = lp_bl(dd, price, pen, K, C, npl, ebar)
                xl, _ = lp_bl(dd, price, pen, K, C, npl, ebar, 1.0, f0)
                xh, _ = lp_bl(dd, price, pen, K, C, npl, ebar, -1.0, f0)
                ties = [("min", xl), ("mid", (xl + xh) / 2), ("max", xh)] + ([("default", x0)] if nm in ("mean", "pi") else [])
                width = float((xh[NS:2 * NS + 1].sum() - xl[NS:2 * NS + 1].sum()) / ((NS + 1) * K))
                for tie, x in ties:
                    c, u = sim(position(x, npl))
                    rows.append((*key, f"{nm}@{mg}|{tie}", c, u, width, float(f0)))
    return out_rule, rows


def load_events():
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
    return {d: (g.step.to_numpy(), g.energy.to_numpy()) for d, g in a.groupby("day")}, ebar


def analyse(suf=""):
    RU = pd.read_parquet(OUT / f"bl_rule{suf}.parquet")
    FC = pd.read_parquet(OUT / f"bl_fc{suf}.parquet")
    AG = pd.read_parquet(OUT / "bss_packs_value_fc.parquet") if (suf == "" and (OUT / "bss_packs_value_fc.parquet").exists()) else None
    lines, tab = [], []
    for (c, pc, n), g in RU.groupby(["cfg", "pc", "n"], sort=False):
        Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
        Ms = g.pivot(index="day", columns="th", values="short").sort_index()
        base = g.drop_duplicates("day").set_index("day")[["full", "short_full", "dem"]]
        h = FC[(FC.cfg == c) & (FC.pc == pc)]
        Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
        Fs = h.pivot(index="day", columns="cand", values="short").sort_index()
        lgb = [k for k in Fc.columns if not k.startswith("pi@")]
        pis = [k for k in Fc.columns if k.startswith("pi@")]
        days = Fc.index[Fc.index >= START]
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        pol, sh = {}, {}

        def run(name, M, Sx, cols, fb):
            ks = [pick(M[cols], d_, fb) for d_ in days]
            pol[name] = np.array([M.loc[d_, k] for d_, k in zip(days, ks)])
            sh[name] = np.array([Sx.loc[d_, k] for d_, k in zip(days, ks)])
            return ks
        run("rule_roll", Mc, Ms, list(Mc.columns), k0)
        fk = run("fc_roll", Fc, Fs, lgb, "mean@0|mid")
        run("fc_roll0", Fc, Fs, [k for k in lgb if "@0|" in k], "mean@0|mid")
        run("pi_roll", Fc, Fs, pis, "pi@0|mid")
        ALL = pd.concat([Mc.loc[Fc.index].rename(columns=lambda k: f"rule{k}"), Fc[lgb]], axis=1)
        ALLs = pd.concat([Ms.loc[Fc.index].rename(columns=lambda k: f"rule{k}"), Fs[lgb]], axis=1)
        ak = run("all_roll", ALL, ALLs, list(ALL.columns), "mean@0|mid")
        for nm, col in (("fc_default", "mean@0|default"), ("fc_min", "mean@0|min"), ("fc_mid", "mean@0|mid"), ("fc_max", "mean@0|max"), ("fc_mid1", "mean@1|mid"), ("fc_max1", "mean@1|max"), ("fc_max2", "mean@2|max"),
                        ("pi_default", "pi@0|default"), ("pi_mid", "pi@0|mid"), ("pi_max", "pi@0|max"), ("pi_max1", "pi@1|max"), ("pi_max2", "pi@2|max")):
            pol[nm], sh[nm] = Fc.loc[days, col].to_numpy(), Fs.loc[days, col].to_numpy()
        pol["rule_static"], sh["rule_static"] = Mc.loc[days, k0].to_numpy(), Ms.loc[days, k0].to_numpy()
        pol["full"], sh["full"] = base.loc[days, "full"].to_numpy(), base.loc[days, "short_full"].to_numpy()
        pol["rule_hind"] = Mc.loc[days].min(axis=1).to_numpy()
        dem = base.loc[days, "dem"].sum()
        ref = pol["rule_roll"]
        par = eval(c)
        row = dict(cfg=c, pc=pc, n=n, hours=n * 0.25, cost_rule=ref.mean(), width=float(h[h.cand == "mean@0|mid"].width.median()),
                   share_fc=float(np.mean([not str(k).startswith("rule") for k in ak])))
        s = f"{c} {pc:.0f} kW（充满 {n * 0.25:.2f} h，{len(days)} 天）：滚动规则 {ref.mean():.0f} 元/日；相对滚动规则 %："
        for nm in ("rule_static", "full", "fc_default", "fc_min", "fc_mid", "fc_max", "fc_mid1", "fc_max1", "fc_max2", "fc_roll0", "fc_roll", "all_roll", "pi_default", "pi_mid", "pi_max", "pi_max1", "pi_max2", "pi_roll", "rule_hind"):
            m, lo, hi = rel_ci(pol[nm], ref)
            s += f" {nm} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            row[nm], row[nm + "_lo"], row[nm + "_hi"] = 100 * m, 100 * lo, 100 * hi
        for a_, b_, nm in (("fc_mid", "fc_default", "mid_vs_def"), ("fc_mid", "fc_min", "mid_vs_min"), ("fc_max", "fc_default", "max_vs_def")):
            m, lo, hi = rel_ci(pol[a_], pol[b_])
            s += f"；{nm} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            row[nm], row[nm + "_lo"], row[nm + "_hi"] = 100 * m, 100 * lo, 100 * hi
        if AG is not None:
            ag = AG[(AG.cfg == c) & (AG.pc == pc)]
            if len(ag):
                Ac = ag.pivot(index="day", columns="cand", values="cost").sort_index()
                dd = [d_ for d_ in days if d_ in Ac.index]
                if len(dd) == len(days):
                    for col in ("mean|mid", "mean|max", "pi|default"):
                        m, lo, hi = rel_ci(Ac.loc[days, col].to_numpy(), ref)
                        s += f"；聚合LP {col} {100 * m:+.1f}"
                        row["agg_" + col] = 100 * m
        s += "；未满足 % " + "，".join(f"{nm} {100 * sh[nm].sum() / dem:.2f}" for nm in ("rule_roll", "fc_default", "fc_mid", "fc_max", "fc_roll", "pi_mid", "full"))
        s += f"；最优面宽度中位 {row['width']:.3f}；滚动常选 {dict(pd.Series(fk).value_counts().head(3))}"
        lines.append(s)
        tab.append(row)
    txt = "\n".join(lines)
    (OUT / f"exp_bl{suf}.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(tab).to_csv(OUT / f"exp_bl{suf}.csv", index=False, encoding="utf-8-sig")
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
    Tk = []
    for d in pd.date_range("2024-01-15", cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all() or d not in EV:
            continue
        fcs = None
        if d >= cfg.MP_EVAL_START:
            f = FQ["mean"].reindex(idx).to_numpy()
            Q = FQ[cfg.QC].reindex(idx).to_numpy()
            if not (np.isfinite(f).all() and np.isfinite(Q).all()):
                continue
            fcs = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        Tk.append((d, *EV[d], y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, fcs, ebar))
    test = len(sys.argv) > 1
    if test:
        Tk = [t for t in Tk if t[7] is not None][:int(sys.argv[1])]
    print(len(Tk), "tasks; ebar", round(ebar, 2), "; hmean", round(hmean, 1), "; 充满步数", {pc: nsteps(pc, ebar) for pc in (8, 12, 18, 25, 50)}, flush=True)
    res = Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk)
    suf = "_test" if test else ""
    RU = pd.DataFrame([(d, c, pc, n, i, cc[i], uu[i], cf, uf, dem) for b, _ in res for (d, c, pc, n, cc, uu, cf, uf, dem) in b for i in range(len(TH))],
                      columns=["day", "cfg", "pc", "n", "th", "cost", "short", "full", "short_full", "dem"])
    RU.to_parquet(OUT / f"bl_rule{suf}.parquet")
    pd.DataFrame([r for _, rr in res for r in rr], columns=["day", "cfg", "pc", "n", "cand", "cost", "short", "width", "plan_cost"]).to_parquet(OUT / f"bl_fc{suf}.parquet")
    if not test:
        analyse()
    else:
        FC = pd.read_parquet(OUT / "bl_fc_test.parquet")
        FC = FC[FC.cand.str.startswith(("mean", "pi"))]
        print(FC.groupby(["cfg", "pc", "cand"]).cost.mean().unstack("cand").round(0).to_string())
        print((RU.groupby(["cfg", "pc", "th"]).cost.mean().unstack("th").min(axis=1)).round(0).to_string())
        print(RU.drop_duplicates(["day", "cfg", "pc"]).groupby(["cfg", "pc"]).full.mean().round(0).to_string())
