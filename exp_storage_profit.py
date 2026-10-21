"""储能站在利润口径下的计划价值、电价形状与 15 分钟结算检验（超限按服务费 φ_t = 零售总价 − 电度电价计价，见 exp_cost_basis_st.py）。
1. 各电价（上海、广东、PG&E EV2-A、两段式；形状按月缩放到该站当月电度电价均值，服务费不变）：
   18 个分时规则变体（exp_storage_rule.FAM）的成本，LightGBM 与季节经验分位数（近期曲线）均值预测的中点计划遗憾；规则按 D−29…D−2 滚动选。
2. 上海电价下按 15 分钟执行与结算：小时计划（中点；LightGBM 另含默认）均分到 15 分钟执行，15 分钟完全信息 LP 为基准；规则同样在 15 分钟执行并滚动选。
资源：9 种储能情景（E 为日均电量的 10%/25%/50%，L 为开发期日峰值第 30/50/70 百分位）。
输出：结果/storage_profit_tariffs.parquet、storage_profit_15min.parquet。用法：python exp_storage_profit.py [tariffs|q15|ana]"""
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
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import cfg
import data as D
import sim
from dec_mp import scales
from exp_storage_rule import FAM
from exp_tariffs import shape
from exp_cost_basis_st import CFGS, fee, exec_split

OUT = ROOT / "results"
RAW = ROOT / "data" / "raw" / "mp_evdata"
ETA = cfg.ETA
TARIFFS = ("shanghai", "guangdong", "pge_ev2a", "two_block")
Q_MODELS = ["lgb_local", "timesfm3_cov", "chronos2_cov", "combo5", "naive", "seasonal_q"]
N = 96


def price_st(name, st, month):
    base = D.price(st, month)
    if name == "shanghai":
        return base
    s = shape(name, month)
    return s * base.mean() / s.mean()


def rule_cost(y, price, E, L, phi, par):
    """exp_storage_rule.rule_cost 的逐小时 φ 版本。"""
    cmode, dmode, rho = par
    cls = np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6))
    top = cls.max()
    P = E / 2
    soc, tot = E / 2, 0.0
    for t in range(24):
        ch = cls[t] == 0 or (cmode == 1 and cls[t] == 1)
        dis = (not ch) and ((dmode == 0 and cls[t] >= 2) or (dmode == 1 and cls[t] == top and top >= 2) or (dmode == 2 and cls[t] >= 1))
        c_p = P if ch else 0.0
        d_p = min(P, max(soc - rho * E, 0.0) * ETA) if dis else 0.0
        net, soc = sim._st_step(c_p, d_p, y[t], soc, E, L, True)
        tot += price[t] * net + phi[t] * max(net - L, 0)
    return tot + sim.terminal_value(soc, E / 2, price.min())


def step15(c_p, d_p, y, soc, E, L, P):
    c = min(c_p, (E - soc) / ETA)
    d = min(d_p, y, soc * ETA)
    net = y + c - d
    if net > L:
        if c > 0:
            cut = min(c, net - L)
            c -= cut
            net -= cut
        d += max(min(net - L, P - d, y - d, soc * ETA - d), 0)
        net = y + c - d
    return net, soc + ETA * c - d / ETA


def exec15(c, d, y15, price, E, L, phi):
    soc, tot = E / 2, 0.0
    Lk, Pk = L / 4, E / 8
    for k in range(N):
        h = k // 4
        net, soc = step15(c[h] / 4, d[h] / 4, y15[k], soc, E, Lk, Pk)
        tot += price[h] * net + phi[h] * max(net - Lk, 0)
    return tot + sim.terminal_value(soc, E / 2, price.min())


def rule15(y15, price, E, L, phi, par):
    cmode, dmode, rho = par
    cls = np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6))
    top = cls.max()
    Lk, Pk = L / 4, E / 8
    soc, tot = E / 2, 0.0
    for k in range(N):
        h = k // 4
        ch = cls[h] == 0 or (cmode == 1 and cls[h] == 1)
        dis = (not ch) and ((dmode == 0 and cls[h] >= 2) or (dmode == 1 and cls[h] == top and top >= 2) or (dmode == 2 and cls[h] >= 1))
        c_p = Pk if ch else 0.0
        d_p = min(Pk, max(soc - rho * E, 0.0) * ETA) if dis else 0.0
        net, soc = step15(c_p, d_p, y15[k], soc, E, Lk, Pk)
        tot += price[h] * net + phi[h] * max(net - Lk, 0)
    return tot + sim.terminal_value(soc, E / 2, price.min())


def opt15(y15, price, E, L, phi):
    T = N
    Pk, Lk, pv = E / 8, L / 4, price.min()
    ps = np.repeat(price, 4)
    n = 4 * T + 3
    ic, idd, isx, ix, idf, isr = 0, T, 2 * T, 3 * T + 1, 4 * T + 1, 4 * T + 2
    cost = np.zeros(n)
    cost[ic:ic + T], cost[idd:idd + T], cost[ix:ix + T] = ps, -ps, np.repeat(phi, 4)
    cost[idf], cost[isr] = pv / ETA, -pv * ETA
    r = np.repeat(np.arange(T), 3)
    c = np.column_stack([ic + np.arange(T), idd + np.arange(T), ix + np.arange(T)]).ravel()
    A_ub = csr_matrix((np.tile([1, -1, -1], T), (r, c)), shape=(T, n))
    b_ub = Lk - y15
    rr, cc, vv = [], [], []
    for t in range(T):
        rr += [t] * 4; cc += [isx + t + 1, isx + t, ic + t, idd + t]; vv += [1, -1, -ETA, 1 / ETA]
    rr += [T, T + 1, T + 1, T + 1]; cc += [isx, isx + T, idf, isr]; vv += [1, 1, 1, -1]
    A_eq = csr_matrix((vv, (rr, cc)), shape=(T + 2, n))
    b_eq = np.r_[np.zeros(T), E / 2, E / 2]
    bounds = [(0, Pk)] * T + [(0, min(Pk, v)) for v in y15] + [(0, E)] * (T + 1) + [(0, None)] * T + [(0, None), (0, None)]
    return sim._solve(cost, A_ub, b_ub, A_eq, b_eq, bounds)[1] + float(ps @ y15)


def hourly_cost(c, d, y, price, E, L, phi):
    el, _, pn = exec_split(c, d, y, price, E, L, phi)
    return el + pn


def task_tariffs(st, d, y, sc, fl, fs):
    rows = []
    phi = fee(st, d.month)
    for name in TARIFFS:
        p = price_st(name, st, d.month)
        for fe, fq in CFGS:
            E, L = fe * sc["dmean"], sc["peaks"][fq]
            _, _, obj = sim.st_plan(y, p, E, L, phi, None)
            opt = obj + float(p @ y)
            rules = [rule_cost(y, p, E, L, phi, f) - opt for f in FAM]
            rl = rs = np.nan
            if fl is not None:
                c, dd, _ = sim.st_plan(fl, p, E, L, phi, "mid")
                rl = hourly_cost(c, dd, y, p, E, L, phi) - opt
                c, dd, _ = sim.st_plan(fs, p, E, L, phi, "mid")
                rs = hourly_cost(c, dd, y, p, E, L, phi) - opt
            rows.append((st, d, name, f"({fe}, {fq})", opt, rules, rl, rs))
    return rows


def task_q15(st, d, y, y15, price, sc, fcs):
    rows = []
    phi = fee(st, d.month)
    for fe, fq in CFGS:
        E, L = fe * sc["dmean"], sc["peaks"][fq]
        r = dict(station=st, day=d, cfg=f"({fe}, {fq})", rules=[rule15(y15, price, E, L, phi, f) for f in FAM])
        if fcs is not None:
            r["opt_q"] = opt15(y15, price, E, L, phi)
            for m, f in fcs.items():
                c, dd, _ = sim.st_plan(f, price, E, L, phi, "mid")
                r[f"q_{m}"] = exec15(c, dd, y15, price, E, L, phi) - r["opt_q"]
            c, dd, _ = sim.st_plan(fcs["lgb_local"], price, E, L, phi, "default")
            r["q_lgb_default"] = exec15(c, dd, y15, price, E, L, phi) - r["opt_q"]
        rows.append(r)
    return rows


def run_tariffs():
    H = D.mp_hourly()
    SC = scales(H)
    FL = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    FS = pd.read_parquet(cfg.OUT / "fc" / "mp_seasonal_q.parquet", columns=["series", "t", "mean"])
    Tk = []
    for st in cfg.POWER:
        gl, gs = FL[FL.series == st].set_index("t")["mean"], FS[FS.series == st].set_index("t")["mean"]
        for d in pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D"):
            idx = pd.date_range(d, periods=24, freq="h")
            y = H[st].reindex(idx).to_numpy()
            if not np.isfinite(y).all():
                continue
            fl = fs = None
            if d >= cfg.MP_EVAL_START:
                fl, fs = np.maximum(gl.reindex(idx).to_numpy(), 0), np.maximum(gs.reindex(idx).to_numpy(), 0)
                if not (np.isfinite(fl).all() and np.isfinite(fs).all()):
                    continue
            Tk.append((st, d, y, SC[st], fl, fs))
    print("tariffs", len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 6)), batch_size=2, verbose=2)(delayed(task_tariffs)(*t) for t in Tk)
    pd.DataFrame([r for rr in res for r in rr], columns=["station", "day", "tariff", "cfg", "opt", "rules", "lgb_mid", "sq_mid"]).to_parquet(OUT / "storage_profit_tariffs.parquet")


def run_q15():
    H = D.mp_hourly()
    SC = scales(H)
    Tk = []
    for st in cfg.POWER:
        q = pd.read_excel(RAW / "station_level_load_profile_15min.xlsx", sheet_name=st)
        q["datetime"] = pd.to_datetime(q.datetime)
        q = q.set_index("datetime").power.astype(float) * 0.25
        G = {}
        for m in Q_MODELS:
            F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "mean"])
            G[m] = F[F.series == st].set_index("t")["mean"]
        for d in pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D"):
            idx = pd.date_range(d, periods=24, freq="h")
            y = H[st].reindex(idx).to_numpy()
            y15 = q.reindex(pd.date_range(d, periods=N, freq="15min")).to_numpy()
            if not (np.isfinite(y).all() and np.isfinite(y15).all()):
                continue
            fcs = None
            if d >= cfg.MP_EVAL_START:
                fcs = {m: np.maximum(G[m].reindex(idx).to_numpy(), 0) for m in Q_MODELS}
                if not all(np.isfinite(v).all() for v in fcs.values()):
                    continue
            Tk.append((st, d, y, y15, D.price(st, d.month), SC[st], fcs))
    print("q15", len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 6)), batch_size=2, verbose=2)(delayed(task_q15)(*t) for t in Tk)
    pd.DataFrame([r for rr in res for r in rr]).to_parquet(OUT / "storage_profit_15min.parquet")


def ana():
    from exp_packs_value import rel_ci
    lines = []

    def say(s):
        lines.append(s)
        print(s)

    def pick(M, d, fb):
        past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
        return past.sum().idxmin() if len(past) >= 7 else fb

    T = pd.read_parquet(OUT / "storage_profit_tariffs.parquet")
    say("## 各电价下：滚动规则、近期曲线中点计划、LightGBM 中点计划（遗憾合计之比；遗憾差占执行成本的百分点）")
    for (tf, st), g in T.groupby(["tariff", "station"], sort=False):
        R = np.vstack(g.rules.to_list())
        g = g.assign(**{f"r{i}": R[:, i] for i in range(R.shape[1])})
        rule_d, l_d, s_d, c_d = [], [], [], []
        for c, h in g.groupby("cfg"):
            M = h.set_index("day")[[f"r{i}" for i in range(R.shape[1])]].sort_index()
            k0 = M[M.index < cfg.MP_EVAL_START].sum().idxmin()
            ev = h[h.day >= pd.Timestamp("2024-05-01")].set_index("day")
            rule_d.append(pd.Series([M.loc[d, pick(M, d, k0)] for d in ev.index], index=ev.index))
            l_d.append(ev.lgb_mid)
            s_d.append(ev.sq_mid)
            c_d.append(ev.opt)
        rr, ll, ss, oo = (pd.concat(x).groupby(level=0).sum() for x in (rule_d, l_d, s_d, c_d))
        gap = 100 * (ss.sum() - ll.sum()) / (oo.sum() + ll.sum())
        say(f"  {tf} {st}：LightGBM/规则 {1 + rel_ci(ll, rr)[0]:.2f}（{'*' if (lambda r: r[1] > 0 or r[2] < 0)(rel_ci(ll, rr)) else ''}），"
            f"近期曲线/规则 {1 + rel_ci(ss, rr)[0]:.2f}，近期曲线与 LightGBM 的遗憾差 {gap:+.2f} 个百分点（占执行成本）")
    Q = pd.read_parquet(OUT / "storage_profit_15min.parquet")
    say("\n## 15 分钟执行与结算（上海电价）")
    hi = 0
    for st, g in Q.groupby("station"):
        R = np.vstack(g.rules.to_list())
        g = g.assign(**{f"r{i}": R[:, i] for i in range(R.shape[1])})
        parts = {k: [] for k in ["rule", "lgb_local", "timesfm3_cov", "chronos2_cov", "combo5", "lgb_default", "seasonal_q"]}
        for c, h in g.groupby("cfg"):
            M = h.set_index("day")[[f"r{i}" for i in range(R.shape[1])]].sort_index()
            k0 = M[M.index < cfg.MP_EVAL_START].sum().idxmin()
            ev = h[h.day >= pd.Timestamp("2024-05-01")].set_index("day")
            parts["rule"].append(pd.Series([M.loc[d, pick(M, d, k0)] for d in ev.index], index=ev.index) - ev.opt_q)
            for m in parts:
                if m != "rule":
                    parts[m].append(ev[f"q_{m}"])
        P = {k: pd.concat(v).groupby(level=0).sum() for k, v in parts.items()}
        t = rel_ci(P["timesfm3_cov"], P["lgb_local"])
        hi += t[1] > 0
        r = rel_ci(P["lgb_local"], P["rule"])
        say(f"  {st}：TimesFM/LightGBM {1 + t[0]:.2f} [{1 + t[1]:.2f}, {1 + t[2]:.2f}]；Chronos/LightGBM {1 + rel_ci(P['chronos2_cov'], P['lgb_local'])[0]:.2f}；"
            f"LightGBM 中点/规则 {1 + r[0]:.2f} [{1 + r[1]:.2f}, {1 + r[2]:.2f}]；近期曲线/规则 {1 + rel_ci(P['seasonal_q'], P['rule'])[0]:.2f}；中点/默认 {1 + rel_ci(P['lgb_local'], P['lgb_default'])[0]:.2f}")
    say(f"  TimesFM 显著更差的站数：{hi}")
    (OUT / "storage_profit.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    mode = (sys.argv[1:] or ["tariffs"])[0]
    {"tariffs": run_tariffs, "q15": run_q15, "ana": ana}[mode]()
