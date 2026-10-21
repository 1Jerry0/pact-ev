"""电池级仿真（稳健性）：把库存换成一块块电池，用 A7 的真实逐笔换电记录驱动。
  - 站内有 K = 库容 / 平均单次换电电量 块电池；只有充满的电池才能换出；换回的电池缺电量 = 该笔会话的实际电量；
  - 每块电池在充电位上以固定功率 p_c 充电（12 kW 时一块平均要 2 小时多），同时在充的块数不超过 总功率 / p_c；
  - 15 分钟步长；电费按实际充入电量和当时电价计；换不到电池的需求按罚价计；日末按站内总缺电量相对半满估值。
计划来自带充电提前期 L 的日前 LP（exp_leadtime.lp；L 取与 p_c 相称的小时数），默认 / 中点 / 最大库存解；
执行：每小时把"充满 + 在充"的电池数补到计划的库存位置（计划在 t+L 的库存 + 该小时到货 + 提前期内的预测需求，折成块数）；
对照：完全信息计划（同一执行方式）、阈值规则（按电价时段把库存位置补到 θ·K，阈值在 1–3 月选定）、换下即充。
用法：python exp_packs.py"""
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
from analysis import boot_mean
from exp_leadtime import lp, unpack

OUT = HERE.parents[1] / "results"
RAW = ROOT / "data" / "raw" / "mp_evdata"
S = cfg.BSS
ETA = cfg.ETA
T = 24
GRID = [((4, 2.5, 3), 12.0, 2), ((6, 2.5, 3), 12.0, 2), ((8, 2.5, 3), 12.0, 2), ((6, 4.0, 3), 12.0, 2), ((6, 2.5, 10), 12.0, 2),
        ((6, 2.5, 3), 25.0, 1), ((6, 2.5, 3), 12.0, 3)]
LV = np.round(np.arange(0, 1.01, 0.1), 2)
TH = [(a, b) for a in LV for b in LV if b <= a]


def positions(plan, dhat, L):
    """计划隐含的逐时库存位置目标（kWh）：到 t+L 时的计划库存 + 该小时到货 + 提前期内的计划净需求。"""
    g, s, u = plan
    Z = np.zeros(T)
    for t in range(T):
        if t + L <= T - 1:
            Z[t] = s[t + L] + ETA * g[t] + sum(max(dhat[j], 0) - u[j] for j in range(t, t + L))
        else:
            Z[t] = s[T] + ETA * g[max(T - L, 0):t + 1].sum() + sum(max(dhat[j], 0) - u[j] for j in range(t, T))
    return Z


def run_day(ev_step, ev_e, price, pen, cap, pmax, pc, ebar, target_packs):
    """target_packs: 24 个小时的库存位置目标（块）。返回 (成本, 未满足电量, 未满足次数)。"""
    K = max(int(round(cap / ebar)), 2)
    C = max(int(pmax // pc), 1)
    need = np.zeros(K)
    need[K // 2:] = ebar                                       # 半数充满，半数缺一个平均电量：站内电量 = 半满
    on = np.zeros(K, bool)
    add = ETA * pc * 0.25
    cost = unserved = 0.0
    lost = 0
    k = 0
    for h in range(T):
        for q in range(4):
            ready = need <= 1e-9
            on &= ~ready
            want = int(round(target_packs[h])) - int(ready.sum())
            want = max(min(want, C, int((~ready).sum())), 0)
            n_on = int(on.sum())
            if want > n_on:                                    # 再接入若干块：先接缺电最少的
                idle = np.flatnonzero(~ready & ~on)
                on[idle[np.argsort(need[idle])[:want - n_on]]] = True
            elif want < n_on:                                  # 多了就停掉缺电最多的
                act = np.flatnonzero(on)
                on[act[np.argsort(-need[act])[:n_on - want]]] = False
            e = np.minimum(need[on], add)
            need[on] -= e
            cost += price[h] * e.sum() / ETA
            step = 4 * h + q
            while k < len(ev_step) and ev_step[k] == step:
                rd = np.flatnonzero(need <= 1e-9)
                if len(rd):
                    need[rd[0]] = ev_e[k]
                    on[rd[0]] = False
                else:
                    unserved += ev_e[k]
                    lost += 1
                    cost += pen[h] * ev_e[k]
                k += 1
    lo = price.min()
    gap = need.sum() - K * ebar / 2                            # 站内缺电量相对半满的差
    cost += gap * lo / ETA if gap > 0 else gap * lo * ETA
    return cost, unserved, lost


def task(d, ev_step, ev_e, y, price, tot, hmean, f, ebar):
    rows, rule = [], []
    cls = np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6))
    for par, pc, L in GRID:
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(T)))
        K = max(int(round(cap / ebar)), 2)
        ee = np.minimum(ev_e, 1.6 * ebar)
        rc = []
        for a, b in TH:
            tgt = np.where(cls == 0, K, np.where(cls == 1, a * K, b * K))
            rc.append(run_day(ev_step, ee, price, pen, cap, pmax, pc, ebar, tgt)[:2])
        rule.append((d, str(par), pc, L, [c for c, _ in rc], [u for _, u in rc]))
        if f is None:
            continue
        r = dict(day=d, cfg=str(par), pc=pc, L=L, dem=float(ee.sum()), n=len(ee), K=K)
        _, r["opt"] = lp(y, price, pen, cap, pmax, L)
        x0, f0 = lp(f, price, pen, cap, pmax, L)
        xl, _ = lp(f, price, pen, cap, pmax, L, 1.0, f0)
        xh, _ = lp(f, price, pen, cap, pmax, L, -1.0, f0)
        xp, _ = lp(y, price, pen, cap, pmax, L)
        for name, x, fc in (("default", x0, f), ("mid", (xl + xh) / 2, f), ("max", xh, f), ("pi", xp, y)):
            Z = positions(unpack(x), fc, L) / ebar
            c, u, nl = run_day(ev_step, ee, price, pen, cap, pmax, pc, ebar, Z)
            r[f"cost_{name}"], r[f"short_{name}"], r[f"lost_{name}"] = c, u, nl
        c, u, nl = run_day(ev_step, ee, price, pen, cap, pmax, pc, ebar, np.full(T, K))
        r["cost_full"], r["short_full"] = c, u
        rows.append(r)
    return rows, rule


if __name__ == "__main__":
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
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    Tk = []
    for d in pd.date_range("2024-01-15", cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all() or d not in EV:
            continue
        f = None
        if d >= cfg.MP_EVAL_START:
            f = np.maximum(G.reindex(idx).to_numpy(), 0)
            if not np.isfinite(f).all():
                continue
        elif d.dayofyear % 2:
            continue
        Tk.append((d, *EV[d], y, D.price(S, d.month), float(D.price("A7", d.month, "total").max()), hmean, f, ebar))
    if len(sys.argv) > 1:
        Tk = [t for t in Tk if t[7] is not None][:int(sys.argv[1])]
    print(len(Tk), "tasks; mean swap energy", round(ebar, 2), flush=True)
    res = Parallel(n_jobs=7, batch_size=1, verbose=2)(delayed(task)(*t) for t in Tk)
    R = pd.DataFrame([r for a_, _ in res for r in a_])
    RU = pd.DataFrame([(d, c, pc, L, i, cc[i], uu[i]) for _, b in res for (d, c, pc, L, cc, uu) in b for i in range(len(TH))], columns=["day", "cfg", "pc", "L", "th", "cost", "short"])
    RU.to_parquet(OUT / "bss_packs_rule.parquet")
    dev = RU[RU.day < cfg.MP_EVAL_START]
    if len(dev):
        roll = []                                             # 滚动选阈值：用 D−29…D−2 的成本（与预测相同的信息滞后）
        for (c_, pc_, L_), g_ in RU.groupby(["cfg", "pc", "L"]):
            Mc = g_.pivot(index="day", columns="th", values="cost").sort_index()
            Ms = g_.pivot(index="day", columns="th", values="short").sort_index()
            for d_ in Mc.index[Mc.index >= cfg.MP_EVAL_START]:
                past = Mc.loc[d_ - pd.Timedelta(days=29): d_ - pd.Timedelta(days=2)]
                k_ = int(past.sum().idxmin())
                roll.append((d_, c_, pc_, L_, Mc.loc[d_, k_], Ms.loc[d_, k_]))
        R = R.merge(pd.DataFrame(roll, columns=["day", "cfg", "pc", "L", "cost_roll", "short_roll"]), on=["day", "cfg", "pc", "L"])
        best = dev.groupby(["cfg", "pc", "L", "th"]).cost.sum().unstack("th").idxmin(axis=1).rename("best").reset_index()
        ev = RU[RU.day >= cfg.MP_EVAL_START].merge(best, on=["cfg", "pc", "L"])
        ev = ev[ev.th == ev.best][["day", "cfg", "pc", "L", "cost", "short"]].rename(columns={"cost": "cost_rule", "short": "short_rule"})
        R = R.merge(ev, on=["day", "cfg", "pc", "L"])
    R.to_parquet(OUT / "bss_packs.parquet")
    names = [n for n in ("default", "mid", "max", "pi", "rule", "roll", "full") if f"cost_{n}" in R]
    lines = []
    for (c, pc, L), g in R.groupby(["cfg", "pc", "L"]):
        ex = {n: (g[f"cost_{n}"] - g.cost_pi).sum() / len(g) for n in names}
        sh = {n: g[f"short_{n}"].sum() / g.dem.sum() for n in names}
        dd = g.set_index("day")
        from arch.bootstrap import StationaryBootstrap, optimal_block_length
        xa, xb = (dd.cost_mid - dd.cost_pi).to_numpy(), (dd.cost_default - dd.cost_pi).to_numpy()
        bs = StationaryBootstrap(max(1.0, float(optimal_block_length(xa - xb)["stationary"].iloc[0])), np.arange(len(xa)), seed=cfg.SEED)
        rr = np.array([1 - xa[i[0][0]].sum() / xb[i[0][0]].sum() for i in bs.bootstrap(2000)])
        m, ci = 1 - xa.sum() / xb.sum(), (np.quantile(rr, 0.025), np.quantile(rr, 0.975))
        vs_rule = ""
        if "cost_rule" in dd:
            xr = (dd.cost_rule - dd.cost_pi).to_numpy()
            r2 = np.array([1 - xa[i[0][0]].sum() / xr[i[0][0]].sum() for i in bs.bootstrap(2000)])
            vs_rule = f"；中点 vs 规则（超额成本降幅）{1 - xa.sum() / xr.sum():.3f} [{np.quantile(r2, 0.025):.3f}, {np.quantile(r2, 0.975):.3f}]"
            xo = (dd.cost_roll - dd.cost_pi).to_numpy()
            r3 = np.array([1 - xa[i[0][0]].sum() / xo[i[0][0]].sum() for i in bs.bootstrap(2000)])
            vs_rule += f"；中点 vs 滚动规则 {1 - xa.sum() / xo.sum():.3f} [{np.quantile(r3, 0.025):.3f}, {np.quantile(r3, 0.975):.3f}]；各策略超额占完全信息计划成本 %：" + "，".join(f"{n} {100 * (g['cost_' + n] - g.cost_pi).sum() / g.cost_pi.sum():.1f}" for n in names if n != "pi")
        s = f"{c} p_c={pc:.0f}kW L={L} K={int(g.K.iloc[0])}：相对完全信息计划的超额成本（元/日）" + "，".join(f"{n} {ex[n]:.0f}" for n in names if n != "pi")
        s += f"；中点 vs 默认（超额成本降幅）{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" + vs_rule
        s += "；未满足占比 " + "，".join(f"{n} {sh[n]:.4f}" for n in names)
        s += f"；完全信息计划日均成本 {g.cost_pi.mean():.0f}（聚合 LP 最优 {g.opt.mean():.0f}）"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_packs.txt").write_text(txt, encoding="utf-8")
    print(txt)
