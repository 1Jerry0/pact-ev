"""电池级仿真里的误差价格与 DSWE（把论文的核心概念放到电池级上重做一遍，对应小时模型的 5.2–5.3 节与预设检验 P3）。
价格：以实际逐小时需求为"预测"做电池级日前计划（exp_bl.lp_bl），在真实换电记录上仿真得到基准成本；
把第 t 小时的"预测"减少 / 增加 δ（10% 开发期平均小时需求），重做计划、重新仿真，成本增量 / δ 即低估 / 高估价格 π^u_t、π^o_t。
选解规则：求解器默认解、中点、最大库存解（裕量 0），与 bl_fc_models.parquet 里九个模型的 "mean@0|{tie}" 成本对应。
DSWE_m = Σ_t π^u_t (y−f)^+ + π^o_t (f−y)^+；事前版本用 D−29…D−2 的平均价格。
检验：同一天、同一配置、同一选解规则下九个模型的成本差，能被 MAE、RMSE、CRPS 解释多少，加上 DSWE 后多少（组内去均值、标准化，Driscoll–Kraay 标准误）。
用法：python exp_bl_prices.py [天数]；python exp_bl_prices.py ana"""
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
import sim
from analysis import MAIN_POOL
from exp_bl import lp_bl, position, run_day_bl, nsteps, load_events, NS

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((6, 2.5, 3), 12.0), ((6, 2.5, 3), 18.0), ((6, 2.5, 3), 25.0), ((4, 2.5, 3), 25.0)]
TIES = ("default", "mid", "max")
REL = 0.1


def task(d, ev_step, ev_e, y, price, tot, hmean, ebar):
    rows = []
    ee = np.minimum(ev_e, 1.6 * ebar)
    delta = REL * hmean
    for par, pc in GRID:
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)

        def costs(f):
            dd = np.repeat(np.maximum(f, 0) / ebar / 4, 4)
            x0, f0 = lp_bl(dd, price, pen, K, C, n, ebar)
            xl, _ = lp_bl(dd, price, pen, K, C, n, ebar, 1.0, f0)
            xh, _ = lp_bl(dd, price, pen, K, C, n, ebar, -1.0, f0)
            out = {}
            for tie, x in (("default", x0), ("mid", (xl + xh) / 2), ("max", xh)):
                out[tie] = run_day_bl(ev_step, ee, price, pen, K, C, pc, ebar, position(x, n))[0]
            return out
        base = costs(y)
        wu, wo = {t: np.zeros(24) for t in TIES}, {t: np.zeros(24) for t in TIES}
        for h in range(24):
            up, dn = y.copy(), y.copy()
            up[h] += delta
            dn[h] = max(dn[h] - delta, 0.0)
            cu, cd = costs(up), costs(dn)
            for t in TIES:
                wo[t][h] = (cu[t] - base[t]) / delta
                wu[t][h] = (cd[t] - base[t]) / max(y[h] - dn[h], 1e-9)
        for t in TIES:
            rows.append((d, str(par), pc, t, base[t], wu[t], wo[t]))
    return rows


def dk_fit(X, y, rhs, tcode):
    import statsmodels.api as sm
    return sm.OLS(y, X[rhs]).fit(cov_type="hac-groupsum", cov_kwds={"time": tcode, "maxlags": 7})


def analyse():
    from exp_packs_value import rel_ci
    P = pd.read_pickle(OUT / "bl_prices.pkl")
    F = pd.read_parquet(OUT / "bl_fc_models.parquet")
    H = D.mp_hourly()
    y_all = H[S]
    hmean = float(y_all.loc[:cfg.MP_DEV_END].mean())
    fc = {}
    for m in MAIN_POOL:
        X = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet")
        fc[m] = X[X.series == S].set_index("t")
    lines = []
    # 1. 价格的样子：均值、集中度、与小时模型的对比
    hp = pd.read_pickle(OUT / "rev_prices_a7.pkl")
    hp = hp[hp.cfg == "(6, 2.5, 3)"]
    hp = hp[hp.day >= cfg.MP_EVAL_START]
    pr_mean = float(np.mean([D.price(S, d.month).mean() for d in hp.day]))
    lines.append(f"电价均值 {pr_mean:.3f} 元/kWh；小时模型（6 h, 2.5×, ×3）平均价格：默认解 低估 {np.mean(np.vstack(hp.wu0)):.2f} / 高估 {np.mean(np.vstack(hp.wo0)):.2f}；中点解 {np.mean(np.vstack(hp.wu)):.2f} / {np.mean(np.vstack(hp.wo)):.2f}")
    for (c, pc, tie), g in P.groupby(["cfg", "pc", "tie"], sort=False):
        Wu, Wo = np.vstack(g.wu), np.vstack(g.wo)
        top = np.sort(np.maximum(Wu.mean(0), 0))[::-1]
        lines.append(f"电池级 {c} {pc:.0f} kW {tie}：低估价格均值 {Wu.mean():.2f}，高估 {Wo.mean():.2f}；低估价格为负的小时占 {100 * (Wu < -1e-9).mean():.0f}%，高估价格≈0（|·|<0.01）的占 {100 * (np.abs(Wo) < 0.01).mean():.0f}%；"
                     f"最贵 6 小时占低估价格之和 {100 * top[:6].sum() / max(top.sum(), 1e-9):.0f}%；按小时均值（0…23）低估 " + " ".join(f"{v:.2f}" for v in Wu.mean(0)))
    # 2. 跨模型回归
    recs = []
    Pd = {(r.day, r.cfg, r.pc, r.tie): (r.wu, r.wo) for r in P.itertuples(index=False)}
    days = sorted(P.day.unique())
    for (c, pc, tie), g in P.groupby(["cfg", "pc", "tie"], sort=False):
        cand = f"mean@0|{tie}"
        Fc = F[(F.cfg == c) & (F.pc == pc) & (F.cand == cand)].set_index(["day", "model"]).cost
        for d in days:
            if (d, c, pc, tie) not in Pd:
                continue
            idx = pd.date_range(d, periods=24, freq="h")
            y = y_all.reindex(idx).to_numpy()
            wu, wo = Pd[(d, c, pc, tie)]
            past = [Pd[(d - pd.Timedelta(days=k), c, pc, tie)] for k in range(2, 30) if (d - pd.Timedelta(days=k), c, pc, tie) in Pd]
            ex = (np.mean([p[0] for p in past], 0), np.mean([p[1] for p in past], 0)) if len(past) >= 14 else (None, None)
            for m in MAIN_POOL:
                if (d, m) not in Fc.index:
                    continue
                f = fc[m]["mean"].reindex(idx).to_numpy()
                Q = fc[m][cfg.QC].reindex(idx).to_numpy()
                e = y - f
                r = dict(cfg=c, pc=pc, tie=tie, day=d, model=m, cost=float(Fc.loc[(d, m)]), mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                         crps=float(np.mean(sim.crps_q(y, Q))), bias=e.mean(),
                         dswe=float((wu * np.maximum(e, 0) + wo * np.maximum(-e, 0)).sum()),
                         dswe_pos=float((np.maximum(wu, 0) * np.maximum(e, 0) + np.maximum(wo, 0) * np.maximum(-e, 0)).sum()))
                r["dswe_ea"] = np.nan if ex[0] is None else float((ex[0] * np.maximum(e, 0) + ex[1] * np.maximum(-e, 0)).sum())
                recs.append(r)
    X0 = pd.DataFrame(recs)
    X0.to_parquet(OUT / "bl_prices_panel.parquet")
    cols = ["cost", "mae", "rmse", "crps", "bias", "dswe", "dswe_pos", "dswe_ea"]

    def prep(X):
        X = X.copy()
        grp = ["cfg", "pc", "tie", "day"]
        for c in cols:
            X[c] = X[c] - X.groupby(grp)[c].transform("mean")
            X[c] = X[c] / X.groupby(["cfg", "pc", "tie"])[c].transform("std").replace(0, np.nan)
        return X
    for label, sub in (("全部（4 组 × 3 种选解）", X0), *[(f"{tie}", X0[X0.tie == tie]) for tie in TIES]):
        X = prep(sub).dropna(subset=["cost", "mae", "rmse", "crps", "dswe"])
        tcode = pd.factorize(X.day)[0]
        f0 = dk_fit(X, X.cost, ["mae", "rmse", "crps"], tcode)
        f1 = dk_fit(X, X.cost, ["mae", "rmse", "crps", "dswe"], tcode)
        fb = dk_fit(X, X.cost, ["mae", "rmse", "crps", "bias"], tcode)
        Xe = X.dropna(subset=["dswe_ea"])
        te = pd.factorize(Xe.day)[0]
        g0 = dk_fit(Xe, Xe.cost, ["mae", "rmse", "crps"], te)
        g1 = dk_fit(Xe, Xe.cost, ["mae", "rmse", "crps", "dswe_ea"], te)
        lines.append(f"[{label}] n={len(X)}：同日 R² 精度指标 {f0.rsquared:.3f} → +DSWE（事后）{f1.rsquared:.3f}（DSWE 标准化系数 {f1.params['dswe']:.2f}，p={f1.pvalues['dswe']:.1e}；"
                     f"MAE {f1.params['mae']:+.2f} p={f1.pvalues['mae']:.2f}，RMSE {f1.params['rmse']:+.2f} p={f1.pvalues['rmse']:.2f}，CRPS {f1.params['crps']:+.2f} p={f1.pvalues['crps']:.2f}）；"
                     f"+平均偏差 {fb.rsquared:.3f}；事前价格（n={len(Xe)}）{g0.rsquared:.3f} → {g1.rsquared:.3f}（系数 {g1.params['dswe_ea']:.2f}，p={g1.pvalues['dswe_ea']:.1e}）")
    for (c, pc), sub in X0.groupby(["cfg", "pc"]):
        X = prep(sub).dropna(subset=["cost", "mae", "rmse", "crps", "dswe"])
        import statsmodels.api as sm
        r0 = sm.OLS(X.cost, X[["mae", "rmse", "crps"]]).fit().rsquared
        r1 = sm.OLS(X.cost, X[["mae", "rmse", "crps", "dswe"]]).fit().rsquared
        lines.append(f"  {c} {pc:.0f} kW：{r0:.2f} → {r1:.2f}")
    # 3. DSWE 能否按成本给模型排序（整期合计）
    for (c, pc, tie), g in X0.groupby(["cfg", "pc", "tie"]):
        tot = g.groupby("model")[["cost", "mae", "dswe"]].sum()
        from scipy.stats import kendalltau
        lines.append(f"  整期排序 {c} {pc:.0f} kW {tie}：τ(MAE, 成本) {kendalltau(tot.mae, tot.cost).statistic:+.2f}，τ(DSWE, 成本) {kendalltau(tot.dswe, tot.cost).statistic:+.2f}")
    txt = "\n".join(lines)
    (OUT / "exp_bl_prices.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    Tk = []
    for d in pd.date_range("2024-03-04", cfg.MP_FC_END, freq="D"):
        y = H[S].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
        if np.isfinite(y).all() and d in EV:
            Tk.append((d, *EV[d], y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, ebar))
    if len(sys.argv) > 1:
        Tk = Tk[40:40 + int(sys.argv[1])]
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 7)), batch_size=1, verbose=5)(delayed(task)(*t) for t in Tk)
    P = pd.DataFrame([r for rr in res for r in rr], columns=["day", "cfg", "pc", "tie", "base", "wu", "wo"])
    P.to_pickle(OUT / ("bl_prices_test.pkl" if len(sys.argv) > 1 else "bl_prices.pkl"))
    if len(sys.argv) > 1:
        print(P.groupby(["cfg", "pc", "tie"]).apply(lambda g: pd.Series(dict(wu=np.mean(np.vstack(g.wu)), wo=np.mean(np.vstack(g.wo))))))
    else:
        analyse()
