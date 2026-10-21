"""探索性方法实验（非预注册，规则在运行前固定）：在最优面上按缓冲比例 α 选计划，并用概率信息选 α。
最优面是凸集，最小库存解 x_min 与最大库存解 x_max 的凸组合 x_α = (1 − α) x_min + α x_max 都是同一点预测下的最优计划，
计划成本完全相同；α ∈ {0, 0.25, 0.5, 0.75, 1}（0.5 即预注册的中点解）。两种事前可执行的选 α 规则：
  情景规则：用目标日 D−2 及以前 ≤28 天的整日残差向量构造情景（与情景优化相同），按现规则模拟执行，选平均成本最低的 α；
  滚动规则：按 (d−30, d−2] 内各 α 的实际遗憾均值选最低者（与 S6 的事后可得信息一致）。
范围：A7（27 组合 × 9 模型）、A8（27 × 9，4–10 月）、7 个常规储能站（代表性 9 组合 × 3 模型）、UrbanEV 换电（3 组合 × 5 模型）。
共同评估窗口：MP 5 月起（A8 至 10 月），UrbanEV 12 月起（保证两种规则都有历史）。用法：python buffer_alpha.py [A7 A8 ST UEV]"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "formal" / "code"))
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
import sim
from dec_mp import scales
from analysis import MAIN_POOL, boot_mean

OUT = (pathlib.Path(__file__).resolve().parents[2] / "results" / "profit_basis" / "tables_extra")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def alpha_plans(P, f):
    f = np.maximum(np.asarray(f, float), 0)
    if P.kind == "storage":
        lo = sim.st_plan(f, P.price, P.E, P.L, P.lam, "min")
        hi = sim.st_plan(f, P.price, P.E, P.L, P.lam, "max")
        return {a: ("st", (1 - a) * lo[0] + a * hi[0], (1 - a) * lo[1] + a * hi[1], f) for a in ALPHAS}
    lo = sim.bss_plan(f, P.price, P.pen, P.cap, P.pmax, "min")
    hi = sim.bss_plan(f, P.price, P.pen, P.cap, P.pmax, "max")
    return {a: ("bss", (1 - a) * lo[0] + a * hi[0], (1 - a) * lo[1] + a * hi[1], f, (1 - a) * lo[3] + a * hi[3]) for a in ALPHAS}


def task(unit, kind, d, y, price, sc, grid, fcs, hist):
    rows = []
    for par in grid:
        P = sim.Problem(kind, y, price, par, sc)
        for m, f in fcs.items():
            pl = alpha_plans(P, f)
            H = hist.get(m)
            Ys = np.maximum(np.maximum(f, 0)[None, :] + H, 0) if H is not None and len(H) >= 7 else None
            for a, p in pl.items():
                rows.append((unit, d, str(par), m, a, P.cost(p) - P.opt,
                             float(np.mean([P.cost(p, ys) for ys in Ys])) if Ys is not None else np.nan))
    return rows


def residual_hist(series_mean, series_y, days, start):
    """每个目标日 d ≥ start：(d−2−28, d−2] 内完整日的残差矩阵（start 之前只用于滚动规则的历史，不做情景模拟）。"""
    R = (series_y - series_mean).sort_index()
    t0 = R.index.min().normalize()
    full = R.reindex(pd.date_range(t0, R.index.max().normalize() + pd.Timedelta(hours=23), freq="h"))
    A = full.to_numpy().reshape(-1, 24)
    ok = np.isfinite(A).all(1)
    out = {}
    for d in days:
        if d < start:
            out[d] = None
            continue
        i = (d - t0).days
        idx = [j for j in range(max(0, i - 2 - cfg.SCEN_WINDOW + 1), i - 1) if j < len(A) and ok[j]]
        out[d] = A[idx] if idx else None
    return out


def tasks_mp(tag):
    H = D.mp_hourly()
    SC = scales(H)
    if tag in ("A7", "A8"):
        st = cfg.BSS if tag == "A7" else "A8"
        stations, models, kind = [st], MAIN_POOL, "bss"
        end = cfg.MP_FC_END if tag == "A7" else pd.Timestamp("2024-10-31")
    else:
        stations, models, kind, end = cfg.POWER, ["lgb_local", "chronos2_cov", "combo5"], "storage", cfg.MP_FC_END
    days = D.days(cfg.MP_EVAL_START, end)
    F = {m: pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "mean"]) for m in models}
    T = []
    for st in stations:
        y_all = H[st]
        G = {m: F[m][F[m].series == st].set_index("t")["mean"] for m in models}
        HI = {m: residual_hist(G[m], y_all.reindex(G[m].index), days, pd.Timestamp("2024-05-01")) for m in models}
        pst = cfg.PRICE_PROXY.get(st, st)
        if kind == "bss":
            hmean = float(y_all.loc[:cfg.MP_DEV_END].mean())
        for d in days:
            idx = pd.date_range(d, periods=24, freq="h")
            y = y_all.reindex(idx).to_numpy()
            fcs = {m: G[m].reindex(idx).to_numpy() for m in models}
            if not (np.isfinite(y).all() and all(np.isfinite(v).all() for v in fcs.values())):
                continue
            if kind == "bss":
                sc = dict(hmean=hmean, tot_max=HK(D.price(pst, d.month, "total")))
                grid = cfg.BSS_GRID
            else:
                sc, grid = SC[st], cfg.ST_REP
            T.append((st, kind, d, y, D.price(pst, d.month), sc, grid, fcs, {m: HI[m][d] for m in models}))
    return T


def tasks_uev():
    V = D.uev_volume()
    models = ["naive", "lgb_global", "chronos2_cov", "timesfm3_cov", "combo5"]
    F = {m: pd.read_parquet(cfg.OUT / "fc" / f"uev_{m}.parquet", columns=["series", "t", "mean"]).pivot(index="t", columns="series", values="mean")
         for m in models}
    days = D.days(cfg.UEV_EVAL_START, cfg.UEV_FC_END)
    grid = [(c, 1.3, 3) for c in (2, 4, 8)]
    T = []
    for z in V.columns:
        hmean = float(V[z].loc[:cfg.UEV_DEV_END].mean())
        if hmean <= 0:
            continue
        HI = {m: residual_hist(F[m][z], V[z].reindex(F[m].index), days, pd.Timestamp("2022-12-01")) for m in models}
        for d in days:
            idx = pd.date_range(d, periods=24, freq="h")
            y = V[z].reindex(idx).to_numpy()
            fcs = {m: F[m][z].reindex(idx).to_numpy() for m in models}
            if not (np.isfinite(y).all() and all(np.isfinite(v).all() for v in fcs.values())):
                continue
            mo = cfg.UEV_PRICE_MONTH[d.month]
            T.append((z, "bss", d, y, D.price("A1", mo), dict(hmean=hmean, tot_max=HK(D.price("A1", mo, "total"))),
                      grid, fcs, {m: (np.nan_to_num(HI[m][d]) if HI[m][d] is not None else None) for m in models}))
    return T


def summarize(R, tag, start):
    key = ["unit", "cfg", "model"]
    R = R.sort_values("day")
    W = R.pivot_table(index=key + ["day"], columns="alpha", values="regret")
    S = R.pivot_table(index=key + ["day"], columns="alpha", values="scen")
    pick_s = S.idxmin(axis=1)
    reg_s = pd.Series([W.loc[i, a] for i, a in zip(pick_s.index, pick_s.values)], index=pick_s.index)
    # 滚动规则：(d−30, d−2] 的实际遗憾均值
    reg_r, pick_r = {}, {}
    for k, g in W.groupby(level=[0, 1, 2]):
        g = g.droplevel([0, 1, 2])
        for d in g.index:
            past = g[(g.index <= d - pd.Timedelta(days=2)) & (g.index > d - pd.Timedelta(days=30))]
            if len(past) >= 7:
                a = past.mean().idxmin()
                reg_r[k + (d,)], pick_r[k + (d,)] = g.loc[d, a], a
    E = W.copy()
    E["scen"], E["roll"] = reg_s, pd.Series(reg_r)
    E["pick_scen"], E["pick_roll"] = pick_s, pd.Series(pick_r)
    E = E.reset_index()
    E = E[(E.day >= start)].dropna(subset=["scen", "roll"])
    daily = E.groupby("day")[ALPHAS + ["scen", "roll"]].sum()
    lines = [f"\n## {tag}（{E.unit.nunique()} 个单元，{daily.shape[0]} 天，自 {start.date()}）\n",
             "各规则的总遗憾（相对 α = 0）：" + "，".join(f"{c}: {daily[c].sum() / daily[0.0].sum():.3f}" for c in ALPHAS + ["scen", "roll"])]
    for c in ("scen", "roll", 1.0):
        for base in (0.0, 0.5):
            m, ci, _ = boot_mean(1 - daily[c] / daily[base])
            lines.append(f"- {c} 相对 α={base} 的日度遗憾降低：{m:.3f}，95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    lines.append("- 情景规则选中的 α 分布：" + E.pick_scen.value_counts(normalize=True).sort_index().round(3).to_dict().__repr__())
    lines.append("- 滚动规则选中的 α 分布：" + E.pick_roll.value_counts(normalize=True).sort_index().round(3).to_dict().__repr__())
    E["mult"] = E.cfg.map(lambda s: eval(s)[2])
    g = E.groupby("mult")[ALPHAS + ["scen", "roll"]].sum()
    lines.append("\n按第三个资源参数（换电为罚价倍数，储能为 λ）的总遗憾 / α=0：\n\n" + g.div(g[0.0], axis=0).round(3).to_markdown())
    return "\n".join(lines)


if __name__ == "__main__":
    for tag in (sys.argv[1:] or ["A7", "A8", "ST", "UEV"]):
        fp = OUT / f"buffer_alpha_{tag}.parquet"
        if not fp.exists():
            T = tasks_uev() if tag == "UEV" else tasks_mp(tag)
            print(tag, len(T), "tasks", flush=True)
            res = Parallel(n_jobs=-1, batch_size=4, verbose=2)(delayed(task)(*t) for t in T)
            pd.DataFrame([r for rr in res for r in rr], columns=["unit", "day", "cfg", "model", "alpha", "regret", "scen"]).to_parquet(fp)
        R = pd.read_parquet(fp)
        start = pd.Timestamp("2022-12-01") if tag == "UEV" else pd.Timestamp("2024-05-01")
        txt = summarize(R, tag, start)
        (OUT / f"buffer_alpha_{tag}.md").write_text(txt, encoding="utf-8")
        print(txt, flush=True)
