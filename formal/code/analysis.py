"""预注册分析：主要假设 P1–P4 的确认性检验（Holm 校正）与次要假设、描述性结果的汇总表。用法：python analysis.py
统计：日期块同步重抽样（平稳自助法，块长按 Politis–White）；回归用 Driscoll–Kraay 标准误；实务界 5%。
输出：out/tables/*.csv 与 out/tables/summary.md。"""
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau
from arch.bootstrap import StationaryBootstrap, optimal_block_length
import cfg
import data as D
import sim
from util import log

T = cfg.OUT / "tables"
MAIN_POOL = ["naive", "naive7", "seasonal_q", "lqr", "lgb_local", "lgb_global", "chronos2_cov", "timesfm3_cov", "combo5"]
B = 2000
lines = []


def note(s):
    lines.append(s)
    log(s, "analysis")


def boot_mean(x, B=B, seed=cfg.SEED):
    """日度序列均值的平稳块自助法置信区间与全部重抽样均值。"""
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 10:
        return np.nan, (np.nan, np.nan), np.array([])
    bl = max(1.0, float(optimal_block_length(x)["stationary"].iloc[0]))
    bs = StationaryBootstrap(bl, x, seed=seed)
    means = np.array([d[0][0].mean() for d in bs.bootstrap(B)])
    return x.mean(), (np.quantile(means, 0.025), np.quantile(means, 0.975)), means


def holm(p):
    p = pd.Series(p).dropna()
    order = p.sort_values()
    adj, run = {}, 0.0
    for i, (k, v) in enumerate(order.items()):
        run = max(run, min(1.0, (len(p) - i) * v))
        adj[k] = run
    return pd.Series(adj)


# ---------------------------------------------------------------- E1 精度
def e1():
    rows = []
    for ds, start in (("mp", cfg.MP_EVAL_START), ("uev", cfg.UEV_EVAL_START)):
        for f in sorted((cfg.OUT / "fc").glob(f"{ds}_*.parquet")):
            F = pd.read_parquet(f)
            F = F[(F.t >= start)].dropna(subset=["y"])
            cal = f.stem.endswith("_cal")
            for s, g in F.groupby("series"):
                y, mu, Q = g.y.to_numpy(), g["mean"].to_numpy(), g[cfg.QC].to_numpy()
                rows.append(dict(ds=ds, series=s, model=g.model.iloc[0], cal=cal, MAE=np.abs(y - mu).mean(),
                                 RMSE=np.sqrt(((y - mu) ** 2).mean()), CRPS=sim.crps_q(y, Q).mean(),
                                 cov90=np.mean((y >= Q[:, 0]) & (y <= Q[:, -1]))))
    A = pd.DataFrame(rows)
    base = A[(A.model == "naive") & (~A.cal)].set_index(["ds", "series"])
    for c in ("MAE", "RMSE", "CRPS"):
        A[f"rel_{c}"] = A[c] / A.set_index(["ds", "series"]).index.map(base[c])
    A.to_csv(T / "e1_accuracy.csv", index=False)
    mp = A[(A.ds == "mp") & (~A.cal)].pivot(index="series", columns="model", values="rel_CRPS")
    note("## E1 精度（MP，4–12 月，CRPS 相对季节朴素）\n\n" + mp.round(3).to_markdown())
    u = A[(A.ds == "uev") & (~A.cal)].groupby("model").rel_CRPS.median()
    note("\nUrbanEV（11 月–次年 2 月，各区 CRPS 相对季节朴素的中位数）\n\n" + u.round(3).to_frame().to_markdown())
    cv = A[A.model.isin(MAIN_POOL)].pivot_table(index="model", columns="cal", values="cov90", aggfunc="median")
    note("\nS7 校准：90% 区间覆盖率中位数（False 原始 / True 校准后）\n\n" + cv.round(3).to_markdown())
    return A


# ---------------------------------------------------------------- P1
def p1():
    S = pd.read_parquet(cfg.OUT / "uev" / "uev_sched.parquet")
    Fz = pd.read_parquet(cfg.OUT / "uev" / "uev_feat.parquet")
    out = []
    for z, g in S[S.E == 0.25].groupby("zone"):
        f = Fz[Fz.zone == z]
        pm = f.pivot(index="day", columns="model", values="mae")
        pv = g.pivot(index="day", columns="model", values="regret")
        ms = [m for m in pm.columns if m in pv.columns]
        ag = n = 0
        for a, b in itertools.combinations(ms, 2):
            dm, dv = pm[a] - pm[b], pv[a] - pv[b]
            okk = (dm.abs() > 1e-9) & (dv.abs() > 1e-9)
            ag += int((np.sign(dm[okk]) == np.sign(dv[okk])).sum())
            n += int(okk.sum())
        fz = f.iloc[0]
        out.append(dict(zone=z, agree=ag / n if n else np.nan, cv_hour=fz.cv_hour, acf24=fz.acf24,
                        zero_share=fz.zero_share, log_kwh=np.log(max(fz.mean_kwh, 1e-6)),
                        conc=g.conc.mean() if "conc" in g else np.nan))
    Z = pd.DataFrame(out).dropna(subset=["agree"])
    Z.to_csv(T / "p1_zone_agreement.csv", index=False)
    X = Z[["cv_hour", "acf24", "zero_share", "log_kwh"]]
    X = sm.add_constant((X - X.mean()) / X.std())
    fit = sm.OLS(Z.agree, X, missing="drop").fit(cov_type="HC1")
    note("\n## P1 区层面精度—价值方向一致率（UrbanEV，储能 25%）\n\n"
         + f"一致率中位数 {Z.agree.median():.3f}（四分位 {Z.agree.quantile(.25):.3f}–{Z.agree.quantile(.75):.3f}），区数 {len(Z)}\n\n"
         + fit.summary2().tables[1].round(4).to_markdown())
    return float(fit.pvalues["cv_hour"]), float(fit.params["cv_hour"])


# ---------------------------------------------------------------- P2
def p2():
    M = pd.read_parquet(cfg.OUT / "mech" / "mech.parquet")
    M = M[M.source == cfg.MECH_SOURCE]
    P = M[(M.treat == "perm") & (M.level != "orig")]
    rows = []
    for mode in ("current", "open", "mpc"):
        col = f"reg_{mode}"
        g = P.dropna(subset=[col]).groupby(["station", "cfg", "level", "day"])[col]
        disp = (g.quantile(0.9) - g.quantile(0.1)) / g.mean().clip(lower=1e-9)
        d = disp.groupby(["station", "cfg", "day"]).mean().reset_index(name="disp")
        d["mode"] = mode
        rows.append(d)
    Dd = pd.concat(rows)
    Dd.to_csv(T / "p2_dispersion_daily.csv", index=False)
    summ = Dd.groupby(["station", "mode"]).disp.median().unstack()
    note("\n## P2 同一持续性档内的同日成本离散度（中位数）\n\n" + summ.round(3).to_markdown())
    a7 = Dd[(Dd.station == cfg.BSS) & (Dd["mode"] == "current")].groupby("day").disp.mean()
    m, ci, means = boot_mean(a7)
    p = float((means <= cfg.SESOI_REGRET).mean()) if len(means) else np.nan
    note(f"\nA7 现规则：平均离散度 {m:.3f}，95% CI [{ci[0]:.3f}, {ci[1]:.3f}]，相对 5% 实务界的单侧 p = {p:.4f}")
    lev = P.groupby(["station", "level", "day"])[["reg_current", "reg_open"]].mean().reset_index()
    ratio = lev.pivot_table(index=["station", "day"], columns="level", values="reg_current")
    rr = (ratio["high"] / ratio["low"]).groupby("station").median()
    note("\n持续性效应：高 / 低持续性的平均遗憾之比（现规则，日中位数）\n\n" + rr.round(3).to_frame("high/low").to_markdown())
    feats = []
    for st, x in P.groupby("station"):
        for mode, fd, fs in (("current", "defrec_cur", "dswe_cur"), ("open", "defrec_open", "dswe_open")):
            col = f"reg_{mode}"
            cs = {}
            for fe in (fd, fs, "dmax", "acf"):
                if fe not in x:
                    continue
                c = x.groupby(["cfg", "day"]).apply(
                    lambda z: z[fe].corr(z[col]) if z[fe].std() > 1e-9 and z[col].std() > 1e-9 else np.nan,
                    include_groups=False)
                cs[fe.replace("_cur", "").replace("_open", "")] = c.median()
            feats.append(dict(station=st, mode=mode, **cs))
    note("\n路径特征与同日成本的相关（中位数）\n\n" + pd.DataFrame(feats).round(3).to_markdown(index=False))
    C = pd.read_parquet(cfg.OUT / "mech" / "ctrl.parquet")
    note(f"\n正对照最大偏差 {np.abs(C[['pc_current', 'pc_open']]).to_numpy().max():.2e} 元；负对照最大偏差 {C.neg_ctrl.max():.2e}")
    return p, m


# ---------------------------------------------------------------- P3
def p3():
    Fe = pd.read_parquet(cfg.OUT / "dec" / "mp_feat.parquet")
    S = pd.read_parquet(cfg.OUT / "dec" / "mp_sched.parquet")
    S = S[(S.strategy == "point") & (S.tie == "mid") & (S["mode"] == "current")]
    X = Fe.merge(S[["station", "day", "cfg", "model", "regret"]], on=["station", "day", "cfg", "model"])
    X = X[X.model.isin(MAIN_POOL)].copy()
    cols = ["regret", "mae", "rmse", "crps", "dswe", "defrec", "rough"]
    grp = ["station", "cfg", "day"]
    for c in cols:
        X[c] = X[c] - X.groupby(grp)[c].transform("mean")
        X[c] = X[c] / X.groupby(["station", "cfg"])[c].transform("std").replace(0, np.nan)
    X = X.dropna(subset=cols)
    X["conc_c"] = X.conc - X.conc.mean()
    X["dswe_x_conc"] = X.dswe * X.conc_c
    tcode = pd.factorize(X.day)[0]
    res = {}
    for name, rhs in (("通用指标", ["mae", "rmse", "crps"]), ("+DSWE", ["mae", "rmse", "crps", "dswe"]),
                      ("+DSWE×集中度", ["mae", "rmse", "crps", "dswe", "dswe_x_conc"]),
                      ("+累计缺口", ["mae", "rmse", "crps", "defrec"]), ("+粗糙度", ["mae", "rmse", "crps", "rough"])):
        fit = sm.OLS(X.regret, X[rhs]).fit(cov_type="hac-groupsum", cov_kwds={"time": tcode, "maxlags": 7})
        res[name] = fit
    tab = pd.DataFrame({k: dict(R2=v.rsquared, **{f"b_{c}": v.params[c] for c in v.params.index},
                                **{f"p_{c}": v.pvalues[c] for c in v.params.index}) for k, v in res.items()}).T
    tab.to_csv(T / "p3_regressions.csv")
    note("\n## P3 同日跨模型的遗憾回归（标准化，Driscoll–Kraay 标准误）\n\n" + tab.round(4).to_markdown())
    by_st = {}
    for st, x in X.groupby("station"):
        r0 = sm.OLS(x.regret, x[["mae", "rmse", "crps"]]).fit().rsquared
        r1 = sm.OLS(x.regret, x[["mae", "rmse", "crps", "dswe"]]).fit().rsquared
        by_st[st] = (r0, r1)
    note("\n各站同日 R²（通用指标 → 加 DSWE）：" + "；".join(f"{k} {a:.2f}→{b:.2f}" for k, (a, b) in by_st.items()))
    return float(res["+DSWE"].pvalues["dswe"]), float(res["+DSWE"].params["dswe"])


# ---------------------------------------------------------------- P4 与杠杆
def p4():
    S = pd.read_parquet(cfg.OUT / "dec" / "mp_sched.parquet")
    a7 = S[(S.station == cfg.BSS) & (S.strategy == "point") & (S["mode"] == "current") & S.model.isin(MAIN_POOL)]
    pv = a7.pivot_table(index=["day", "cfg", "model"], columns="tie", values="regret").reset_index()
    daily = pv.groupby("day")[["default", "mid", "max"]].sum()
    red = 1 - daily["mid"] / daily["default"]
    m, ci, means = boot_mean(red)
    p = float((means <= cfg.SESOI_REGRET).mean()) if len(means) else np.nan
    up = pv.pivot_table(index=["day", "cfg"], columns="model", values="default").groupby("day")[["lgb_local", "combo5"]].sum()
    mg = 1 - up["combo5"] / up["lgb_local"]
    m2, ci2, _ = boot_mean(mg)
    m3, ci3, _ = boot_mean(red - mg.reindex(red.index))
    spread = pv.groupby("model")[["default", "mid"]].mean()
    note("\n## P4 换电中的计划选择（中点解 vs 求解器默认解）\n\n"
         + f"日度遗憾相对降低 {m:.3f}，95% CI [{ci[0]:.3f}, {ci[1]:.3f}]，相对 5% 实务界的单侧 p = {p:.4f}\n\n"
         + f"换模型（LightGBM → 5 成员组合，默认解）的相对降低 {m2:.3f}，CI [{ci2[0]:.3f}, {ci2[1]:.3f}]；"
         + f"计划选择减换模型 {m3:.3f}，CI [{ci3[0]:.3f}, {ci3[1]:.3f}]\n\n"
         + "各模型日均遗憾（默认解 / 中点解）\n\n" + spread.round(1).to_markdown()
         + f"\n\n模型间日均遗憾的极差：默认解 {np.ptp(spread['default']):.1f}，中点解 {np.ptp(spread['mid']):.1f}")
    # 杠杆比较（绝对成本）
    S2 = S[(S.strategy.isin(["point", "saa_qt"])) & (S["mode"] == "current")].copy()
    S2["cost"] = S2.regret + S2.opt
    rows = []
    for st, x in S2.groupby("station"):
        base = x[(x.model == "lgb_local") & (x.tie == "default") & (x.strategy == "point")].groupby("cfg").cost.mean()
        lev = {"换模型（→5 成员组合）": x[(x.model == "combo5") & (x.tie == "default") & (x.strategy == "point")].groupby("cfg").cost.mean(),
               "计划选择（→中点解）": x[(x.model == "lgb_local") & (x.tie == "mid") & (x.strategy == "point")].groupby("cfg").cost.mean(),
               "情景优化（分位数×等价解）": x[(x.model == "lgb_local") & (x.strategy == "saa_qt")].groupby("cfg").cost.mean()}
        for k, v in lev.items():
            common = base.index.intersection(v.index)
            if len(common):
                rows.append(dict(station=st, lever=k, rel_cost_change=float((v[common] / base[common] - 1).median())))
        bc = base.rename(lambda s: eval(s))
        steps = []
        for c in bc.index:
            nxt = (c[0] * 2, c[1], c[2]) if st == cfg.BSS else (c[0] * 2, c[1], c[2])
            if nxt in bc.index:
                steps.append(bc[nxt] / bc[c] - 1)
        if steps:
            rows.append(dict(station=st, lever="加资源（库容或储能容量翻倍）", rel_cost_change=float(np.median(steps))))
    L = pd.DataFrame(rows)
    L.to_csv(T / "levers.csv", index=False)
    note("\n杠杆比较（执行后绝对成本相对 LightGBM + 默认解的变化，资源组合中位数）\n\n"
         + L.pivot(index="station", columns="lever", values="rel_cost_change").round(4).to_markdown())
    return p, m


# ---------------------------------------------------------------- 次要假设与描述性结果
def secondary():
    Dc = pd.read_parquet(cfg.OUT / "dec" / "mp_decl.parquet")
    base = Dc[Dc.model == "naive"].groupby(["station", "strategy", "tau", "b"]).cost.sum()
    rel = Dc.groupby(["station", "model", "strategy", "tau", "b"]).cost.sum() / base
    rel = rel.reset_index(name="rel")
    rel.to_csv(T / "e2_declaration.csv", index=False)
    t = rel[rel.model.isin(MAIN_POOL)].groupby(["model", "strategy", "b"]).rel.median().unstack()
    note("\n## E2 / S2 申报：相对季节朴素的成本（站点 × τ 中位数）\n\n" + t.round(3).to_markdown())
    pp = Dc.pivot_table(index=["station", "model", "tau", "b"], columns="strategy", values="cost", aggfunc="sum")
    note("\n概率策略（校准后）/ 点策略（中位数，按 τ）\n\n" + (pp["prob_cal"] / pp["point"]).groupby("tau").median().round(3).to_frame().to_markdown())
    # S4 天气
    S = pd.read_parquet(cfg.OUT / "dec" / "mp_sched.parquet")
    pt = S[(S.strategy == "point") & (S.tie == "mid") & (S["mode"] == "current")]
    w = pt.pivot_table(index="station", columns="model", values="regret", aggfunc="mean")
    s4 = pd.DataFrame({"LGB 天气预报/LGB": w["lgb_wx"] / w["lgb_local"], "LGB 实际天气/LGB": w["lgb_wx_perfect"] / w["lgb_local"],
                       "Chronos 天气/Chronos": w["chronos2_wx"] / w["chronos2_cov"]})
    note("\n## S4 天气信息：调度遗憾比值\n\n" + s4.round(3).to_markdown())
    # S5 基础模型 vs LightGBM
    s5 = pd.DataFrame({"Chronos-2/LGB": w["chronos2_cov"] / w["lgb_local"], "TimesFM-3.0/LGB": w["timesfm3_cov"] / w["lgb_local"],
                       "组合/LGB": w["combo5"] / w["lgb_local"]})
    note("\n## S5 基础模型相对 LightGBM 的调度遗憾\n\n" + s5.round(3).to_markdown())
    # S8 执行方式
    s8 = S[(S.strategy == "point") & (S.tie == "mid")].pivot_table(index=["station", "model"], columns="mode", values="regret", aggfunc="mean")
    s8 = s8.dropna()
    note("\n## S8 执行方式（代表性组合，日均遗憾）\n\n" + s8.round(1).to_markdown())
    # MP 精度—价值一致性
    Fe = pd.read_parquet(cfg.OUT / "dec" / "mp_feat.parquet")
    X = Fe.merge(pt[["station", "day", "cfg", "model", "regret"]], on=["station", "day", "cfg", "model"])
    X = X[X.model.isin(MAIN_POOL)]
    kt = []
    for (st, c), g in X.groupby(["station", "cfg"]):
        a = g.groupby("model")[["mae", "regret"]].mean()
        kt.append(dict(station=st, cfg=c, tau=kendalltau(a.mae, a.regret).statistic, mismatch=a.mae.idxmin() != a.regret.idxmin()))
    K = pd.DataFrame(kt)
    note("\n## MP 精度—价值一致性（Kendall τ 中位数与 MAE 最优 ≠ 价值最优的比例）\n\n"
         + K.groupby("station").agg(tau=("tau", "median"), mismatch=("mismatch", "mean")).round(3).to_markdown())
    # S6 选模型
    sel = selection(X)
    note("\n## S6 选模型（7–12 月，相对固定 5 成员组合）\n\n" + sel.round(3).to_markdown())
    # E3 聚合
    A = pd.read_parquet(cfg.OUT / "agg" / "mp_agg.parquet")
    g = A.groupby(["model", "group", "tau", "b"])[["kwh", "agg_point", "agg_prob", "agg_prob_indep", "sep_point", "sep_prob"]].sum()
    e3 = pd.DataFrame({"单站：概率额外价值 元/MWh": 1000 * (g.sep_point - g.sep_prob) / g.kwh,
                       "聚合：概率额外价值 元/MWh": 1000 * (g.agg_point - g.agg_prob) / g.kwh,
                       "聚合忽略跨站相关": 1000 * (g.agg_point - g.agg_prob_indep) / g.kwh})
    e3.to_csv(T / "e3_aggregation.csv")
    note("\n## E3 聚合（5 成员组合，零容差）\n\n" + e3.xs(("combo5", 0.0), level=("model", "b")).round(2).to_markdown())
    U = pd.read_parquet(cfg.OUT / "agg" / "uev_agg.parquet")
    u = U.groupby(["tau", "size"])[["kwh", "point", "prob"]].sum()
    ucurve = (1000 * (u.point - u.prob) / u.kwh).unstack(0)
    ucurve.to_csv(T / "e3_uev_scale_curve.csv")
    note("\nUrbanEV 额外价值—聚合规模曲线（元/MWh）\n\n" + ucurve.round(2).to_markdown())


def monthly_contract():
    """8.1 月度分时段合约申报（描述性）：M−1 月最后一日用近 28 天各电价时段日均电量 × 当月天数申报；
    偏差超出 ±3% 的部分按电度电价 × 0.05 计偏差损失（超用系数 1.05、少用系数 0.95，沪经信运〔2025〕843 号）。"""
    H = D.mp_hourly()
    rows = []
    for s in cfg.POWER:
        y = H[s]
        for mo in pd.period_range("2024-04", "2024-12", freq="M"):
            start, end = mo.start_time, mo.end_time.floor("h")
            hist = y.loc[start - pd.Timedelta(days=28):start - pd.Timedelta(hours=1)]
            price = D.price(s, mo.month)
            for level in np.unique(price):
                hrs = np.where(price == level)[0]
                daily_hist = hist[hist.index.hour.isin(hrs)].sum() / 28
                decl = daily_hist * mo.days_in_month
                act = y.loc[start:end][y.loc[start:end].index.hour.isin(hrs)].sum()
                over = max(act - 1.03 * decl, 0)
                under = max(0.97 * decl - act, 0)
                rows.append(dict(station=s, month=str(mo), price=level, declared=decl, actual=act,
                                 dev=(act - decl) / decl if decl > 0 else np.nan, loss=0.05 * level * (over + under),
                                 energy_cost=level * act))
    R = pd.DataFrame(rows)
    R.to_csv(T / "s_monthly_contract.csv", index=False)
    g = R.groupby("station").agg(mean_abs_dev=("dev", lambda x: x.abs().mean()), loss=("loss", "sum"), cost=("energy_cost", "sum"))
    g["loss_share"] = g.loss / g.cost
    note("\n## 8.1 月度分时段合约申报（描述性，4–12 月）\n\n" + g.round(4).to_markdown())


def selection(X):
    S = X.pivot_table(index=["station", "cfg", "day"], columns="model", values="regret")
    met = {c: X.pivot_table(index=["station", "cfg", "day"], columns="model", values=c) for c in ("mae", "crps", "dswe")}
    met["value"] = S
    res = []
    for (st, c), Sg in S.groupby(level=[0, 1]):
        Sg = Sg.droplevel([0, 1])
        days = Sg.index
        ev = days[days >= pd.Timestamp("2024-07-01")]
        for win in (14, 30, 90):
            for crit in ("mae", "crps", "dswe", "value"):
                Mg = met[crit].loc[(st, c)]
                vals = []
                for mo, evm in pd.Series(ev, index=ev).groupby(ev.to_period("M")):
                    past = Mg[(Mg.index < evm.iloc[0]) & (Mg.index >= evm.iloc[0] - pd.Timedelta(days=win))]
                    pick = past.mean().idxmin()
                    vals.append(Sg.loc[evm.values, pick].mean())
                res.append(dict(station=st, cfg=c, strategy=f"按{crit}·{win}天", regret=np.mean(vals)))
        fixed = Sg[Sg.index < pd.Timestamp("2024-07-01")].mean().idxmin()
        res.append(dict(station=st, cfg=c, strategy="固定（4–6 月最佳）", regret=Sg.loc[ev, fixed].mean()))
        res.append(dict(station=st, cfg=c, strategy="固定组合", regret=Sg.loc[ev, "combo5"].mean()))
        best = [Sg.loc[evm.values].mean().min() for _, evm in pd.Series(ev, index=ev).groupby(ev.to_period("M"))]
        res.append(dict(station=st, cfg=c, strategy="事后逐月最佳", regret=float(np.mean(best))))
    R = pd.DataFrame(res)
    R.to_csv(T / "s6_selection.csv", index=False)
    base = R[R.strategy == "固定组合"].set_index(["station", "cfg"]).regret
    R["rel"] = R.regret / R.set_index(["station", "cfg"]).index.map(base)
    return R.groupby(["strategy", "station"]).rel.median().unstack()


def main():
    note(f"# 正式实验结果摘要（{pd.Timestamp.now():%Y-%m-%d %H:%M}）\n")
    e1()
    pvals, effects = {}, {}
    for name, fn in (("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4)):
        try:
            pvals[name], effects[name] = fn()
        except Exception as ex:
            note(f"\n{name} 失败：{type(ex).__name__}: {ex}")
    adj = holm(pvals)
    note("\n## 主要假设汇总（Holm 校正）\n\n" + pd.DataFrame({"效应": effects, "p": pvals, "Holm 调整 p": adj}).round(4).to_markdown())
    for fn in (secondary, monthly_contract):
        try:
            fn()
        except Exception as ex:
            note(f"\n{fn.__name__} 失败：{type(ex).__name__}: {ex}")
    (T / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log("分析完成：out/tables/summary.md", "analysis")


if __name__ == "__main__":
    main()
