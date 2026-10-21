"""第八稿的图（英文标注，利润口径）：fig1_example、fig2_accuracy_value、fig3_prices、fig4_dswe、fig5_plan、fig6_value、fig7_when。
数据：结果\（小时模型、储能站、误差价格、假设检验的利润口径输出）与 结果\利润口径\（电池级仿真等脚本副本的输出）。
写到 results/figures/。用法：python fig_v8.py [concept accuracy prices dswe plan value when]"""
import itertools
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(HERE.parent))
import cfg  # noqa: E402
import data as D  # noqa: E402
import sim  # noqa: E402
from analysis import boot_mean  # noqa: E402
from exp_packs_value import rel_ci  # noqa: E402
from exp_cost_basis import kappa_profit, exec_split, mix  # noqa: E402

OUT = ROOT / "results"
PR = OUT / "profit_basis"
FIG = ROOT / "results" / "figures"
TAB = cfg.OUT / "tables"
S7 = cfg.BSS
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
MODEL = {"naive": "Seasonal naive", "naive7": "Weekly naive", "seasonal_q": "Seasonal quantile", "lqr": "Linear QR",
         "lgb_local": "LightGBM (local)", "lgb_global": "LightGBM (global)", "chronos2_cov": "Chronos-2",
         "timesfm3_cov": "TimesFM-3.0", "combo5": "5-member combination"}
MAIN = list(MODEL)
COL = dict(zip(MAIN, plt.cm.tab10(np.arange(len(MAIN)))))
C_RULE, C_PROF, C_LGB = "k", "C1", "C0"
TUNE0 = pd.Timestamp("2024-05-01")


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def profile():
    W = pd.read_pickle(OUT / "rev_prices_profit.pkl")
    W = W[W.day >= cfg.MP_EVAL_START]
    prof = {k: np.mean(np.vstack(W[k].to_list()), axis=0) for k in ("wu", "wo", "wu0", "wo0")}
    tou = np.mean([D.price(S7, m) for m in range(4, 13)], axis=0)
    return prof, tou


def pick(M, d, fb):
    past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
    return past.sum().idxmin() if len(past) >= 7 else fb


# ---------------------------------------------------------------- 图 1
def concept():
    prof, _ = profile()
    pz, pm = prof["wu0"], prof["wu"]
    rng = np.random.default_rng(3)
    e = np.clip(rng.normal(0, 18, 24), -40, 40)
    e[[20, 21]] = [32, 26]
    fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(7.6, 2.35), gridspec_kw={"wspace": 0.42})
    hrs = np.arange(24)
    b1.bar(hrs, e, color=np.where(e > 0, "C3", "C0"), width=0.8)
    b1.axhline(0, color="0.3", lw=0.6)
    b1.set(title="(a) One day's errors (illustrative)", xlabel="Hour of day", ylabel="kWh (+ under-forecast)", xlim=(-0.5, 23.5), xticks=range(0, 24, 6))
    b2.plot(hrs, pz, color="C3", lw=1.5, label="Zero-buffer plan")
    b2.plot(hrs, pm, color="C0", lw=1.5, label="Midpoint plan")
    b2.set(title="(b) Under-forecast prices, A7", xlabel="Hour of day", ylabel="CNY per kWh", xlim=(0, 23), xticks=range(0, 24, 6))
    b2.legend(frameon=False, fontsize=6.3, loc="upper left")
    bill = [float(np.sum(np.clip(e, 0, None) * p)) for p in (pz, pm)]
    b3.bar([0, 1], bill, color=["C3", "C0"], width=0.6)
    b3.set_xticks([0, 1], ["Zero-buffer\nplan", "Midpoint\nplan"], fontsize=6.8)
    b3.set(title="(c) Same errors, two bills", ylabel="CNY (under-forecasts × prices)")
    for i, v in enumerate(bill):
        b3.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
    print("bills:", [round(v, 1) for v in bill], "evening default prices 20-23:", np.round(pz[20:24], 2), "mid:", np.round(pm[20:24], 2))
    save(fig, "fig1_example")


# ---------------------------------------------------------------- 图 3（精度与成本）
def zone_agreement():
    S = pd.read_parquet(OUT / "h1_profit.parquet")
    Fz = pd.read_parquet(cfg.OUT / "uev" / "uev_feat.parquet")
    out = []
    for z, g in S.groupby("zone"):
        f = Fz[Fz.zone == z]
        pm, pv = f.pivot(index="day", columns="model", values="mae"), g.pivot(index="day", columns="model", values="regret")
        ms = [m for m in pm.columns if m in pv.columns]
        ag = n = 0
        for a, b in itertools.combinations(ms, 2):
            dm, dv = pm[a] - pm[b], pv[a] - pv[b]
            ok = (dm.abs() > 1e-9) & (dv.abs() > 1e-9)
            ag += int((np.sign(dm[ok]) == np.sign(dv[ok])).sum())
            n += int(ok.sum())
        out.append(dict(zone=z, agree=ag / n if n else np.nan, cv_hour=f.iloc[0].cv_hour))
    return pd.DataFrame(out).dropna()


def accuracy():
    A = pd.read_csv(TAB / "e1_accuracy.csv")
    A = A[(A.ds == "mp") & (~A.cal) & A.model.isin(MAIN)]
    E = pd.read_csv(TAB / "e2_declaration.csv")
    E = E[(E.strategy == "prob_cal") & (E.b == 0.0) & (E.tau == 0.83) & E.model.isin(MAIN)]
    ST = pd.read_parquet(OUT / "cb_storage_sched.parquet")
    ST = ST[ST.tie == "mid"]
    reg = (ST.E + ST.Pen - ST.cost_opt).groupby([ST.station, ST.model]).mean()
    P = pd.read_parquet(OUT / "cb_hourly_plans.parquet", filters=[("inp", "==", "mean"), ("tie", "==", "mid")])
    P = P[P.model.isin(MAIN)]
    a7 = (P.E + P.Pen - P.E_opt - P.Pen_opt).groupby(P.model).mean()
    reg = pd.concat([reg, pd.Series(a7.to_numpy(), index=pd.MultiIndex.from_product([[S7], a7.index], names=["station", "model"]))])
    reg.index.names = ["station", "model"]
    reg = (reg / reg.xs("naive", level="model").reindex(reg.index.get_level_values(0)).to_numpy()).rename("rel_regret").reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    x = A.merge(E, left_on=["series", "model"], right_on=["station", "model"])
    for m, g in x.groupby("model"):
        ax[0].scatter(g.rel_CRPS, g.rel, s=10, color=COL[m], label=MODEL[m])
    r = x.groupby("station").apply(lambda g: spearmanr(g.rel_CRPS, g.rel).statistic, include_groups=False).median()
    ax[0].set(xlabel="CRPS relative to seasonal naive", ylabel="Declaration cost rel. to naive", title=f"(a) Day-ahead declaration, τ = 0.83\nwithin-station median ρ = {r:.2f}")
    y = A.merge(reg, left_on=["series", "model"], right_on=["station", "model"])
    for m, g in y.groupby("model"):
        ax[1].scatter(g.rel_MAE, g.rel_regret, s=10, color=COL[m])
    r = y.groupby("station").apply(lambda g: spearmanr(g.rel_MAE, g.rel_regret).statistic, include_groups=False).median()
    ax[1].set(xlabel="MAE relative to seasonal naive", ylabel="Scheduling regret rel. to naive", title=f"(b) Storage and swap scheduling\nwithin-station median ρ = {r:.2f}")
    Z = zone_agreement()
    ax[2].scatter(Z.cv_hour, Z.agree, s=5, color="0.4", alpha=0.6)
    b = np.polyfit(Z.cv_hour, Z.agree, 1)
    xx = np.linspace(Z.cv_hour.min(), Z.cv_hour.max(), 50)
    ax[2].plot(xx, np.polyval(b, xx), color="C3")
    ax[2].set(xlabel="Hourly coefficient of variation (dev. period)", ylabel="Accuracy–value agreement rate", title="(c) UrbanEV zones (n = 275)\nagreement vs irregularity")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    print("fig2 (b) median rho:", round(r, 2))
    save(fig, "fig2_accuracy_value")


# ---------------------------------------------------------------- 图 4（误差价格）
def disp_h2(station, grid):
    M = pd.read_parquet(OUT / "h2_profit.parquet")
    M = M[(M.station == station) & (M.grid == grid)]
    g = M.groupby(["cfg", "level", "day"]).regret
    d = ((g.quantile(0.9) - g.quantile(0.1)) / g.mean().clip(lower=1e-9)).groupby("day").mean()
    m, ci, _ = boot_mean(d)
    return m, ci[0], ci[1]


def prices():
    prof, tou = profile()
    P = pd.read_pickle(PR / "bl_prices.pkl")
    P = P[pd.to_datetime(P.day) >= cfg.MP_EVAL_START]
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.8), gridspec_kw={"wspace": 0.75, "width_ratios": [1, 1, 0.9]})
    hrs = np.arange(24)
    ax[0].step(hrs, tou, where="mid", color="0.55", lw=0.9, ls=":", label="Electricity price")
    ax[0].plot(hrs, prof["wu0"], color="C3", lw=1.5, label=f"Solver's plan (zero buffer), mean {prof['wu0'].mean():.2f}")
    ax[0].plot(hrs, prof["wu"], color="C0", lw=1.5, label=f"Midpoint plan, mean {prof['wu'].mean():.2f}")
    ax[0].set(xlabel="Hour of day", ylabel="CNY per kWh of under-forecast", title="(a) Hourly model", xlim=(0, 23), xticks=range(0, 24, 6))
    ax[0].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    for pc, ls in ((12.0, "-"), (25.0, "--")):
        for tie, col, lab in (("default", "C3", "solver's plan"), ("max", "C2", "maximum buffer")):
            g = P[(P.cfg == "(6, 2.5, 3)") & (P.pc == pc) & (P.tie == tie)]
            w = np.mean(np.vstack(g.wu.to_list()), axis=0)
            ax[1].plot(hrs, w, color=col, ls=ls, lw=1.3, label=f"{pc:.0f} kW, {lab}: {w.mean():.2f}")
    ax[1].step(hrs, tou, where="mid", color="0.55", lw=0.9, ls=":")
    ax[1].axhline(0, color="0.3", lw=0.5)
    ax[1].set(xlabel="Hour of day", ylabel="CNY per kWh of under-forecast", title="(b) Battery simulation", xlim=(0, 23), xticks=range(0, 24, 6))
    ax[1].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, title="Plan: mean price", title_fontsize=5.8)
    rows = [("A1 storage, hourly", *disp_h2("A1", "storage"), "0.6"), ("A3 storage, hourly", *disp_h2("A3", "storage"), "0.6"),
            ("A7 swap, hourly", *disp_h2(S7, "anchor"), "C0")]
    e4 = PR / "e4_dispersion.csv"
    if e4.exists():
        for _, r in pd.read_csv(e4).iterrows():
            rows.append((r.label, r.m, r.lo, r.hi, "C3"))
    lab, val, lo, hi, col = zip(*rows)
    val, lo, hi = map(np.array, (val, lo, hi))
    yy = np.arange(len(val))[::-1]
    ax[2].barh(yy, val, color=col, height=0.62)
    ax[2].errorbar(val, yy, xerr=[val - lo, hi - val], fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax[2].set_yticks(yy, lab, fontsize=6.2)
    ax[2].set(xlabel="P90 − P10 of same-day cost\n÷ mean cost of the errors", title="(c) Same errors, different bills")
    print("fig3 (c):", [(l, round(v, 3)) for l, v in zip(lab, val)])
    save(fig, "fig3_prices")


# ---------------------------------------------------------------- 图 5（DSWE）
def dswe():
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 2.8), gridspec_kw={"wspace": 0.3, "width_ratios": [0.7, 1.6]})
    # 数值取自 结果\h3_profit.txt（储能站 + A7 数据锚定网格合并）、结果\rev_dswe_profit.txt（A7）、结果\rev_dswe_st_profit.txt（各储能站）
    lab3 = (("MAE + RMSE + CRPS", "0.6"), ("+ errors at ex-post prices", "C1"), ("+ errors at ex-ante prices", "C0"))
    groups = ["Pooled\n(7 storage + A7)", "A7 swap"]
    # 第四根柱：加入三种不随计划变化的成本加权误差（电价、零售价/电价、毛利），见 结果\naive_weights_profit.txt
    lab4 = (("MAE + RMSE + CRPS", "0.6"), ("+ tariff-weighted errors\n(three fixed weightings)", "C2"), ("+ errors at ex-post prices", "C1"), ("+ errors at ex-ante prices", "C0"))
    vals = [(0.221, 0.385), (0.304, 0.492), (0.515, 0.570), (0.560, 0.561)]
    x = np.arange(2)
    for j, (v, (lab, c)) in enumerate(zip(vals, lab4)):
        ax[0].bar(x + (j - 1.5) * 0.2, v, 0.2, label=lab, color=c)
    ax[0].legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax[0].set_xticks(x, groups, fontsize=6.5)
    ax[0].set(ylabel="Within-day R² of cross-model\ncost differences", title="(a) Pooled panel", ylim=(0, 1.05))
    st = ["A1\ntaxi", "A2\ntaxi", "A3\nbus", "A4\nbus", "A5\nresid.", "A6\nresid.", "A9\ntruck", "A7\nswap"]
    acc = [0.356, 0.326, 0.108, 0.071, 0.218, 0.219, 0.485, 0.385]
    post = [0.561, 0.700, 0.978, 0.985, 0.224, 0.321, 0.543, 0.570]
    ante = [0.612, 0.707, 0.928, 0.836, 0.281, 0.397, 0.506, 0.561]
    x = np.arange(len(st))
    for j, (v, (lab, c)) in enumerate(zip((acc, post, ante), lab3)):
        ax[1].bar(x + (j - 1) * 0.27, v, 0.27, label=lab, color=c)
    ax[1].set_xticks(x, st, fontsize=6.3)
    ax[1].set(title="(b) By station", ylim=(0, 1.05))
    ax[1].legend(frameon=False, fontsize=6, loc="upper right", ncol=1)
    save(fig, "fig4_dswe")

# ---------------------------------------------------------------- 图 6（等成本计划）
def tradeoff(ax, c="(6, 2.5)"):
    TP = pd.read_parquet(OUT / "cb_tradeoff_points.parquet")
    g = TP[(TP.cfg == c) & (TP.model == "lgb_local")]
    order = ["mean", "q50", "q60", "q70", "q80", "q90", "q95"]
    for tie, col, mk, lab in (("default", "C3", "o", "Solver's plan"), ("mid", "C0", "s", "Midpoint plan"), ("max", "C2", "^", "Maximum buffer")):
        h = g[g.tie == tie].set_index("inp").reindex(order).dropna(subset=["share"])
        h = h.sort_values("share")
        ax.plot(100 * h.share, 100 * h.exE, mk + "-", color=col, ms=3, lw=1, label=lab)
    RU = pd.read_parquet(OUT / "cb_hourly_rule.parquet")
    RU = RU[(RU.cfg == c) & (RU.day >= cfg.MP_EVAL_START)]
    base = RU.drop_duplicates("day")
    r = RU.groupby("th").agg(E=("E", "sum"), S=("S", "sum"))
    sh, ex = 100 * r.S / base.dem.sum(), 100 * (r.E / base.E_opt.sum() - 1)
    o = np.argsort(sh.to_numpy())
    env = np.minimum.accumulate(ex.to_numpy()[o])                     # 缺供不超过横轴值的规则中电费最低者
    ax.step(sh.to_numpy()[o], env, where="post", color="k", lw=1, ls=":", label="Forecast-free rule (frontier)")
    print("fig5 (d) rule frontier at 1% service:", round(float(env[sh.to_numpy()[o] <= 1.0].min()), 1))
    ax.axvline(1.0, color="0.5", lw=0.7, ls="--")
    ax.text(1.05, 0.95, "1% service\ntarget", transform=ax.get_xaxis_transform(), fontsize=5.8, va="top", color="0.4")
    ax.set(xlim=(0, 6), ylim=(-3, 25), xlabel="Unserved demand (% of demand)", ylabel="Electricity cost above perfect\ninformation (%)",
           title="(d) Cost–service trade-off (hourly,\nLightGBM, 6 h stock, 2.5× power)")
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)


def plan():
    H = D.mp_hourly()
    hmean = float(H[S7].loc[:cfg.MP_DEV_END].mean())
    day = pd.Timestamp("2024-06-15")
    idx = pd.date_range(day, periods=24, freq="h")
    y = H[S7].reindex(idx).to_numpy()
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    f = np.maximum(F[F.series == S7].set_index("t")["mean"].reindex(idx).to_numpy(), 0)
    price, pen = D.price(S7, day.month), kappa_profit(day.month)
    cap, pmax = 6 * hmean, 2.5 * hmean
    opt = sim.bss_plan(y, price, pen, cap, pmax, None)[2]
    fig, ax = plt.subplots(1, 4, figsize=(12.6, 3.0), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.25, 1, 1.05]})
    for lv in np.unique(price):
        for h in np.where(price == lv)[0]:
            ax[0].axvspan(h, h + 1, color=plt.cm.Greys(0.12 + 0.5 * (lv - price.min()) / (price.max() - price.min())), lw=0)
    pl = {}
    for tie in ("min", "max"):
        g, s, _, u = sim.bss_plan(f, price, pen, cap, pmax, tie)
        pl[tie] = ("bss", g, s, f, u)
    pl["mid"] = mix(pl["min"], pl["max"], 0.5)
    regs = {}
    for tie, lab, c in (("min", "Minimum buffer (solver's plan)", "C3"), ("mid", "Midpoint", "C0"), ("max", "Maximum buffer", "C2")):
        e, sh, pn = exec_split(pl[tie], y, price, pen, cap, pmax)
        regs[tie] = e + pn - opt
        ax[0].plot(np.arange(25), pl[tie][2] / cap, color=c, label=f"{lab}: regret {regs[tie]:,.0f} CNY")
    print("example day regrets:", {k: round(v, 0) for k, v in regs.items()})
    ax[0].set(xlim=(0, 24), ylim=(0, 1.05), xticks=range(0, 25, 6), xlabel="Hour (shading: tariff level)", ylabel="Planned stock / capacity",
              title="(a) Three plans, same planned cost\n(hourly model, 15 June 2024)")
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.27))
    fs = PR / "exp_solvers.parquet"
    R = pd.read_parquet(fs if fs.exists() else OUT / "exp_solvers.parquet")
    forms = ["F1 小时库存 LP", "F1r 同上，变量倒序", "F1e 同上，消去库存变量", "F2 电池级 LP"]
    flab = ["Hourly LP", "Hourly LP, columns reversed", "Hourly LP, stock eliminated", "Battery-level LP"]
    solvers = ["HiGHS 对偶单纯形（SciPy，论文所用）", "HiGHS 不预处理的对偶单纯形", "HiGHS 原始单纯形", "HiGHS 内点 + 交叉", "HiGHS 内点，不交叉",
               "GLOP（OR-Tools）", "CLP（OR-Tools）", "CBC（PuLP）", "PDLP（一阶法）"]
    slab = ["HiGHS dual simplex (ours)", "HiGHS dual, no presolve", "HiGHS primal simplex", "HiGHS IPM + crossover", "HiGHS IPM",
            "GLOP", "CLP", "CBC", "PDLP"]
    med = R.dropna(subset=["pos"]).groupby(["solver", "form"]).pos.median().unstack("form").reindex(index=solvers, columns=forms)
    for j, fm in enumerate(forms):
        ax[1].scatter(med[fm], np.arange(len(solvers)) + (j - 1.5) * 0.15, marker=["o", "s", "^", "D"][j], s=16, label=flab[j],
                      color=["C3", "C1", "C4", "C2"][j], zorder=3)
    ax[1].set_yticks(range(len(solvers)), slab, fontsize=6)
    ax[1].invert_yaxis()
    ax[1].set(xlim=(-0.05, 1.05), xlabel="Position on the optimal face\n(0 = minimum buffer, 1 = maximum buffer)", title="(b) Where nine solver configurations land")
    ax[1].grid(axis="x", color="0.9", lw=0.6)
    ax[1].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    B = pd.read_csv(PR / "exp_bl.csv")
    B = B[B.cfg == "(6, 2.5, 3)"].sort_values("pc")
    xs = np.arange(len(B))
    for col, lab, c, st in (("fc_min", "Minimum buffer", "C3", "o-"), ("fc_default", "Solver's plan", "0.3", "x--"), ("fc_mid", "Midpoint", "C0", "s-"),
                            ("fc_max", "Maximum buffer", "C2", "^-"), ("fc_roll", "Tuned pipeline", "k", "D:")):
        ax[2].plot(xs, B[col], st, color=c, ms=3.5, lw=1.1, label=lab)
    ax[2].axhline(0, color="0.3", lw=0.6)
    ax[2].text(len(B) - 0.9, 0.4, "rule", fontsize=6, color="0.3")
    ax[2].set_xticks(xs, [f"{k:.0f}\n({h})" for k, h in zip(B.pc, ["3¼ h", "2¼ h", "1½ h", "1 h", "½ h"])], fontsize=6.3)
    ax[2].set(xlabel="Charger power, kW (time to full)", ylabel="Operating cost vs forecast-free rule (%)", title="(c) Battery simulation: plans on the face")
    ax[2].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    tradeoff(ax[3])
    save(fig, "fig5_plan")


# ---------------------------------------------------------------- 图 7（计划价值与预测价值）
def decomposition(rule_fp, fc_fp, tariff, c="(6, 2.5)"):
    RU, FC = pd.read_parquet(rule_fp), pd.read_parquet(fc_fp)
    out = {}
    for pc in (12.0, 25.0):
        g = RU[(RU.tariff == tariff) & (RU.cfg == c) & (RU.pc == pc) & (RU.th >= 0)]
        Mc = (g.assign(cost=g.E + g.Pen)).pivot(index="day", columns="th", values="cost").sort_index()
        h = FC[(FC.tariff == tariff) & (FC.cfg == c) & (FC.pc == pc)].copy()
        h["k"], h["cost"] = h.model + ":" + h.cand, h.E + h.Pen
        Fc = h.pivot(index="day", columns="k", values="cost").sort_index()
        days = Fc.index[Fc.index >= TUNE0]
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        roll = lambda M, fb, cols: np.array([M.loc[d, pick(M[cols], d, fb)] for d in days])
        rule = roll(Mc, k0, list(Mc.columns))
        cols = lambda p: [k for k in Fc.columns if k.startswith(p)]
        rec, lgb = roll(Fc, "seasonal_q:mean@0|mid", cols("seasonal_q:")), roll(Fc, "lgb_local:mean@0|mid", cols("lgb_local:"))
        pi = roll(Fc, "pi:count@0|mid", cols("pi:")) if cols("pi:") else None
        out[pc] = dict(rec=rel_ci(rec, rule), lgb=rel_ci(lgb, rule), pi=rel_ci(pi, rule)[0] if pi is not None else np.nan, fc=rel_ci(lgb, rec))
    return out


def value():
    names = [("shanghai", "Shanghai", OUT / "blp_main_rule.parquet", OUT / "blp_main_fc.parquet"),
             ("guangdong", "Guangdong", OUT / "blp_tariffs_rule.parquet", OUT / "blp_tariffs_fc.parquet"),
             ("pge_ev2a", "PG&E EV2-A", OUT / "blp_tariffs_rule.parquet", OUT / "blp_tariffs_fc.parquet"),
             ("two_block", "Two periods", OUT / "blp_tariffs_rule.parquet", OUT / "blp_tariffs_fc.parquet")]
    rows, fc = [], []
    for tf, lab, rfp, ffp in names:
        res = decomposition(rfp, ffp, tf)
        for pc, r in res.items():
            rows.append((f"{lab}, {pc:.0f} kW", r["rec"], r["lgb"], r["pi"]))
            fc.append((f"A7 {lab}, {pc:.0f} kW", *r["fc"]))
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.3), gridspec_kw={"wspace": 0.75, "width_ratios": [1.15, 1]})
    yy = np.arange(len(rows))[::-1].astype(float)
    for (lab, pr, lg, pi), y in zip(rows, yy):
        pr, lg = [100 * v for v in pr], [100 * v for v in lg]
        if np.isfinite(pi):
            ax[0].plot([0, 100 * pi], [y, y], color="0.85", lw=4, solid_capstyle="butt", zorder=1)
            ax[0].plot(100 * pi, y, "|", color="k", ms=8, mew=1.5, zorder=3)
        ax[0].errorbar(pr[0], y + 0.13, xerr=[[pr[0] - pr[1]], [pr[2] - pr[0]]], fmt="o", color=C_PROF, ms=4, capsize=1.5, lw=0.8, zorder=3)
        ax[0].errorbar(lg[0], y - 0.13, xerr=[[lg[0] - lg[1]], [lg[2] - lg[0]]], fmt="D", color=C_LGB, ms=3.5, capsize=1.5, lw=0.8, zorder=3)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_yticks(yy, [r[0] for r in rows], fontsize=6.5)
    ax[0].plot([], [], "o", color=C_PROF, label="Profile plan (last four weeks, no model)")
    ax[0].plot([], [], "D", color=C_LGB, label="LightGBM pipeline")
    ax[0].plot([], [], "|", color="k", ms=8, mew=1.5, label="Perfect information")
    ax[0].set(xlabel="Operating cost vs forecast-free rule (%)", title="(a) A7 under four tariffs: planning vs forecast")
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    a8 = PR / "exp_bl_a8_profile_fc.csv"
    uev = PR / "uev_profile_fc.csv"
    fc = [(lab, *[100 * v for v in t]) for lab, *t in fc]          # A7 各项：比例 → %
    for fp in (uev, a8):                                            # 这两个文件按 % 记录
        if fp.exists():
            for _, r in pd.read_csv(fp).iterrows():
                fc.append((r.label, r.m, r.lo, r.hi))
    yy = np.arange(len(fc))[::-1].astype(float)
    for (lab, m, lo, hi), y in zip(fc, yy):
        c = "C3" if lab.startswith("A8") else ("C2" if "zones" in lab else C_LGB)
        ax[1].errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="D", color=c, ms=3.5, capsize=1.5, lw=0.8)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_yticks(yy, [r[0] for r in fc], fontsize=6.2)
    ax[1].set(xlabel="LightGBM pipeline vs profile plan (%)", title="(b) What the forecast itself adds")
    for r in rows:
        print("fig6 (a)", r[0], [round(100 * v, 1) for v in r[1]], [round(100 * v, 1) for v in r[2]], round(100 * r[3], 1))
    save(fig, "fig6_value")


# ---------------------------------------------------------------- 图 2（研究框架；与 rev_figures.framework 相同，评估框补充利润口径与服务水平）
def framework():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_xlim(-0.8, 100.8), ax.set_ylim(0, 40), ax.axis("off")
    w, gap = 18.0, 2.5
    X = [0.0 + i * (w + gap) for i in range(5)]
    y0, h = 8.0, 25.5
    cols = ["#e8eef7", "#eaf4e8", "#eaf4e8", "#fdf1e2", "#f6e6ea"]
    head = [("Forecast", "errors $e_t$"), ("Decision\nstructure", "error propagation"), ("Day-ahead\nplan", "plan choice"),
            ("Execution", "recourse"), ("Evaluation", "cost of errors")]
    body = ["9 models from\nnaive benchmarks\nto LightGBM and\nthe foundation\nmodels Chronos-2\nand TimesFM-3.0\n[3.7]",
            "separable: hourly\npower declaration\ncoupled: storage-\nassisted charging;\nbattery-swap\ninventory\n[3.2]",
            "equally optimal\nplans under flat\nTOU periods:\nsolver default;\nmin, mid, max\nbuffer (share α)\n[3.5]",
            "tracking rule;\nhourly LP model;\n15-min battery-\nlevel simulation\n[3.3, 3.6]",
            "lost profit vs\nperfect\ninformation;\n1% service level;\nerror prices $\\pi_t$\nand DSWE;\nbaselines: tuned\nrule, recent-\nprofile plan\n[3.3, 3.4]"]
    for x, c, (h1, h2), b in zip(X, cols, head, body):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=c, ec="0.3", lw=0.8))
        ax.text(x + w / 2, y0 + h - 0.6, h1, ha="center", va="top", fontsize=7.6, weight="bold", linespacing=1.05)
        ax.text(x + w / 2, y0 + h - 5.0, h2, ha="center", va="top", fontsize=6.6, style="italic", color="0.25")
        ax.plot([x + 1.0, x + w - 1.0], [y0 + h - 7.3, y0 + h - 7.3], color="0.6", lw=0.5)
        ax.text(x + w / 2, y0 + h - 8.0, b, ha="center", va="top", fontsize=6.6, linespacing=1.25)
    for x0, x1 in zip(X[:-1], X[1:]):
        ax.annotate("", xy=(x1 - 0.3, y0 + h / 2), xytext=(x0 + w + 0.4, y0 + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.0))
    for x, tag in zip(X[1:4], ["price-setter 1", "price-setter 2\n(often left to\nthe solver)", "price-setter 3"]):
        ax.text(x + w / 2, y0 - 0.9, tag, ha="center", va="top", fontsize=6.8, color="C3", weight="bold", linespacing=1.15)
    ax.text(50, 0.0, "Data: MP-EVData (Shanghai, 9 stations) · UrbanEV (Shenzhen, 275 zones) · 4 TOU tariffs  |  Tests: H1–H4 [3.8]",
            ha="center", va="bottom", fontsize=6.6, color="0.2")
    ax.text(50, 39.8, "Error price $\\pi_t$ = recourse cost − shadow price;   regret ≈ $\\Sigma_t\\ \\pi_t \\times e_t$",
            ha="center", va="top", fontsize=8.0)
    save(fig, "fig2_framework")


# ---------------------------------------------------------------- 图 8（管线何时胜过规则）
def month_table():
    """读 利润口径\\ana_split_month.txt 的逐月表（%，相对滚动规则）。"""
    L = (PR / "ana_split_month.txt").read_text(encoding="utf-8").splitlines()
    head = L[1].split()
    rows = [ln.split() for ln in L[3:] if ln.strip()]
    cols = [" ".join(head[i:i + 2]) for i in range(0, 30, 2)] + head[30:]
    return pd.DataFrame([[float(v) for v in r[1:]] for r in rows], index=[int(r[0]) for r in rows], columns=cols)


def when():
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.9), gridspec_kw={"wspace": 0.45})
    kw = [8.0, 12.0, 18.0, 25.0, 50.0]
    xs = np.arange(5)
    B = pd.read_csv(PR / "exp_bl.csv")
    B = B[B.cfg == "(6, 2.5, 3)"].set_index("pc").loc[kw]
    B2 = pd.read_csv(PR / "exp_bl2_a.csv")
    B2 = B2[B2.cfg == "(6, 2.5, 3)"].set_index("pc").loc[kw]
    C = pd.read_csv(PR / "exp_bl_carry.csv")
    C = C[(C.cfg == "(6, 2.5, 3)") & (C.pol == "调参管线（滚动选）")].set_index("pc").loc[kw]
    sep = np.c_[B.fc_roll, B.fc_roll_lo, B.fc_roll_hi]
    con = np.c_[C.rel, C.lo, C.hi]
    if np.nanmax(np.abs(con)) < 1:
        con = 100 * con
    for k, (v, lab, c, mk) in enumerate(((sep, "Tuned pipeline, days separate", "C0", "o"), (con, "Tuned pipeline, consecutive days", "C1", "s"))):
        ax[0].errorbar(xs + (k - 0.5) * 0.12, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk + "-", color=c, ms=3.5, capsize=1.5, lw=1, label=lab)
    ax[0].plot(xs, B2.pih_roll, "^--", color="0.5", ms=3.5, lw=0.9, label="Perfect hourly forecast")
    ax[0].plot(xs, B.pi_roll, "v:", color="0.2", ms=3.5, lw=0.9, label="Perfect information")
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_xticks(xs, [f"{k:.0f}\n({h})" for k, h in zip(kw, ["3¼ h", "2¼ h", "1½ h", "1 h", "½ h"])], fontsize=6.3)
    ax[0].set(xlabel="Charger power, kW (time to full)", ylabel="Operating cost vs rule (%)", title="(a) Charging time (A7)")
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    M = month_table()
    for k, c in ((8, "C3"), (12, "C1"), (25, "C0")):
        v = M[f"{k}kW 管线"]
        rho = spearmanr(M.y, v).statistic
        ax[1].scatter(M.y, v, s=14, color=c, label=f"{k} kW (ρ = {rho:.2f})")
        b = np.polyfit(M.y, v, 1)
        xx = np.linspace(M.y.min() - 5, M.y.max() + 5, 20)
        ax[1].plot(xx, np.polyval(b, xx), color=c, lw=0.8)
        print(f"fig7 (b) {k} kW rho = {rho:.2f}")
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set(xlabel="Mean demand of the month (kWh/h)", ylabel="Pipeline vs rule (%)", title="(b) Demand relative to stock (A7, by month)")
    ax[1].legend(frameon=False, fontsize=6, loc="lower left")
    Z = pd.read_csv(PR / "exp_bl_uev.csv")
    for pc, c, a in ((12.0, "C0", 0.9), (25.0, "0.6", 0.6)):
        g = Z[Z.pc == pc]
        gain = 100 * (g.tuned / g.rule - 1)
        rho = spearmanr(np.log(g.hmean), gain).statistic
        ax[2].scatter(g.hmean, gain, s=10, color=c, alpha=a, label=f"{pc:.0f} kW chargers (ρ = {rho:.2f})")
        print(f"fig7 (c) {pc:.0f} kW: share cheaper {100 * (gain < 0).mean():.0f}%, median {gain.median():.1f}, rho(log size) {rho:.2f}")
    ax[2].set_xscale("log")
    ax[2].xaxis.set_minor_formatter(plt.NullFormatter())
    ax[2].set_xticks([50, 100, 200, 400, 800], ["50", "100", "200", "400", "800"])
    ax[2].axhline(0, color="k", lw=0.7)
    ax[2].set(xlabel="Mean zone demand (kWh/h, log scale)", ylabel="Pipeline vs rule (%)", title="(c) Scale (60 Shenzhen zones)")
    ax[2].legend(frameon=False, fontsize=6, loc="upper right")
    save(fig, "fig7_when")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["concept", "accuracy", "prices", "dswe", "plan", "value", "when"]
    for t in todo:
        globals()[t]()
