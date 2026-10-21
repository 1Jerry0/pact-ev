"""Applied Energy 投稿版的图（Figure_1—Figure_8 与图形摘要）：数据与 fig_v8.py 完全相同，只改版式。
按最终印刷尺寸作图（全栏宽 6.5 英寸，正文字号 7—8 磅），用色盲友好的 Okabe–Ito 配色，同时输出矢量 PDF 和 600 dpi PNG；
图 6 改为 2×2 排列，图 2（研究框架）重画，新增图形摘要（13 cm 宽）。
写到 results/figures_ae/。用法：python fig_ae.py [concept framework accuracy prices dswe plan value when abstract]"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr

import fig_v8 as V                                   # 复用 fig_v8 的数据读取与计算（路径、口径、种子都相同）

cfg, D, sim = V.cfg, V.D, V.sim
OUT, PR, TAB, S7 = V.OUT, V.PR, V.TAB, V.S7
FIG = V.ROOT / "results" / "figures_ae"
FIG.mkdir(parents=True, exist_ok=True)
W = 6.5                                              # 全栏宽（英寸）
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 120, "pdf.fonttype": 42, "ps.fonttype": 42})
# Okabe–Ito 配色；同一含义在各图中用同一颜色
C = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7",
         black="#000000", grey="#999999", lgrey="#D9D9D9")
C_MIN, C_MID, C_MAX, C_DEF = C["verm"], C["blue"], C["green"], "0.35"       # 最小缓冲（求解器默认）、中点、最大缓冲、电池级默认
C_RULE, C_PROF, C_LGB = C["black"], C["orange"], C["blue"]
MSTYLE = {"naive": (C["grey"], "o"), "naive7": (C["orange"], "o"), "seasonal_q": (C["sky"], "o"), "lqr": (C["verm"], "s"),
          "lgb_local": (C["blue"], "s"), "lgb_global": (C["green"], "s"), "chronos2_cov": (C["purple"], "^"),
          "timesfm3_cov": (C["black"], "^"), "combo5": (C["yellow"], "D")}


def panel(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8, fontweight="bold", pad=4)


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03, **kw)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- 图 1
def concept_data():
    prof, _ = V.profile()
    rng = np.random.default_rng(3)                   # 与 fig_v8.concept 相同的示意误差
    e = np.clip(rng.normal(0, 18, 24), -40, 40)
    e[[20, 21]] = [32, 26]
    return e, prof["wu0"], prof["wu"]


def concept():
    e, pz, pm = concept_data()
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.05), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1, 0.8]})
    hrs = np.arange(24)
    ax[0].bar(hrs, e, color=np.where(e > 0, C["verm"], C["sky"]), width=0.8)
    ax[0].axhline(0, color="0.3", lw=0.6)
    ax[0].set(xlabel="Hour of day", ylabel="Error (kWh; + = under-forecast)", xlim=(-0.6, 23.6), xticks=range(0, 24, 6))
    panel(ax[0], "a", "Errors on an illustrative day")
    ax[1].plot(hrs, pz, color=C_MIN, label="Zero-buffer plan (solver)")
    ax[1].plot(hrs, pm, color=C_MID, label="Midpoint plan")
    ax[1].set(xlabel="Hour of day", ylabel="CNY per kWh of under-forecast", xlim=(0, 23), xticks=range(0, 24, 6), ylim=(-0.02, 0.8))
    ax[1].legend(loc="upper left", handlelength=1.5)
    panel(ax[1], "b", "Under-forecast price, A7")
    bill = [float(np.sum(np.clip(e, 0, None) * p)) for p in (pz, pm)]
    ax[2].bar([0, 1], bill, color=[C_MIN, C_MID], width=0.6)
    ax[2].set_xticks([0, 1], ["Zero-buffer\nplan", "Midpoint\nplan"])
    ax[2].set(ylabel="Cost of the errors (CNY)", ylim=(0, max(bill) * 1.18), xlim=(-0.6, 1.6))
    for i, v in enumerate(bill):
        ax[2].text(i, v + max(bill) * 0.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5)
    panel(ax[2], "c", "Same errors, two costs")
    print("bills:", [round(v, 1) for v in bill])
    save(fig, "Figure_1")


# ---------------------------------------------------------------- 图 2（研究框架）
def framework():
    fig, ax = plt.subplots(figsize=(W, 3.25))
    ax.set_xlim(0, 100), ax.set_ylim(0, 50), ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 43.2), 98, 6, boxstyle="round,pad=0,rounding_size=1.2", fc="#F2F2F2", ec="none"))
    ax.text(50, 46.2, "Error price  $\\pi_t$ = recourse cost $r_t$ − shadow price $\\lambda^*_t$          "
            "regret  ≈  $\\Sigma_t\\,\\pi_t\\,|e_t|$  (DSWE)", ha="center", va="center", fontsize=8.5)
    w, gap, y0, h, hh = 18.4, 2.0, 11.0, 29.0, 6.6
    X = [0.0 + i * (w + gap) for i in range(5)]
    head = ["Forecast", "Decision structure", "Day-ahead plan", "Execution", "Evaluation"]
    hc = ["#4D7EA8", "#3E8E7E", "#3E8E7E", "#C98B2E", "#A65B6E"]
    bc = ["#EAF1F7", "#E8F3F0", "#E8F3F0", "#FAF0E1", "#F5E8EC"]
    body = ["Nine models:\nseasonal and weekly\nnaive, quantile\nregression, LightGBM,\nChronos-2,\nTimesFM-3.0,\ncombination",
            "Separable: hourly\npower declaration\n\nCoupled: storage-\nassisted charging;\nbattery-swap\ninventory",
            "Equal-cost optimal\nface under flat\nTOU periods:\nsolver's default,\nminimum, midpoint\nand maximum buffer",
            "Tracking rule\n\nHourly linear\nprogramme\n\n15-min battery-\nlevel simulation",
            "Regret vs perfect\ninformation;\n1% service level;\nerror prices, DSWE;\nbenchmarks: tuned\nrule, recent profile"]
    sec = ["Section 2.7", "Section 2.2", "Section 2.5", "Sections 2.3, 2.6", "Sections 2.3, 2.4"]
    for x, h1, b, s, c1, c2 in zip(X, head, body, sec, hc, bc):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=c2, ec=c1, lw=0.9))
        ax.add_patch(FancyBboxPatch((x, y0 + h - hh), w, hh, boxstyle="round,pad=0,rounding_size=1.2", fc=c1, ec=c1, lw=0.9))
        ax.text(x + w / 2, y0 + h - hh / 2, h1, ha="center", va="center", fontsize=7.5, color="white", weight="bold")
        ax.text(x + w / 2, y0 + h - hh - 1.3, b, ha="center", va="top", fontsize=6.8, linespacing=1.2, color="0.1")
        ax.text(x + w / 2, y0 + 1.2, s, ha="center", va="bottom", fontsize=6.5, color="0.35", style="italic")
    for x0, x1 in zip(X[:-1], X[1:]):
        ax.annotate("", xy=(x1 - 0.2, y0 + h / 2), xytext=(x0 + w + 0.2, y0 + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.1, mutation_scale=9))
    tags = ["Price-setter 1:\nerror propagation", "Price-setter 2: buffer\n(often left to the solver)", "Price-setter 3:\nrecourse"]
    for x, t in zip(X[1:4], tags):
        ax.text(x + w / 2, y0 - 1.0, t, ha="center", va="top", fontsize=7, color=C["verm"], weight="bold", linespacing=1.15)
    ax.text(50, 0.0, "Data: MP-EVData (Shanghai, nine stations); UrbanEV (Shenzhen, 275 zones); four TOU tariffs\n"
            "Pre-specified hypotheses H1–H4 and experimental design: Sections 2.7–2.8",
            ha="center", va="bottom", fontsize=6.8, color="0.25", linespacing=1.3)
    save(fig, "Figure_2")


# ---------------------------------------------------------------- 图 3（精度与成本）
def accuracy():
    A = pd.read_csv(TAB / "e1_accuracy.csv")
    A = A[(A.ds == "mp") & (~A.cal) & A.model.isin(V.MAIN)]
    E = pd.read_csv(TAB / "e2_declaration.csv")
    E = E[(E.strategy == "prob_cal") & (E.b == 0.0) & (E.tau == 0.83) & E.model.isin(V.MAIN)]
    ST = pd.read_parquet(OUT / "cb_storage_sched.parquet")
    ST = ST[ST.tie == "mid"]
    reg = (ST.E + ST.Pen - ST.cost_opt).groupby([ST.station, ST.model]).mean()
    P = pd.read_parquet(OUT / "cb_hourly_plans.parquet", filters=[("inp", "==", "mean"), ("tie", "==", "mid")])
    P = P[P.model.isin(V.MAIN)]
    a7 = (P.E + P.Pen - P.E_opt - P.Pen_opt).groupby(P.model).mean()
    reg = pd.concat([reg, pd.Series(a7.to_numpy(), index=pd.MultiIndex.from_product([[S7], a7.index], names=["station", "model"]))])
    reg.index.names = ["station", "model"]
    reg = (reg / reg.xs("naive", level="model").reindex(reg.index.get_level_values(0)).to_numpy()).rename("rel_regret").reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"wspace": 0.42})
    x = A.merge(E, left_on=["series", "model"], right_on=["station", "model"])
    for m in V.MAIN:
        g = x[x.model == m]
        c, mk = MSTYLE[m]
        ax[0].scatter(g.rel_CRPS, g.rel, s=13, color=c, marker=mk, edgecolor="0.2", linewidth=0.3, label=V.MODEL[m], zorder=3)
    r0 = x.groupby("station").apply(lambda g: spearmanr(g.rel_CRPS, g.rel).statistic, include_groups=False).median()
    ax[0].set(xlabel="CRPS relative to seasonal naive", ylabel="Cost relative to seasonal naive")
    ax[0].text(0.97, 0.04, f"cost ratio τ = 0.83\nmedian ρ = {r0:.2f}", transform=ax[0].transAxes, ha="right", va="bottom", fontsize=7)
    panel(ax[0], "a", "Power declaration")
    y = A.merge(reg, left_on=["series", "model"], right_on=["station", "model"])
    for m in V.MAIN:
        g = y[y.model == m]
        c, mk = MSTYLE[m]
        ax[1].scatter(g.rel_MAE, g.rel_regret, s=13, color=c, marker=mk, edgecolor="0.2", linewidth=0.3, zorder=3)
    r1 = y.groupby("station").apply(lambda g: spearmanr(g.rel_MAE, g.rel_regret).statistic, include_groups=False).median()
    ax[1].set(xlabel="MAE relative to seasonal naive", ylabel="Regret relative to seasonal naive")
    ax[1].text(0.97, 0.04, f"median ρ = {r1:.2f}", transform=ax[1].transAxes, ha="right", va="bottom", fontsize=7)
    panel(ax[1], "b", "Storage and swapping plans")
    Z = V.zone_agreement()
    ax[2].scatter(Z.cv_hour, Z.agree, s=6, color=C["grey"], alpha=0.7, linewidth=0)
    b = np.polyfit(Z.cv_hour, Z.agree, 1)
    xx = np.linspace(Z.cv_hour.min(), Z.cv_hour.max(), 50)
    ax[2].plot(xx, np.polyval(b, xx), color=C["verm"])
    ax[2].set(xlabel="Hourly coefficient of variation", ylabel="Agreement of accuracy and regret")
    panel(ax[2], "c", "UrbanEV zones (n = 275)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.05), columnspacing=1.2,
               handletextpad=0.3)
    print("fig3 median rho (a), (b):", round(r0, 2), round(r1, 2))
    save(fig, "Figure_3")


# ---------------------------------------------------------------- 图 4（误差价格）
def prices():
    prof, tou = V.profile()
    P = pd.read_pickle(PR / "bl_prices.pkl")
    P = P[pd.to_datetime(P.day) >= cfg.MP_EVAL_START]
    fig = plt.figure(figsize=(W, 4.45))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.95, wspace=0.32)
    a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    hrs = np.arange(24)
    a0.step(hrs, tou, where="mid", color="0.55", lw=0.9, ls=":", label="Electricity price")
    a0.plot(hrs, prof["wu0"], color=C_MIN, label=f"Solver's plan (zero buffer), mean {prof['wu0'].mean():.2f}")
    a0.plot(hrs, prof["wu"], color=C_MID, label=f"Midpoint plan, mean {prof['wu'].mean():.2f}")
    a0.set(xlabel="Hour of day", ylabel="CNY per kWh of under-forecast", xlim=(0, 23), xticks=range(0, 24, 6))
    a0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    panel(a0, "a", "Hourly model")
    for pc, ls in ((12.0, "-"), (25.0, "--")):
        for tie, col, lab in (("default", C_MIN, "solver's plan"), ("max", C_MAX, "maximum buffer")):
            g = P[(P.cfg == "(6, 2.5, 3)") & (P.pc == pc) & (P.tie == tie)]
            w = np.mean(np.vstack(g.wu.to_list()), axis=0)
            a1.plot(hrs, w, color=col, ls=ls, lw=1.1, label=f"{pc:.0f} kW, {lab}, mean {w.mean():.2f}".replace("-", "−"))
    a1.step(hrs, tou, where="mid", color="0.55", lw=0.9, ls=":")
    a1.axhline(0, color="0.3", lw=0.5)
    a1.set(xlabel="Hour of day", ylabel="CNY per kWh of under-forecast", xlim=(0, 23), xticks=range(0, 24, 6))
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=1.8)
    panel(a1, "b", "Battery-level simulation")
    rows = [("A1 storage, hourly model", *V.disp_h2("A1", "storage"), C["grey"]), ("A3 storage, hourly model", *V.disp_h2("A3", "storage"), C["grey"]),
            ("A7 swapping, hourly model", *V.disp_h2(S7, "anchor"), C["blue"])]
    for _, r in pd.read_csv(PR / "e4_dispersion.csv").iterrows():
        rows.append((r.label.replace("A7 battery", "A7 battery level"), r.m, r.lo, r.hi, C["verm"]))
    lab, val, lo, hi, col = zip(*rows)
    val, lo, hi = map(np.array, (val, lo, hi))
    yy = np.arange(len(val))[::-1]
    a2.barh(yy, val, color=col, height=0.62)
    a2.errorbar(val, yy, xerr=[val - lo, hi - val], fmt="none", ecolor="k", capsize=2, lw=0.7)
    a2.set_yticks(yy, lab)
    a2.set(xlabel="P90 − P10 of same-day cost divided by the mean cost of the errors")
    panel(a2, "c", "Dispersion of same-day cost when errors are permuted within TOU periods")
    print("fig4 (c):", [(l, round(v, 3)) for l, v in zip(lab, val)])
    save(fig, "Figure_4")


# ---------------------------------------------------------------- 图 5（DSWE；数值见 fig_v8.dswe 的注释）
def dswe():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"wspace": 0.18, "width_ratios": [0.62, 1.6]})
    cols = [("MAE + RMSE + CRPS", C["grey"]), ("+ errors at fixed tariff weights", C["sky"]),
            ("+ errors at ex-post error prices", C["orange"]), ("+ errors at ex-ante error prices", C["blue"])]
    vals = [(0.221, 0.385), (0.304, 0.492), (0.515, 0.570), (0.560, 0.561)]
    x = np.arange(2)
    for j, (v, (lab, c)) in enumerate(zip(vals, cols)):
        ax[0].bar(x + (j - 1.5) * 0.2, v, 0.2, label=lab, color=c)
    ax[0].set_xticks(x, ["Pooled", "A7"])
    ax[0].set(ylabel="Within-day $R^2$ of cross-model\ncost differences", ylim=(0, 1.05))
    panel(ax[0], "a", "Pooled panel")
    st = ["A1\ntaxi", "A2\ntaxi", "A3\nbus", "A4\nbus", "A5\nresid.", "A6\nresid.", "A9\ntruck", "A7\nswap"]
    acc = [0.356, 0.326, 0.108, 0.071, 0.218, 0.219, 0.485, 0.385]
    post = [0.561, 0.700, 0.978, 0.985, 0.224, 0.321, 0.543, 0.570]
    ante = [0.612, 0.707, 0.928, 0.836, 0.281, 0.397, 0.506, 0.561]
    x = np.arange(len(st))
    for j, (v, (lab, c)) in enumerate(zip((acc, post, ante), (cols[0], cols[2], cols[3]))):
        ax[1].bar(x + (j - 1) * 0.27, v, 0.27, color=c)
    ax[1].set_xticks(x, st)
    ax[1].set(ylim=(0, 1.05))
    ax[1].tick_params(axis="y", labelleft=True)
    panel(ax[1], "b", "By station")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2,
               handlelength=1.2)
    save(fig, "Figure_5")


# ---------------------------------------------------------------- 图 6（等成本计划，2×2）
def example_plans():
    H = D.mp_hourly()
    hmean = float(H[S7].loc[:cfg.MP_DEV_END].mean())
    day = pd.Timestamp("2024-06-15")
    idx = pd.date_range(day, periods=24, freq="h")
    y = H[S7].reindex(idx).to_numpy()
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    f = np.maximum(F[F.series == S7].set_index("t")["mean"].reindex(idx).to_numpy(), 0)
    price, pen = D.price(S7, day.month), V.kappa_profit(day.month)
    cap, pmax = 6 * hmean, 2.5 * hmean
    opt = sim.bss_plan(y, price, pen, cap, pmax, None)[2]
    pl = {}
    for tie in ("min", "max"):
        g, s, _, u = sim.bss_plan(f, price, pen, cap, pmax, tie)
        pl[tie] = ("bss", g, s, f, u)
    pl["mid"] = V.mix(pl["min"], pl["max"], 0.5)
    regs = {}
    for tie in ("min", "mid", "max"):
        e, sh, pn = V.exec_split(pl[tie], y, price, pen, cap, pmax)
        regs[tie] = e + pn - opt
    return price, cap, pl, regs


def plan():
    price, cap, pl, regs = example_plans()
    fig = plt.figure(figsize=(W, 5.7))
    gs = fig.add_gridspec(2, 2, hspace=0.78, wspace=0.62, width_ratios=[1, 1.08])
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for lv in np.unique(price):
        for h in np.where(price == lv)[0]:
            ax[0].axvspan(h, h + 1, color=plt.cm.Greys(0.08 + 0.42 * (lv - price.min()) / (price.max() - price.min())), lw=0)
    for tie, lab, c in (("min", "Minimum buffer (solver's plan)", C_MIN), ("mid", "Midpoint", C_MID), ("max", "Maximum buffer", C_MAX)):
        ax[0].plot(np.arange(25), pl[tie][2] / cap, color=c, label=f"{lab}: regret {regs[tie]:,.0f} CNY")
    print("example day regrets:", {k: round(v, 0) for k, v in regs.items()})
    ax[0].set(xlim=(0, 24), ylim=(0, 1.05), xticks=range(0, 25, 6), xlabel="Hour (shading: darker = higher price)", ylabel="Planned stock / capacity")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    panel(ax[0], "a", "Three equal-cost plans, 15 June 2024")
    fs = PR / "exp_solvers.parquet"
    R = pd.read_parquet(fs if fs.exists() else OUT / "exp_solvers.parquet")
    forms = ["F1 小时库存 LP", "F1r 同上，变量倒序", "F1e 同上，消去库存变量", "F2 电池级 LP"]
    flab = ["Hourly LP", "Hourly LP, columns reversed", "Hourly LP, stock eliminated", "Battery-level LP"]
    solvers = ["HiGHS 对偶单纯形（SciPy，论文所用）", "HiGHS 不预处理的对偶单纯形", "HiGHS 原始单纯形", "HiGHS 内点 + 交叉", "HiGHS 内点，不交叉",
               "GLOP（OR-Tools）", "CLP（OR-Tools）", "CBC（PuLP）", "PDLP（一阶法）"]
    slab = ["HiGHS dual simplex (default)", "HiGHS dual, no presolve", "HiGHS primal simplex", "HiGHS IPM + crossover", "HiGHS IPM",
            "GLOP", "CLP", "CBC", "PDLP"]
    med = R.dropna(subset=["pos"]).groupby(["solver", "form"]).pos.median().unstack("form").reindex(index=solvers, columns=forms)
    for j, fm in enumerate(forms):
        ax[1].scatter(med[fm], np.arange(len(solvers)) + (j - 1.5) * 0.16, marker=["o", "s", "^", "D"][j], s=16, label=flab[j],
                      color=[C["verm"], C["orange"], C["purple"], C["green"]][j], edgecolor="0.2", linewidth=0.3, zorder=3)
    ax[1].set_yticks(range(len(solvers)), slab)
    ax[1].invert_yaxis()
    ax[1].set(xlim=(-0.05, 1.05), xticks=[0, 0.25, 0.5, 0.75, 1], xlabel="Position on the optimal face\n(0 = minimum buffer, 1 = maximum buffer)")
    ax[1].grid(axis="x", color="0.9", lw=0.6)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.35, -0.3), ncol=2, columnspacing=1.0, handletextpad=0.2)
    panel(ax[1], "b", "Plans returned by solvers")
    B = pd.read_csv(PR / "exp_bl.csv")
    B = B[B.cfg == "(6, 2.5, 3)"].sort_values("pc")
    xs = np.arange(len(B))
    for col, lab, c, st in (("fc_min", "Minimum buffer", C_MIN, "o-"), ("fc_default", "Solver's plan", C_DEF, "x--"), ("fc_mid", "Midpoint", C_MID, "s-"),
                            ("fc_max", "Maximum buffer", C_MAX, "^-"), ("fc_roll", "Tuned pipeline", C["black"], "D:")):
        ax[2].plot(xs, B[col], st, color=c, ms=3.5, lw=1.0, label=lab)
    ax[2].axhline(0, color="0.3", lw=0.6)
    ax[2].text(len(B) - 1.05, 0.35, "rule", fontsize=7, color="0.3")
    ax[2].set_xticks(xs, [f"{k:.0f}\n({h})" for k, h in zip(B.pc, ["3¼ h", "2¼ h", "1½ h", "1 h", "½ h"])])
    ax[2].set(xlabel="Charger power, kW (time to full)", ylabel="Operating cost vs forecast-free rule (%)")
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, columnspacing=1.0)
    panel(ax[2], "c", "Battery-level simulation")
    tradeoff(ax[3])
    panel(ax[3], "d", "Service–cost trade-off (hourly model)")
    save(fig, "Figure_6")


def tradeoff(ax, c="(6, 2.5)"):
    TP = pd.read_parquet(OUT / "cb_tradeoff_points.parquet")
    g = TP[(TP.cfg == c) & (TP.model == "lgb_local")]
    order = ["mean", "q50", "q60", "q70", "q80", "q90", "q95"]
    for tie, col, mk, lab in (("default", C_MIN, "o", "Solver's plan"), ("mid", C_MID, "s", "Midpoint plan"), ("max", C_MAX, "^", "Maximum buffer")):
        h = g[g.tie == tie].set_index("inp").reindex(order).dropna(subset=["share"]).sort_values("share")
        ax.plot(100 * h.share, 100 * h.exE, mk + "-", color=col, ms=3, lw=1, label=lab)
    RU = pd.read_parquet(OUT / "cb_hourly_rule.parquet")
    RU = RU[(RU.cfg == c) & (RU.day >= cfg.MP_EVAL_START)]
    base = RU.drop_duplicates("day")
    r = RU.groupby("th").agg(E=("E", "sum"), S=("S", "sum"))
    sh, ex = 100 * r.S / base.dem.sum(), 100 * (r.E / base.E_opt.sum() - 1)
    o = np.argsort(sh.to_numpy())
    env = np.minimum.accumulate(ex.to_numpy()[o])
    ax.step(sh.to_numpy()[o], env, where="post", color=C["black"], lw=1, ls=":", label="Forecast-free rule (frontier)")
    print("fig6 (d) rule frontier at 1% service:", round(float(env[sh.to_numpy()[o] <= 1.0].min()), 1))
    ax.axvline(1.0, color="0.5", lw=0.7, ls="--")
    ax.text(1.12, 23.5, "1% service level", fontsize=7, va="top", color="0.35")
    ax.set(xlim=(0, 6), ylim=(-3, 25), xlabel="Unserved demand (% of demand)", ylabel="Electricity cost above perfect\ninformation (%)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86))


# ---------------------------------------------------------------- 图 7（计划价值与预测价值）
def value():
    names = [("shanghai", "Shanghai", OUT / "blp_main_rule.parquet", OUT / "blp_main_fc.parquet"),
             ("guangdong", "Guangdong", OUT / "blp_tariffs_rule.parquet", OUT / "blp_tariffs_fc.parquet"),
             ("pge_ev2a", "PG&E EV2-A", OUT / "blp_tariffs_rule.parquet", OUT / "blp_tariffs_fc.parquet"),
             ("two_block", "Two periods", OUT / "blp_tariffs_rule.parquet", OUT / "blp_tariffs_fc.parquet")]
    rows, fc = [], []
    for tf, lab, rfp, ffp in names:
        for pc, r in V.decomposition(rfp, ffp, tf).items():
            rows.append((f"{lab}, {pc:.0f} kW", r["rec"], r["lgb"], r["pi"]))
            fc.append((f"A7 {lab}, {pc:.0f} kW", *r["fc"]))
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.15), gridspec_kw={"wspace": 0.85, "width_ratios": [1.15, 1]})
    yy = np.arange(len(rows))[::-1].astype(float)
    for (lab, pr, lg, pi), y in zip(rows, yy):
        pr, lg = [100 * v for v in pr], [100 * v for v in lg]
        if np.isfinite(pi):
            ax[0].plot([0, 100 * pi], [y, y], color=C["lgrey"], lw=4, solid_capstyle="butt", zorder=1)
            ax[0].plot(100 * pi, y, "|", color="k", ms=7, mew=1.4, zorder=3)
        ax[0].errorbar(pr[0], y + 0.14, xerr=[[pr[0] - pr[1]], [pr[2] - pr[0]]], fmt="o", color=C_PROF, ms=3.5, capsize=1.5, lw=0.8, zorder=3)
        ax[0].errorbar(lg[0], y - 0.14, xerr=[[lg[0] - lg[1]], [lg[2] - lg[0]]], fmt="D", color=C_LGB, ms=3, capsize=1.5, lw=0.8, zorder=3)
    ax[0].axvline(0, color="k", lw=0.7)
    ax[0].set_yticks(yy, [r[0] for r in rows])
    ax[0].plot([], [], "o", color=C_PROF, ms=3.5, label="Recent-profile plan (no model)")
    ax[0].plot([], [], "D", color=C_LGB, ms=3, label="LightGBM pipeline")
    ax[0].plot([], [], "|", color="k", ms=7, mew=1.4, label="Perfect information")
    ax[0].set(xlabel="Operating cost vs forecast-free rule (%)")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2, columnspacing=1.0)
    panel(ax[0], "a", "A7 under four tariffs")
    fc = [(lab, *[100 * v for v in t]) for lab, *t in fc]
    for fp in (PR / "uev_profile_fc.csv", PR / "exp_bl_a8_profile_fc.csv"):
        for _, r in pd.read_csv(fp).iterrows():
            fc.append((r.label, r.m, r.lo, r.hi))
    yy = np.arange(len(fc))[::-1].astype(float)
    for (lab, m, lo, hi), y in zip(fc, yy):
        c = C["verm"] if lab.startswith("A8") else (C["green"] if "zones" in lab else C_LGB)
        ax[1].errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="D", color=c, ms=3, capsize=1.5, lw=0.8)
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].set_yticks(yy, [r[0] for r in fc])
    ax[1].set(xlabel="LightGBM pipeline vs recent-profile plan (%)")
    panel(ax[1], "b", "What the forecast itself adds")
    save(fig, "Figure_7")


# ---------------------------------------------------------------- 图 8（管线何时胜过规则）
def when():
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw={"wspace": 0.5, "width_ratios": [1.3, 1, 1]})
    kw = [8.0, 12.0, 18.0, 25.0, 50.0]
    xs = np.arange(5)
    B = pd.read_csv(PR / "exp_bl.csv")
    B = B[B.cfg == "(6, 2.5, 3)"].set_index("pc").loc[kw]
    B2 = pd.read_csv(PR / "exp_bl2_a.csv")
    B2 = B2[B2.cfg == "(6, 2.5, 3)"].set_index("pc").loc[kw]
    Cc = pd.read_csv(PR / "exp_bl_carry.csv")
    Cc = Cc[(Cc.cfg == "(6, 2.5, 3)") & (Cc.pol == "调参管线（滚动选）")].set_index("pc").loc[kw]
    sep = np.c_[B.fc_roll, B.fc_roll_lo, B.fc_roll_hi]
    con = np.c_[Cc.rel, Cc.lo, Cc.hi]
    if np.nanmax(np.abs(con)) < 1:
        con = 100 * con
    for k, (v, lab, c, mk) in enumerate(((sep, "Tuned pipeline, days separate", C["blue"], "o"),
                                         (con, "Tuned pipeline, consecutive days", C["orange"], "s"))):
        ax[0].errorbar(xs + (k - 0.5) * 0.14, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk + "-", color=c, ms=3, capsize=1.5,
                       lw=0.9, label=lab)
    ax[0].plot(xs, B2.pih_roll, "^--", color=C["grey"], ms=3, lw=0.9, label="Perfect hourly forecast")
    ax[0].plot(xs, B.pi_roll, "v:", color=C["black"], ms=3, lw=0.9, label="Perfect information")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(xs, [f"{k:.0f}\n({h})" for k, h in zip(kw, ["3¼ h", "2¼ h", "1½ h", "1 h", "½ h"])], fontsize=6.5)
    ax[0].set(xlabel="Charger power, kW (time to full)", ylabel="Operating cost vs rule (%)")
    panel(ax[0], "a", "Charging time (A7)")
    M = V.month_table()
    for k, c in ((8, C["verm"]), (12, C["orange"]), (25, C["blue"])):
        v = M[f"{k}kW 管线"]
        rho = spearmanr(M.y, v).statistic
        ax[1].scatter(M.y, v, s=12, color=c, label=f"{k} kW, ρ = {rho:.2f}".replace("-", "−"), zorder=3)
        b = np.polyfit(M.y, v, 1)
        xx = np.linspace(M.y.min() - 5, M.y.max() + 5, 20)
        ax[1].plot(xx, np.polyval(b, xx), color=c, lw=0.8)
        print(f"fig8 (b) {k} kW rho = {rho:.2f}")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="Mean monthly demand (kWh/h)", ylabel="Pipeline vs rule (%)", xlim=(220, 368), ylim=(-9.6, 4.8))
    ax[1].legend(loc="lower left", handletextpad=0.1, fontsize=6.3, borderaxespad=0.2, labelspacing=0.3)
    panel(ax[1], "b", "Demand vs stock (A7)")
    Z = pd.read_csv(PR / "exp_bl_uev.csv")
    for pc, c, a in ((12.0, C["blue"], 0.85), (25.0, C["grey"], 0.75)):
        g = Z[Z.pc == pc]
        gain = 100 * (g.tuned / g.rule - 1)
        rho = spearmanr(np.log(g.hmean), gain).statistic
        ax[2].scatter(g.hmean, gain, s=8, color=c, alpha=a, linewidth=0, label=f"{pc:.0f} kW (ρ = {rho:.2f})".replace("-", "−"))
        print(f"fig8 (c) {pc:.0f} kW: share cheaper {100 * (gain < 0).mean():.0f}%, rho(log size) {rho:.2f}")
    ax[2].set_xscale("log")
    ax[2].xaxis.set_minor_formatter(plt.NullFormatter())
    ax[2].set_xticks([50, 100, 200, 400, 800], ["50", "100", "200", "400", "800"])
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(xlabel="Zone demand (kWh/h, log scale)", ylabel="Pipeline vs rule (%)")
    ax[2].legend(loc="upper right", handletextpad=0.1, fontsize=6.5, borderaxespad=0.2)
    panel(ax[2], "c", "Scale (60 Shenzhen zones)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1), columnspacing=2.0)
    save(fig, "Figure_8")


# ---------------------------------------------------------------- 图形摘要（13 cm 宽）
def abstract():
    e, pz, pm = concept_data()
    price, cap, pl, regs = example_plans()
    fig = plt.figure(figsize=(5.12, 2.8))
    fig.text(0.5, 0.975, "More accurate forecasts, not necessarily lower costs", ha="center", va="top", fontsize=9.5, weight="bold")
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.585, top=0.78, bottom=0.33, wspace=0.5)
    a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for lv in np.unique(price):
        for h in np.where(price == lv)[0]:
            a0.axvspan(h, h + 1, color=plt.cm.Greys(0.08 + 0.42 * (lv - price.min()) / (price.max() - price.min())), lw=0)
    a0.plot(np.arange(25), pl["min"][2] / cap, color=C_MIN, lw=1.4)
    a0.plot(np.arange(25), pl["mid"][2] / cap, color=C_MID, lw=1.4)
    a0.set(xlim=(0, 24), ylim=(0, 1.05), xticks=[0, 12, 24], yticks=[0, 0.5, 1], xlabel="Hour")
    a0.set_ylabel("Planned stock / capacity", fontsize=7)
    a0.set_title("Same planned cost", fontsize=7.5, loc="left", weight="bold")
    hrs = np.arange(24)
    a1.plot(hrs, pz, color=C_MIN, lw=1.4, label="Zero-buffer plan (solver's default)")
    a1.plot(hrs, pm, color=C_MID, lw=1.4, label="Midpoint plan")
    a1.set(xlim=(0, 23), xticks=[0, 12, 23], ylim=(-0.02, 0.8), yticks=[0, 0.4, 0.8], xlabel="Hour")
    a1.set_ylabel("CNY per kWh", fontsize=7)
    a1.set_title("Under-forecast price", fontsize=7.5, loc="left", weight="bold")
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.335, 0.075), ncol=1, fontsize=7, handlelength=1.6)
    msgs = [("When storage couples the hours,\nthe most accurate forecast is\noften not the cheapest", C["black"]),
            ("Choosing the midpoint plan cuts\nforecast-induced losses by 47%\nat no planning cost", C_MID),
            ("A four-week demand profile\ncaptures nearly all savings of\nforecast-based planning", C_PROF)]
    for k, (t, c) in enumerate(msgs):
        fig.text(0.635, 0.745 - k * 0.265, t, ha="left", va="center", fontsize=7, color="0.1", linespacing=1.2,
                 bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.3", fc="white", ec=c, lw=1.1))
    fig.text(0.5, 0.012, "Nine Shanghai stations · 275 Shenzhen zones · nine forecasting models, including foundation models",
             ha="center", va="bottom", fontsize=6.3, color="0.35")
    fig.savefig(FIG / "Graphical_abstract.pdf")
    fig.savefig(FIG / "Graphical_abstract.png", dpi=400)
    fig.savefig(FIG / "Graphical_abstract.tif", dpi=400, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved Graphical_abstract")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["concept", "framework", "accuracy", "prices", "dswe", "plan", "value", "when", "abstract"]
    for t in todo:
        globals()[t]()
