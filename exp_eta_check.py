"""充放电效率敏感性（小时模型，A7，数据锚定网格，LightGBM 均值预测，利润口径）：η 取行业标准 NB/T 33001—2018 的下限 0.93，对照主分析 0.95。
指标：中点计划相对求解器默认计划的日遗憾降幅；两种计划的缺供率；LightGBM 中点计划相对每日按前 28 天调参的规则（66 组阈值）的总遗憾。
单进程运行：先改 cfg.ETA 再导入 sim 与 exp_cost_basis，使计划、执行和终值都用同一个 η。用法：python exp_eta_check.py 0.93 → 结果/eta_check_093.txt"""
import pathlib
import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import cfg

ETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.93
cfg.ETA = ETA
import data as D  # noqa: E402
import sim  # noqa: E402
import exp_cost_basis as CB  # noqa: E402
from analysis import boot_mean  # noqa: E402
from exp_packs_value import rel_ci  # noqa: E402

assert sim.ETA == ETA and CB.ETA == ETA
S = cfg.BSS
H = D.mp_hourly()
hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local_cal.parquet")
F = F[F.series == S].set_index("t")
rows, rule = [], []
for d in pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D"):
    idx = pd.date_range(d, periods=24, freq="h")
    y = H[S].reindex(idx).to_numpy()
    if not np.isfinite(y).all():
        continue
    price, pen = D.price(S, d.month), CB.kappa_profit(d.month)
    f = F["mean"].reindex(idx).to_numpy() if d >= cfg.MP_EVAL_START else None
    if f is not None and not np.isfinite(f).all():
        f = None
    for c, p in CB.CFGS:
        cap, pmax = c * hmean, p * hmean
        g0, s0_, obj, u0 = sim.bss_plan(y, price, pen, cap, pmax, None)
        opt = obj
        E, Sh, Pn = CB.rule_split(y, price, pen, cap, pmax)
        rule += [(d, f"({c}, {p})", i, float(E[i] + Pn[i] - opt)) for i in range(len(CB.TH))]
        if f is None:
            continue
        f = np.maximum(f, 0)
        pl = {}
        for tie in ("default", "min", "max"):
            gg, ss, _, uu = sim.bss_plan(f, price, pen, cap, pmax, tie)
            pl[tie] = ("bss", gg, ss, f, uu)
        pl["mid"] = CB.mix(pl["min"], pl["max"], 0.5)
        r = dict(day=d, cfg=f"({c}, {p})", dem=float(y.sum()))
        for tie in ("default", "mid"):
            e, sh, pn = CB.exec_split(pl[tie], y, price, pen, cap, pmax)
            r[f"reg_{tie}"], r[f"short_{tie}"] = e + pn - opt, sh
        rows.append(r)
X = pd.DataFrame(rows)
R = pd.DataFrame(rule, columns=["day", "cfg", "th", "reg"])
lines = [f"η = {ETA}；{X.day.nunique()} 天 × {X.cfg.nunique()} 种配置；LightGBM 均值预测"]
daily = X.groupby("day")[["reg_default", "reg_mid", "short_default", "short_mid", "dem"]].sum()
m, ci, _ = boot_mean(1 - daily.reg_mid / daily.reg_default)
lines.append(f"中点 vs 求解器默认计划：日遗憾降幅 {100 * m:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]；总遗憾之比 {daily.reg_mid.sum() / daily.reg_default.sum():.3f}；"
             f"缺供率 默认 {100 * daily.short_default.sum() / daily.dem.sum():.2f}% → 中点 {100 * daily.short_mid.sum() / daily.dem.sum():.2f}%")
TUNE0 = pd.Timestamp("2024-05-01")
rr = []
for c, g in R.groupby("cfg"):
    M = g.pivot(index="day", columns="th", values="reg").sort_index()
    k0 = M[M.index < cfg.MP_EVAL_START].sum().idxmin()
    days = M.index[M.index >= TUNE0]
    picks = [M.loc[x - pd.Timedelta(days=29): x - pd.Timedelta(days=2)].sum().idxmin() if len(M.loc[x - pd.Timedelta(days=29): x - pd.Timedelta(days=2)]) >= 7 else k0 for x in days]
    rr.append(pd.Series([M.loc[x, k] for x, k in zip(days, picks)], index=days))
rr = pd.concat(rr).groupby(level=0).sum()
lm = X.groupby("day").reg_mid.sum().reindex(rr.index)
m, lo, hi = rel_ci(lm.to_numpy(), rr.to_numpy())
lines.append(f"LightGBM 中点计划 vs 调参规则（5 月起 {len(rr)} 天）：总遗憾 {100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]")
txt = "\n".join(lines)
(ROOT / "results" / f"eta_check_{int(round(ETA * 100)):03d}.txt").write_text(txt, encoding="utf-8")
print(txt)
