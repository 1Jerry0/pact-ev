"""电池级 LP 计划的两项补充（沿用 exp_bl.py 的 LP、仿真与配置）：
A. "完全的逐小时预测"：把真实的逐小时需求均摊到 15 分钟（不含小时内的到达时刻信息），与"完全预测（真实逐 15 分钟到达）"分开，
   看日前逐小时预测的上限在哪；同时记录求解器默认解在最优面上的位置（按满电块数 ΣR 归一到 0–1）。
B. 9 个模型：均值预测 × {最小, 中点, 最大, 默认} 与"最大库存 + 2 步裕量"，看计划选择对每个模型是否成立、模型成本排序。
用法：python exp_bl2.py A | B | ana"""
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
from scipy.stats import kendalltau
from joblib import Parallel, delayed
import cfg
import data as D
from analysis import MAIN_POOL
from exp_bl import lp_bl, position, run_day_bl, nsteps, load_events, GRID, NS
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[1] / "results"
S = cfg.BSS
GRID_B = [((6, 2.5, 3), 12.0), ((4, 2.5, 3), 12.0), ((6, 2.5, 3), 18.0), ((6, 2.5, 3), 25.0), ((6, 2.5, 10), 12.0)]


def setup(par, pc, tot, hmean, ebar):
    cap, pmax = par[0] * hmean, par[1] * hmean
    pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
    return pen, max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)


def task_a(d, ev_step, ev_e, y, price, tot, hmean, f, ebar):
    rows, pos = [], []
    ee = np.minimum(ev_e, 1.6 * ebar)
    dh, df = np.repeat(y / ebar / 4, 4), np.repeat(np.maximum(f, 0) / ebar / 4, 4)
    for par, pc in GRID:
        pen, K, C, n = setup(par, pc, tot, hmean, ebar)
        key = (d, str(par), pc)
        for mg in (0, 1, 2):
            x0, f0 = lp_bl(dh, price, pen, K, C, n + mg, ebar)
            xl, _ = lp_bl(dh, price, pen, K, C, n + mg, ebar, 1.0, f0)
            xh, _ = lp_bl(dh, price, pen, K, C, n + mg, ebar, -1.0, f0)
            for tie, x in (("min", xl), ("mid", (xl + xh) / 2), ("max", xh), ("default", x0)):
                c, u = run_day_bl(ev_step, ee, price, pen, K, C, pc, ebar, position(x, n + mg))
                rows.append((*key, f"pih@{mg}|{tie}", c, u))
        x0, f0 = lp_bl(df, price, pen, K, C, n, ebar)
        xl, _ = lp_bl(df, price, pen, K, C, n, ebar, 1.0, f0)
        xh, _ = lp_bl(df, price, pen, K, C, n, ebar, -1.0, f0)
        r0, rl, rh = (v[NS:2 * NS + 1].sum() for v in (x0, xl, xh))
        pos.append((*key, float((r0 - rl) / (rh - rl)) if rh - rl > 1e-6 else np.nan, float((rh - rl) / ((NS + 1) * K))))
    return rows, pos


def task_b(d, ev_step, ev_e, y, price, tot, hmean, fcs, ebar):
    rows = []
    ee = np.minimum(ev_e, 1.6 * ebar)
    for par, pc in GRID_B:
        pen, K, C, n = setup(par, pc, tot, hmean, ebar)
        for m, f in fcs.items():
            dd = np.repeat(np.maximum(f, 0) / ebar / 4, 4)
            r = dict(day=d, cfg=str(par), pc=pc, model=m, mae=float(np.abs(y - f).mean()))
            x0, f0 = lp_bl(dd, price, pen, K, C, n, ebar)
            xl, _ = lp_bl(dd, price, pen, K, C, n, ebar, 1.0, f0)
            xh, _ = lp_bl(dd, price, pen, K, C, n, ebar, -1.0, f0)
            for tie, x in (("min", xl), ("mid", (xl + xh) / 2), ("max", xh), ("default", x0)):
                r[f"c_{tie}"] = run_day_bl(ev_step, ee, price, pen, K, C, pc, ebar, position(x, n))[0]
            y0, g0 = lp_bl(dd, price, pen, K, C, n + 2, ebar)
            yh, _ = lp_bl(dd, price, pen, K, C, n + 2, ebar, -1.0, g0)
            r["c_max2"] = run_day_bl(ev_step, ee, price, pen, K, C, pc, ebar, position(yh, n + 2))[0]
            rows.append(r)
    return rows


def ana():
    lines = []
    if (OUT / "bl_fc_pih.parquet").exists():
        FC = pd.concat([pd.read_parquet(OUT / "bl_fc.parquet")[["day", "cfg", "pc", "cand", "cost", "short"]], pd.read_parquet(OUT / "bl_fc_pih.parquet")])
        RU = pd.read_parquet(OUT / "bl_rule.parquet")
        PO = pd.read_parquet(OUT / "bl_pos.parquet")
        tab = []
        for (c, pc), g in RU.groupby(["cfg", "pc"], sort=False):
            Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
            h = FC[(FC.cfg == c) & (FC.pc == pc)]
            Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
            days = Fc.index[Fc.index >= START]
            k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
            ref = np.array([Mc.loc[d, pick(Mc, d, k0)] for d in days])
            out = {}
            for nm, pref, fb in (("fc_roll", ("mean@", "q"), "mean@0|mid"), ("pih_roll", ("pih@",), "pih@0|mid"), ("pi_roll", ("pi@",), "pi@0|mid")):
                cols = [k for k in Fc.columns if k.startswith(pref)]
                x = np.array([Fc.loc[d, pick(Fc[cols], d, fb)] for d in days])
                out[nm] = rel_ci(x, ref)
            for nm, col in (("pih_default", "pih@0|default"), ("pih_max", "pih@0|max"), ("pih_max2", "pih@2|max")):
                out[nm] = rel_ci(Fc.loc[days, col].to_numpy(), ref)
            p = PO[(PO.cfg == c) & (PO.pc == pc)]
            s = f"{c} {pc:.0f} kW：相对滚动规则 % " + " ".join(f"{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]" for k, v in out.items())
            s += f"；默认解在最优面上的位置 中位 {p.pos.median():.2f} 均值 {p.pos.mean():.2f}（四分位 {p.pos.quantile(.25):.2f}–{p.pos.quantile(.75):.2f}）；最优面宽度（满电块数占 K）中位 {p.width.median():.3f}"
            lines.append(s)
            tab.append(dict(cfg=c, pc=pc, pos_med=p.pos.median(), pos_mean=p.pos.mean(), width=p.width.median(), **{k: 100 * v[0] for k, v in out.items()},
                            **{k + "_lo": 100 * v[1] for k, v in out.items()}, **{k + "_hi": 100 * v[2] for k, v in out.items()}))
        pd.DataFrame(tab).to_csv(OUT / "exp_bl2_a.csv", index=False, encoding="utf-8-sig")
    if (OUT / "bl_models.parquet").exists():
        M = pd.read_parquet(OUT / "bl_models.parquet")
        for (c, pc), g in M.groupby(["cfg", "pc"], sort=False):
            dd = g.groupby("day")[["c_min", "c_mid", "c_max", "c_default", "c_max2"]].sum()
            s = f"[9 模型] {c} {pc:.0f} kW："
            for a, b in (("c_mid", "c_min"), ("c_max", "c_min"), ("c_mid", "c_default"), ("c_max", "c_default"), ("c_max2", "c_default")):
                m, lo, hi = rel_ci(dd[a], dd[b])
                per = g.groupby("model")[[a, b]].sum()
                pm = 100 * (per[a] / per[b] - 1)
                s += f" {a[2:]} vs {b[2:]} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]（逐模型 {pm.min():+.1f}…{pm.max():+.1f}，{int((pm < 0).sum())}/9 为负）；"
            mod = g.groupby("model")[["mae", "c_min", "c_mid", "c_max", "c_default", "c_max2"]].mean().sort_values("mae")
            lg = mod.loc["lgb_local"]
            s += "按 MAE 排序的各模型成本 / LightGBM（默认；最大）：" + "，".join(f"{k} {mod.loc[k, 'c_default'] / lg.c_default:.3f}；{mod.loc[k, 'c_max'] / lg.c_max:.3f}" for k in mod.index)
            s += f"；Kendall τ(MAE, 成本) 默认 {kendalltau(mod.mae, mod.c_default)[0]:.2f} 最大 {kendalltau(mod.mae, mod.c_max)[0]:.2f}"
            for nm, col in (("TimesFM vs LightGBM", "timesfm3_cov"), ("季节朴素 vs LightGBM", "naive")):
                for cc in ("c_default", "c_max"):
                    a = g[g.model == col].set_index("day")[cc]
                    b = g[g.model == "lgb_local"].set_index("day")[cc].reindex(a.index)
                    m, lo, hi = rel_ci(a.to_numpy(), b.to_numpy())
                    s += f"；{nm}（{cc[2:]}）{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            s += f"；季节朴素+最大 / LightGBM+默认 {mod.loc['naive', 'c_max'] / lg.c_default:.3f}，季节朴素+中点 / LightGBM+最小 {mod.loc['naive', 'c_mid'] / lg.c_min:.3f}"
            lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl2.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ana"
    if mode == "ana":
        ana()
        sys.exit()
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    pool = ["lgb_local"] if mode == "A" else MAIN_POOL
    F = {m: pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "mean"]) for m in pool}
    G = {m: F[m][F[m].series == S].set_index("t")["mean"] for m in pool}
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        fcs = {m: np.maximum(G[m].reindex(idx).to_numpy(), 0) for m in pool}
        if np.isfinite(y).all() and d in EV and all(np.isfinite(v).all() for v in fcs.values()):
            Tk.append((d, *EV[d], y, D.price(S, d.month), float(D.price("A7", d.month, "total").max()), hmean, fcs["lgb_local"] if mode == "A" else fcs, ebar))
    print(mode, len(Tk), "tasks", flush=True)
    if mode == "A":
        res = Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task_a)(*t) for t in Tk)
        pd.DataFrame([r for rr, _ in res for r in rr], columns=["day", "cfg", "pc", "cand", "cost", "short"]).to_parquet(OUT / "bl_fc_pih.parquet")
        pd.DataFrame([r for _, pp in res for r in pp], columns=["day", "cfg", "pc", "pos", "width"]).to_parquet(OUT / "bl_pos.parquet")
    else:
        res = Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task_b)(*t) for t in Tk)
        pd.DataFrame([r for rr in res for r in rr]).to_parquet(OUT / "bl_models.parquet")
    ana()
