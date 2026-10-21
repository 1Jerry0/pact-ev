"""电池级仿真：预测到底值不值——预测管线与不用预测的规则在同等条件下对照。
exp_packs.py 的结果显示：按 D−29…D−2 滚动选阈值的规则明显好于 1–3 月一次选定的阈值，也好于中点解。这里把对照做公平：
  规则类   ：谷段补满、平/峰段把库存位置补到 θ·K（66 组阈值）；静态（1–3 月选定）或滚动（D−29…D−2）选阈值；
  预测类   ：LightGBM 的 {均值, q0.6, q0.7, q0.8, q0.9} × {最小, 中点, 最大库存解} + 均值的求解器默认解（16 个候选）；
             固定用"均值 + 中点"（预先规定），或与规则同样按 D−29…D−2 的电池级成本滚动选候选；
  完全预测 ：真实需求 × {默认, 中点, 最大}，同样滚动选——日前点预测经这条管线能做到的上限；
  二者合并 ：82 个候选一起滚动选，看选中的是不是预测类。
配置：库容 {2,4,6,8} 小时 × 功率 {1.3,1.6,2.5,4.0} 倍均值（罚价 ×3，12 kW / L=2），外加罚价 ×10 与 25 kW / L=1 各两组。
比较样本：2024-05-01 起（预测类候选此前已有 ≥ 28 天历史）。用法：python exp_packs_value.py"""
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
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from joblib import Parallel, delayed
import cfg
import data as D
from exp_leadtime import lp, unpack
from exp_packs import run_day, positions, RAW, T, TH

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((c, p, 3), 12.0, 2) for c in (2, 4, 6, 8) for p in (1.3, 1.6, 2.5, 4.0)] + \
       [((4, 2.5, 10), 12.0, 2), ((6, 2.5, 10), 12.0, 2), ((4, 2.5, 3), 25.0, 1), ((6, 2.5, 3), 25.0, 1)]
QL = (0.6, 0.7, 0.8, 0.9)
START = pd.Timestamp("2024-05-01")


def task(d, ev_step, ev_e, y, price, tot, hmean, f, Q, ebar, grid=None):
    rule, rows = [], []
    cls = np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6))
    ee = np.minimum(ev_e, 1.6 * ebar)
    for par, pc, L in (grid or GRID):
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(T)))
        K = max(int(round(cap / ebar)), 2)
        key = (d, str(par), pc, L)
        rc = [run_day(ev_step, ee, price, pen, cap, pmax, pc, ebar, np.where(cls == 0, K, np.where(cls == 1, a * K, b * K)))[:2] for a, b in TH]
        cf = run_day(ev_step, ee, price, pen, cap, pmax, pc, ebar, np.full(T, K))[:2]
        rule.append((*key, [c for c, _ in rc], [u for _, u in rc], cf[0], cf[1], float(ee.sum())))
        if f is None:
            continue
        levels = {"mean": f, **({} if Q is None else {f"q{q}": np.maximum(np.array([np.interp(q, cfg.QS, row) for row in Q]), 0) for q in QL}), "pi": y}
        for nm, fc in levels.items():
            x0, f0 = lp(fc, price, pen, cap, pmax, L)
            xl, _ = lp(fc, price, pen, cap, pmax, L, 1.0, f0)
            xh, _ = lp(fc, price, pen, cap, pmax, L, -1.0, f0)
            ties = [("min", xl), ("mid", (xl + xh) / 2), ("max", xh)] + ([("default", x0)] if nm in ("mean", "pi") else [])
            for tie, x in ties:
                c, u, _ = run_day(ev_step, ee, price, pen, cap, pmax, pc, ebar, positions(unpack(x), fc, L) / ebar)
                rows.append((*key, f"{nm}|{tie}", c, u))
    return rule, rows


def pick(M, d, fallback):
    past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
    return past.sum().idxmin() if len(past) >= 7 else fallback


def rel_ci(a, b):
    """(Σa − Σb) / Σb 及其平稳自助区间。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = StationaryBootstrap(max(1.0, float(optimal_block_length(a - b)["stationary"].iloc[0])), np.arange(len(a)), seed=cfg.SEED)
    r = np.array([a[i[0][0]].sum() / b[i[0][0]].sum() - 1 for i in bs.bootstrap(2000)])
    return a.sum() / b.sum() - 1, np.quantile(r, 0.025), np.quantile(r, 0.975)


def analyse(suf=""):
    RU = pd.read_parquet(OUT / f"bss_packs_value_rule{suf}.parquet")
    FC = pd.read_parquet(OUT / f"bss_packs_value_fc{suf}.parquet")
    lines, tab = [], []
    for (c, pc, L), g in RU.groupby(["cfg", "pc", "L"], sort=False):
        Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
        Ms = g.pivot(index="day", columns="th", values="short").sort_index()
        base = g.drop_duplicates("day").set_index("day")[["full", "short_full", "dem"]]
        h = FC[(FC.cfg == c) & (FC.pc == pc) & (FC.L == L)]
        Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
        Fs = h.pivot(index="day", columns="cand", values="short").sort_index()
        lgb = [k for k in Fc.columns if not k.startswith("pi|")]
        pis = [k for k in Fc.columns if k.startswith("pi|")]
        days = Fc.index[Fc.index >= START]
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        ALL = pd.concat([Mc.loc[Fc.index].rename(columns=lambda k: f"rule{k}"), Fc[lgb]], axis=1)
        ALLs = pd.concat([Ms.loc[Fc.index].rename(columns=lambda k: f"rule{k}"), Fs[lgb]], axis=1)
        pol, sh, chosen = {}, {}, []
        def run(name, M, Sx, cols, fb):
            ks = [pick(M[cols], d, fb) for d in days]
            pol[name] = np.array([M.loc[d, k] for d, k in zip(days, ks)])
            sh[name] = np.array([Sx.loc[d, k] for d, k in zip(days, ks)])
            return ks
        run("rule_roll", Mc, Ms, list(Mc.columns), k0)
        fk = run("fc_roll", Fc, Fs, lgb, "mean|mid")
        run("pi_roll", Fc, Fs, pis, "pi|mid")
        ak = run("all_roll", ALL, ALLs, list(ALL.columns), "mean|mid")
        for nm, col in (("rule_static", None), ("fc_default", "mean|default"), ("fc_mid", "mean|mid"), ("fc_max", "mean|max"), ("pi_plan", "pi|default")):
            pol[nm] = (Mc.loc[days, k0] if col is None else Fc.loc[days, col]).to_numpy()
            sh[nm] = (Ms.loc[days, k0] if col is None else Fs.loc[days, col]).to_numpy()
        pol["full"], sh["full"] = base.loc[days, "full"].to_numpy(), base.loc[days, "short_full"].to_numpy()
        pol["rule_hind"], sh["rule_hind"] = Mc.loc[days].min(axis=1).to_numpy(), np.zeros(len(days))
        dem = base.loc[days, "dem"].sum()
        ref = pol["rule_roll"]
        s = f"{c} p_c={pc:.0f}kW L={L}（{len(days)} 天）：滚动规则日均成本 {ref.mean():.0f}；相对滚动规则的成本差（%）"
        row = dict(cfg=c, pc=pc, L=L, cost_rule=ref.mean())
        for nm in ("rule_static", "full", "fc_default", "fc_mid", "fc_max", "fc_roll", "all_roll", "pi_plan", "pi_roll", "rule_hind"):
            m, lo, hi = rel_ci(pol[nm], ref)
            s += f" {nm} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            row[nm], row[nm + "_lo"], row[nm + "_hi"] = 100 * m, 100 * lo, 100 * hi
        m, lo, hi = rel_ci(pol["fc_mid"], pol["fc_default"])
        s += f"；中点 vs 默认 {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        row["mid_vs_def"], row["mid_vs_def_lo"], row["mid_vs_def_hi"] = 100 * m, 100 * lo, 100 * hi
        s += "；未满足 % " + "，".join(f"{nm} {100 * sh[nm].sum() / dem:.2f}" for nm in ("rule_roll", "fc_default", "fc_mid", "fc_max", "fc_roll", "pi_plan", "full"))
        row.update({f"short_{nm}": 100 * sh[nm].sum() / dem for nm in ("rule_roll", "fc_default", "fc_mid", "fc_max", "fc_roll", "pi_plan", "full")})
        s += f"；合并滚动选中预测类的天数占比 {np.mean([not str(k).startswith('rule') for k in ak]):.2f}；预测类滚动常选 {dict(pd.Series(fk).value_counts().head(3))}"
        row["share_fc"] = float(np.mean([not str(k).startswith("rule") for k in ak]))
        lines.append(s)
        tab.append(row)
    txt = "\n".join(lines)
    (OUT / f"exp_packs_value{suf}.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(tab).to_csv(OUT / f"exp_packs_value{suf}.csv", index=False, encoding="utf-8-sig")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
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
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    Tk = []
    for d in pd.date_range("2024-01-15", cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all() or d not in EV:
            continue
        f = Q = None
        if d >= cfg.MP_EVAL_START:
            f = np.maximum(FQ["mean"].reindex(idx).to_numpy(), 0)
            Q = FQ[cfg.QC].reindex(idx).to_numpy()
            if not (np.isfinite(f).all() and np.isfinite(Q).all()):
                continue
        Tk.append((d, *EV[d], y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, f, Q, ebar))
    if len(sys.argv) > 1:
        Tk = [t for t in Tk if t[7] is not None][:int(sys.argv[1])]
    print(len(Tk), "tasks; mean swap energy", round(ebar, 2), "; hmean", round(hmean, 1), flush=True)
    res = Parallel(n_jobs=7, batch_size=1, verbose=2)(delayed(task)(*t) for t in Tk)
    RU = pd.DataFrame([(d, c, pc, L, i, cc[i], uu[i], cf, uf, dem) for b, _ in res for (d, c, pc, L, cc, uu, cf, uf, dem) in b for i in range(len(TH))],
                      columns=["day", "cfg", "pc", "L", "th", "cost", "short", "full", "short_full", "dem"])
    RU.to_parquet(OUT / "bss_packs_value_rule.parquet")
    FC = pd.DataFrame([r for _, rr in res for r in rr], columns=["day", "cfg", "pc", "L", "cand", "cost", "short"])
    FC.to_parquet(OUT / "bss_packs_value_fc.parquet")
    if len(sys.argv) == 1:
        analyse()
