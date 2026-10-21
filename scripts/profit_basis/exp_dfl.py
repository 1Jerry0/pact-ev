"""决策导向的预测对照（小时模型，A7 锚定组合、罚价 ×3 的 6 组）：把"误差价格"从诊断量变成处方。
带方向权重的绝对损失  Σ_t (π^u_t e_t^+ + π^o_t e_t^-)  的最优点预测是逐小时分位数 τ_t = π^u_t / (π^u_t + π^o_t)
（与 Shah 等 2022 的 DirectedWeighted 局部决策损失同型，但权重来自管线本身而不是抽样回归）。这里用事前价格
（D−29…D−2 的平均，发布时刻可得）取 LightGBM 校准分位数里的 τ_t 分位作为计划用的预测，并加一个步长 λ：τ_t(λ) = 0.5 + λ(τ_t − 0.5)。
对照：均值预测；全天同一分位数 q ∈ {0.6,…,0.9}；二者都按 D−29…D−2 的遗憾滚动选参数（q 或 λ），公平比较；
另与已有的情景优化、滚动阈值规则并列。计划选择规则：默认解与中点解各做一遍（价格分别用对应规则下的）。用法：python exp_dfl.py"""
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
from exp_packs_value import pick, rel_ci

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
MULT = int(os.environ.get("MULT", 3))
GRID = [(c, p, MULT) for c in (4, 6, 8) for p in (2.5, 4.0)]
QU = (0.6, 0.7, 0.8, 0.9)
LAM = (0.25, 0.5, 0.75, 1.0)
QS = np.array(cfg.QS)


def task(d, y, price, tot, hmean, f, Q, taus):
    rows = []
    for par in GRID:
        P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
        for tie in ("default", "mid"):
            tau = taus[(str(par), tie)]
            cands = {"mean": f, **{f"q{q}": np.array([np.interp(q, QS, r) for r in Q]) for q in QU}}
            for lam in LAM:
                tl = np.clip(0.5 + lam * (tau - 0.5), QS[0], QS[-1])
                cands[f"p{lam}"] = np.array([np.interp(tl[t], QS, Q[t]) for t in range(24)])
            for nm, fc in cands.items():
                rows.append((d, str(par), tie, nm, P.regret(np.maximum(fc, 0), tie), P.opt, float(np.maximum(fc, 0).sum() / max(y.sum(), 1e-9))))
    return rows


def analyse():
    R = pd.read_parquet(OUT / f"dfl_x{MULT}.parquet")
    lines = []
    base = R[(R.tie == "default") & (R.cand == "mean")].groupby("day").regret.sum()
    lines.append(f"罚价 ×{MULT}，{R.day.nunique()} 天 × {R.cfg.nunique()} 组；总遗憾相对 均值预测 + 默认解（= 1）；遗憾 / 完全信息成本 = {base.sum() / R[(R.tie == 'default') & (R.cand == 'mean')].opt.sum():.3f}")
    res = {}
    for tie, g in R.groupby("tie"):
        M = g.groupby(["day", "cand"]).regret.sum().unstack("cand")
        tot = (M.sum() / base.sum()).round(3)
        days = M.index[28:]
        uni, pri = ["mean"] + [f"q{q}" for q in QU], ["mean"] + [f"p{l}" for l in LAM]
        sel = {}
        for nm, cols in (("tuned_uniform", uni), ("tuned_price", pri), ("tuned_both", sorted(set(uni + pri)))):
            ks = [pick(M[cols], d, "mean") for d in days]
            sel[nm] = np.array([M.loc[d, k] for d, k in zip(days, ks)])
            sel[nm + "_k"] = pd.Series(ks).value_counts().head(3).to_dict()
        b = base.loc[days]
        s = f"[{tie}] 固定候选：" + "，".join(f"{k} {v:.3f}" for k, v in tot.items()) + f"；从第 29 天起（{len(days)} 天）：均值 {M.loc[days, 'mean'].sum() / b.sum():.3f}"
        for nm in ("tuned_uniform", "tuned_price", "tuned_both"):
            s += f"，{nm} {sel[nm].sum() / b.sum():.3f}（常选 {sel[nm + '_k']}）"
        m, lo, hi = rel_ci(sel["tuned_price"], sel["tuned_uniform"])
        s += f"；滚动选 λ 的价格分位 vs 滚动选 q 的统一分位：{100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        bq = min(uni, key=lambda k: M.loc[days, k].sum())
        bp = min(pri, key=lambda k: M.loc[days, k].sum())
        m, lo, hi = rel_ci(M.loc[days, bp].to_numpy(), M.loc[days, bq].to_numpy())
        s += f"；事后最好的固定 λ（{bp}）vs 事后最好的固定 q（{bq}）：{100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        lines.append(s)
        res[tie] = (M, sel, days)
    lines.append("预测总量 / 实际总量（均值）：" + "，".join(f"{t}|{k} {v:.3f}" for (t, k), v in R.groupby(["tie", "cand"]).ratio.mean().items() if k in ("mean", "q0.7", "q0.9", "p0.5", "p1.0")))
    txt = "\n".join(lines)
    (OUT / f"exp_dfl_x{MULT}.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    fp = OUT / ("rev_prices_a7.pkl" if MULT == 3 else f"rev_prices_a7_x{MULT}.pkl")
    if not fp.exists():                                          # 该罚价档的误差价格还没算过：逐日逐组合做有限差分（同 rev_dswe.py）
        H0 = D.mp_hourly()
        hm0 = float(H0[S].loc[:cfg.MP_DEV_END].mean())

        def ptask(d, y, price, tot):
            out = []
            for par in GRID:
                P = sim.Problem("bss", y, price, par, dict(hmean=hm0, tot_max=tot))
                out.append((d, str(par), *P.dswe_weights("current", "mid"), *P.dswe_weights("current", "default")))
            return out
        PT = []
        for d in pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D"):
            y = H0[S].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
            if np.isfinite(y).all():
                PT.append((d, y, D.price(S, d.month), HK(D.price("A7", d.month, "total"))))
        pr = Parallel(n_jobs=int(os.environ.get("NJ", 7)), batch_size=2, verbose=1)(delayed(ptask)(*t) for t in PT)
        pd.DataFrame([r for rr in pr for r in rr], columns=["day", "cfg", "wu", "wo", "wu0", "wo0"]).to_pickle(fp)
    W = pd.read_pickle(fp)
    Wd = {(r.day, r.cfg): (r.wu, r.wo, r.wu0, r.wo0) for r in W.itertuples(index=False)}
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        f, Q = FQ["mean"].reindex(idx).to_numpy(), FQ[cfg.QC].reindex(idx).to_numpy()
        if not (np.isfinite(y).all() and np.isfinite(f).all() and np.isfinite(Q).all()):
            continue
        taus, ok = {}, True
        for par in GRID:
            past = [Wd[(d - pd.Timedelta(days=k), str(par))] for k in range(2, 30) if (d - pd.Timedelta(days=k), str(par)) in Wd]
            if len(past) < 14:
                ok = False
                break
            for tie, (iu, io) in (("mid", (0, 1)), ("default", (2, 3))):
                wu, wo = np.mean([p[iu] for p in past], axis=0), np.mean([p[io] for p in past], axis=0)
                wu, wo = np.maximum(wu, 0), np.maximum(wo, 0)
                taus[(str(par), tie)] = np.where(wu + wo > 1e-9, wu / np.maximum(wu + wo, 1e-12), 0.5)
        if ok:
            Tk.append((d, y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, np.maximum(f, 0), Q, taus))
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 7)), batch_size=2, verbose=1)(delayed(task)(*t) for t in Tk)
    pd.DataFrame([r for rr in res for r in rr], columns=["day", "cfg", "tie", "cand", "regret", "opt", "ratio"]).to_parquet(OUT / f"dfl_x{MULT}.parquet")
    analyse()
