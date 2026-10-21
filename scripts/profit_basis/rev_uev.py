"""修订实验 4：UrbanEV 275 个区的需求套用换电库存模型（风格化复现），锚定功率：库容 4/6/8 小时 × 功率 2.5 × 罚价 3。
5 个模型的默认 / 最小 / 中点 / 最大库存解，以及不用预测的阈值规则（阈值在 9–10 月开发期逐区逐组合选定）。
任务粒度：区。用法：python rev_uev.py"""
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
from analysis import boot_mean
from exp_bss import mix, rule_costs, TH

OUT = HERE.parents[2] / "results" / "profit_basis"
MODELS = ["naive", "lgb_global", "chronos2_cov", "timesfm3_cov", "combo5"]
GRID = [(c, 2.5, 3) for c in (4, 6, 8)]


def task(z, hmean, dev, ev):
    """dev: [(y, price, tot)]；ev: [(day, y, price, tot, {model: f})]。"""
    devc = {str(par): np.zeros(len(TH)) for par in GRID}
    for y, price, tot in dev:
        for par in GRID:
            pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
            devc[str(par)] += rule_costs(y, price, pen, par[0] * hmean, par[1] * hmean)[0]
    best = {k: int(np.argmin(v)) for k, v in devc.items()}
    rows = []
    for d, y, price, tot, fcs in ev:
        for par in GRID:
            P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
            rc, rs = rule_costs(y, price, P.pen, P.cap, P.pmax)
            b = best[str(par)]
            for m, f in fcs.items():
                lo, hi = P.plan(f, "min"), P.plan(f, "max")
                rows.append((z, d, par[0], m, P.opt, P.cost(P.plan(f, "default")) - P.opt, P.cost(lo) - P.opt, P.cost(mix(lo, hi, 0.5)) - P.opt,
                             P.cost(hi) - P.opt, rc[b] - P.opt, rc[-1] - P.opt))
    return rows


if __name__ == "__main__":
    V = D.uev_volume()
    F = {m: pd.read_parquet(cfg.OUT / "fc" / f"uev_{m}.parquet", columns=["series", "t", "mean"]).pivot(index="t", columns="series", values="mean") for m in MODELS}
    dev_days = pd.date_range("2022-09-01", cfg.UEV_DEV_END.normalize(), freq="D")
    ev_days = D.days(cfg.UEV_EVAL_START, cfg.UEV_FC_END)
    PR = {mo: (D.price("A1", mo), HK(D.price("A1", mo, "total"))) for mo in (9, 10, 11, 12, 1, 2)}
    T = []
    zones = list(V.columns)
    if len(sys.argv) > 1:
        zones = zones[:int(sys.argv[1])]
    for z in zones:
        hmean = float(V[z].loc[:cfg.UEV_DEV_END].mean())
        if hmean <= 0:
            continue
        dev, ev = [], []
        for d in dev_days:
            y = V[z].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
            if np.isfinite(y).all():
                dev.append((y, *PR[d.month]))
        for d in ev_days:
            idx = pd.date_range(d, periods=24, freq="h")
            y = V[z].reindex(idx).to_numpy()
            fcs = {m: np.maximum(F[m][z].reindex(idx).to_numpy(), 0) for m in MODELS}
            if np.isfinite(y).all() and all(np.isfinite(v).all() for v in fcs.values()):
                ev.append((d, y, *PR[d.month], fcs))
        T.append((z, hmean, dev, ev))
    print(len(T), "zones", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 6)), batch_size=1, verbose=2)(delayed(task)(*t) for t in T)
    R = pd.DataFrame([r for rr in res for r in rr], columns=["zone", "day", "cap", "model", "opt", "reg_default", "reg_min", "reg_mid", "reg_max", "rule_static", "rule_full"])
    R.to_parquet(OUT / "rev_uev.parquet")
    dd = R.groupby("day")[["reg_default", "reg_min", "reg_mid", "reg_max"]].sum()
    lines = [f"UrbanEV 换电（锚定功率 2.5）：{R.zone.nunique()} 区，{dd.shape[0]} 天"]
    for base in ("reg_default", "reg_min"):
        m, ci, _ = boot_mean(1 - dd.reg_mid / dd[base])
        lines.append(f"中点 vs {base[4:]}: {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]；中点更好的天数比例 {(dd.reg_mid < dd[base]).mean():.3f}")
    L = R[R.model == "lgb_global"]
    tot = L[["reg_default", "reg_mid", "reg_max", "rule_static", "rule_full"]].sum()
    lines.append("LightGBM（全局）总遗憾 / 默认解：" + (tot / tot.reg_default).round(3).to_dict().__repr__())
    zz = L.groupby("zone")[["reg_mid", "rule_static", "reg_default"]].sum()
    lines.append(f"中点解优于规则的区占比 {(zz.reg_mid < zz.rule_static).mean():.3f}；中点解优于默认解的区占比 {(zz.reg_mid < zz.reg_default).mean():.3f}")
    mod = R.groupby("model")[["reg_default", "reg_mid"]].mean()
    lines.append("各模型日均遗憾：\n" + mod.round(2).to_string())
    txt = "\n".join(lines)
    (OUT / "rev_uev.txt").write_text(txt, encoding="utf-8")
    print(txt)
