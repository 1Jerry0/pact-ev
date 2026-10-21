"""修订实验 3：第二座真实换电站 A8（换电次数，2024-04 至 10 月）在锚定网格上的复现：
9 个模型的默认 / 最小 / 中点 / 最大 / 内点法 / 削峰对照，以及不用预测的阈值规则。用法：python rev_a8.py"""
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
from exp_bss import mix, MAIN_POOL, GRIDS, rule_costs, plan_ipm, TH
from exp_flat import plan_flat

OUT = HERE.parents[2] / "results" / "profit_basis"
ST = "A8"
GRID = GRIDS["anchor"]


def task(d, y, price, tot, hmean, fcs):
    rows, rule = [], []
    for par in GRID:
        P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
        rc, rs = rule_costs(y, price, P.pen, P.cap, P.pmax)
        rule.append((d, str(par), rc - P.opt, rs))
        for m, f in fcs.items():
            lo, hi = P.plan(f, "min"), P.plan(f, "max")
            smin, smax = lo[2].sum(), hi[2].sum()
            pd_ = P.plan(f, "default")
            rows.append(dict(day=d, cfg=str(par), model=m, opt=P.opt, dem=y.sum(), reg_default=P.cost(pd_) - P.opt, reg_min=P.cost(lo) - P.opt,
                             reg_mid=P.cost(mix(lo, hi, 0.5)) - P.opt, reg_max=P.cost(hi) - P.opt, reg_ipm=P.cost(plan_ipm(P, f)) - P.opt,
                             reg_flat=P.cost(plan_flat(P, f)) - P.opt, width=(smax - smin) / (25 * P.cap),
                             pos_default=(pd_[2].sum() - smin) / (smax - smin) if smax - smin > 1e-6 else np.nan))
    return rows, rule


if __name__ == "__main__":
    H = D.mp_hourly()
    hmean = float(H[ST].loc[:cfg.MP_DEV_END].mean())
    F = {m: pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "mean"]) for m in MAIN_POOL}
    G = {m: F[m][F[m].series == ST].set_index("t")["mean"] for m in MAIN_POOL}
    T = []
    for d in pd.date_range("2024-02-01", "2024-10-31", freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[ST].reindex(idx).to_numpy()
        if not np.isfinite(y).all():
            continue
        fcs = {}
        if d >= cfg.MP_EVAL_START:
            fcs = {m: np.maximum(G[m].reindex(idx).to_numpy(), 0) for m in MAIN_POOL}
            if not all(np.isfinite(v).all() for v in fcs.values()):
                fcs = {}
        T.append((d, y, D.price(ST, d.month), HK(D.price(ST, d.month, "total")), hmean, fcs))
    print(len(T), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 6)), batch_size=2, verbose=2)(delayed(task)(*t) for t in T)
    R = pd.DataFrame([r for a, _ in res for r in a])
    R.to_parquet(OUT / "rev_a8_plans.parquet")
    rule = [(d, c, i, float(reg[i]), float(sh[i])) for _, b in res for (d, c, reg, sh) in b for i in range(len(TH))]
    pd.DataFrame(rule, columns=["day", "cfg", "th", "regret", "short"]).to_parquet(OUT / "bss_a8_rule.parquet")
    from ana_bss import rule_series
    RU = rule_series("a8")
    dd = R.groupby("day")[["reg_default", "reg_min", "reg_mid", "reg_max", "reg_ipm", "reg_flat"]].sum()
    lines = [f"A8 锚定网格：{dd.shape[0]} 天；默认解位置均值 {R.pos_default.mean():.3f}；最优面宽度中位数 {R.width.median():.3f}"]
    for base in ("reg_default", "reg_min", "reg_ipm", "reg_flat"):
        m, ci, _ = boot_mean(1 - dd.reg_mid / dd[base])
        lines.append(f"中点 vs {base[4:]}: {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]；中点更好的天数比例 {(dd.reg_mid < dd[base]).mean():.3f}")
    L = R[R.model == "lgb_local"].merge(RU, on=["day", "cfg"])
    tot = L[["reg_default", "reg_mid", "reg_max", "rule_static", "rule_roll", "rule_full"]].sum()
    lines.append("LightGBM 总遗憾 / 默认解：" + (tot / tot.reg_default).round(3).to_dict().__repr__())
    L["mult"] = L.cfg.map(lambda s: eval(s)[2])
    g = L.groupby("mult")[["reg_default", "reg_mid", "reg_max", "rule_roll"]].sum()
    lines.append("按罚价（/默认解）：\n" + g.div(g.reg_default, axis=0).round(3).to_string())
    txt = "\n".join(lines)
    (OUT / "rev_a8.txt").write_text(txt, encoding="utf-8")
    print(txt)
