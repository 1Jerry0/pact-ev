"""跨日连续的电池级仿真里的"精度与价值"：九个预测模型的均值预测各自做计划（求解器默认解、最大库存解、最大库存解 + 1 步裕量），
连续仿真全程（exp_bl_carry.run_policy），比较成本与精度的排序。回答：每日重启口径下"精度不决定排序"的结论在跨日连续口径下是否成立。
配置：6 小时库存、2.5 倍功率、罚价 ×3，单块 8/12/18/25/50 kW；评估 5 月 1 日起 245 天。用法：python exp_bl_carry_models.py"""
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
from scipy.stats import kendalltau
from joblib import Parallel, delayed
import cfg
import data as D
from analysis import MAIN_POOL
from exp_bl import load_events
from exp_bl_carry import run_policy, GRID
from exp_packs_value import rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
PLANS = ["mean@0|default", "mean@0|max", "mean@1|max"]

if __name__ == "__main__":
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = {}
    for m in MAIN_POOL:
        X = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "mean"])
        F[m] = X[X.series == S].set_index("t")["mean"]
    days, PR = [], {}
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        if d in EV and np.isfinite(H[S].reindex(idx).to_numpy()).all() and all(np.isfinite(F[m].reindex(idx).to_numpy()).all() for m in MAIN_POOL):
            days.append(d)
            PR[d] = (D.price(S, d.month), HK(D.price("A7", d.month, "total")))
    FCS = {m: {d: {"mean": F[m].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()} for d in days} for m in MAIN_POOL}
    ev = np.array([d >= START for d in days])
    yy = np.concatenate([H[S].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy() for d in days if d >= START])
    mae = {m: float(np.abs(np.concatenate([FCS[m][d]["mean"] for d in days if d >= START]) - yy).mean()) for m in MAIN_POOL}
    keys = [(par, pc, m, p) for par, pc in GRID for m in MAIN_POOL for p in PLANS]
    res = Parallel(n_jobs=int(os.environ.get("NJ", 14)), batch_size=1, verbose=1)(delayed(run_policy)(par, pc, days, EV, FCS[m], PR, hmean, ebar, p) for par, pc, m, p in keys)
    R = {k: r[ev, 0] for k, r in zip(keys, res)}
    lines = [f"跨日连续口径，九个模型的均值预测，{int(ev.sum())} 天；nMAE：" + "，".join(f"{m} {mae[m] / yy.mean():.3f}" for m in MAIN_POOL)]
    rows = []
    for par, pc in GRID:
        for p in PLANS:
            cost = {m: R[(par, pc, m, p)].sum() for m in MAIN_POOL}
            tau = kendalltau([mae[m] for m in MAIN_POOL], [cost[m] for m in MAIN_POOL]).statistic
            best = min(cost, key=cost.get)
            s = f"{par} {pc:.0f} kW {p}：成本最低 {best}；Kendall τ(MAE, 成本) {tau:+.2f}；相对 LightGBM %："
            for m in ("timesfm3_cov", "chronos2_cov", "combo5", "lqr", "naive", "seasonal_q"):
                mm, lo, hi = rel_ci(R[(par, pc, m, p)], R[(par, pc, "lgb_local", p)])
                s += f" {m} {100 * mm:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
                rows.append(dict(cfg=str(par), pc=pc, plan=p, model=m, rel=100 * mm, lo=100 * lo, hi=100 * hi, tau=tau, best=best))
            lines.append(s)
        pooled_d = np.sum([R[(par, pc, m, "mean@0|default")] for m in MAIN_POOL], axis=0)
        pooled_x = np.sum([R[(par, pc, m, "mean@0|max")] for m in MAIN_POOL], axis=0)
        mm, lo, hi = rel_ci(pooled_x, pooled_d)
        nb = sum(R[(par, pc, m, "mean@0|max")].sum() < R[(par, pc, m, "mean@0|default")].sum() for m in MAIN_POOL)
        mm2, lo2, hi2 = rel_ci(R[(par, pc, "naive", "mean@0|max")], R[(par, pc, "lgb_local", "mean@0|default")])
        lines.append(f"  九模型合并 最大库存解 vs 默认解 {100 * mm:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]，{nb}/9 个模型最大库存解更便宜；季节朴素 + 最大库存解 vs LightGBM + 默认解 {100 * mm2:+.1f} [{100 * lo2:+.1f}, {100 * hi2:+.1f}]")
    txt = "\n".join(lines)
    (OUT / "exp_bl_carry_models.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "exp_bl_carry_models.csv", index=False, encoding="utf-8-sig")
    print(txt)
