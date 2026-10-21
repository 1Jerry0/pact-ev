"""储能站成本口径重算的分析：超限负荷按削减充电、损失服务费计价（exp_cost_basis_st.py）。
对每站：中点 / 默认计划下“MAE 最优≠遗憾最低”的情景比例、Kendall τ 中位数、TimesFM-3.0 / LightGBM 遗憾之比及平稳自助区间、削减电量占比。
输出：结果/cb_storage_summary.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(HERE.parent))
from exp_packs_value import rel_ci  # noqa: E402

OUT = ROOT / "results"
X = pd.read_parquet(OUT / "cb_storage_sched.parquet")
X["reg"] = X.E + X.Pen - X.cost_opt
lines = []


def say(s):
    lines.append(s)
    print(s)


def fmt(a, b):
    m, lo, hi = rel_ci(a, b)
    return f"{1 + m:.2f}{'*' if lo > 0 or hi < 0 else ''}"


for tie in ("mid", "default"):
    say(f"## {tie} 计划")
    hi = lo = 0
    for st, g in X[X.tie == tie].groupby("station"):
        bd, tau = [], []
        for c, h in g.groupby("cfg"):
            a = h.groupby("model").agg(mae=("mae", "mean"), reg=("reg", "sum"))
            bd.append(a.mae.idxmin() != a.reg.idxmin())
            tau.append(kendalltau(a.mae, a.reg)[0])
        d = g.pivot_table(index="day", columns="model", values="reg", aggfunc="sum").sort_index()
        m, l, u = rel_ci(d["timesfm3_cov"], d["lgb_local"])
        hi += l > 0
        lo += u < 0
        sh = g.drop_duplicates(["day", "cfg"])
        say(f"  {st}：最准≠最省 {100 * np.mean(bd):.0f}%，τ {np.median(tau):.2f}，TimesFM/LightGBM {1 + m:.2f} [{1 + l:.2f}, {1 + u:.2f}]，"
            f"LQR/LightGBM {fmt(d['lqr'], d['lgb_local'])}，季节经验分位数/LightGBM {fmt(d['seasonal_q'], d['lgb_local'])}；"
            f"完全信息下削减电量占比 {100 * sh.X_opt.sum() / sh.dem.sum():.3f}%")
    say(f"  TimesFM 显著更贵 {hi} 站，显著更省 {lo} 站")
(OUT / "cb_storage_summary.txt").write_text("\n".join(lines), encoding="utf-8")
