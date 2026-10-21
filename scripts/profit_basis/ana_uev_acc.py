"""深圳 60 区：预测管线相对规则的收益与该区日前预测的绝对精度（nMAE）、规模、变异系数的关系；A7 逐月同样看一遍。
问题：模型之间精度不决定排序，但"一个站值不值得用预测"是否由该站预测的绝对精度决定？用法：python ana_uev_acc.py"""
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(HERE.parent))
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import cfg
import data as D

OUT = HERE.parents[2] / "results" / "profit_basis"
V = D.uev_volume()
F = pd.read_parquet(cfg.OUT / "fc" / "uev_lgb_global_cal.parquet", columns=["series", "t", "mean"])
F["series"] = F.series.astype(str)
old = pd.read_csv(OUT / "exp_bl_uev.csv")
old["zone"] = old.zone.astype(str)
car = pd.read_csv(OUT / "exp_bl_carry_uev.csv")
car["zone"] = car.tag.astype(str)
acc = {}
for z in old.zone.unique():
    f = F[F.series == z].set_index("t")["mean"]
    y = V[z].reindex(f.index)
    m = (f.index >= cfg.UEV_EVAL_START) & y.notna().to_numpy()
    e = (y[m] - f[m])
    acc[z] = dict(nmae=float(e.abs().mean() / y[m].mean()), bias=float(-e.mean() / y[m].mean()))
A = pd.DataFrame(acc).T
lines = []
for pc in (12.0, 25.0):
    g = old[old.pc == pc].set_index("zone").join(A)
    g["gain"] = 100 * (g.tuned / g.rule - 1)
    c = car[car.pc == pc].set_index("zone")
    g["gain_c"] = 100 * (c.fc_roll / c.rule_roll - 1)
    s = f"{pc:.0f} kW：nMAE 中位 {g.nmae.median():.2f}（四分位 {g.nmae.quantile(.25):.2f}–{g.nmae.quantile(.75):.2f}）；"
    for col, lab in (("gain", "每日重启"), ("gain_c", "跨日连续")):
        r1, p1 = spearmanr(g.nmae, g[col])
        r2, p2 = spearmanr(np.log(g.hmean), g[col])
        r3, p3 = spearmanr(g.cv, g[col])
        X = sm.add_constant(pd.DataFrame({"nmae": g.nmae, "log_size": np.log(g.hmean), "cv": g.cv}))
        fit = sm.OLS(g[col], X).fit(cov_type="HC1")
        lo = g.nmae <= g.nmae.median()
        s += (f"[{lab}] ρ(nMAE, 收益) {r1:+.2f}（p={p1:.3f}），ρ(规模) {r2:+.2f}，ρ(CV) {r3:+.2f}；三者同时回归：nMAE {fit.params.nmae:+.1f}（p={fit.pvalues.nmae:.3f}），"
              f"log 规模 {fit.params.log_size:+.2f}（p={fit.pvalues.log_size:.2f}），CV {fit.params.cv:+.1f}（p={fit.pvalues.cv:.2f}），R² {fit.rsquared:.2f}；"
              f"nMAE 低于中位的 30 区收益中位 {g.loc[lo, col].median():+.1f}%（管线更便宜 {100 * (g.loc[lo, col] < 0).mean():.0f}%），高于中位的 {g.loc[~lo, col].median():+.1f}%（{100 * (g.loc[~lo, col] < 0).mean():.0f}%）；")
    r, p = spearmanr(g.nmae, np.log(g.hmean))
    s += f"ρ(nMAE, 规模) {r:+.2f}"
    lines.append(s)
# A7 逐月
T = pd.read_csv(OUT / "ana_split_month.csv") if (OUT / "ana_split_month.csv").exists() else None
txt = "\n".join(lines)
(OUT / "ana_uev_acc.txt").write_text(txt, encoding="utf-8")
print(txt)
