"""安全库存对照的汇总（对照方案在运行前写定，见 PAPER_MAP.md）。
滚动调参：每个（配置，模型，选解）在目标日 d 选 (d−30, d−2] 内平均遗憾最低的 k（≥ 7 天历史）；事后最优固定 k 作上界参考。
统计量与表 3 相同：日度遗憾降幅 1 − Σ甲 / Σ乙（对模型与配置求和后逐日计算，取均值），平稳块自助法 95% 置信区间。
评价窗口：2024-05-01 起。输出：results/safety_stock_summary.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
from analysis import boot_mean  # noqa: E402

OUT = ROOT / "results"
START = pd.Timestamp("2024-05-01")
R = pd.read_parquet(OUT / "safety_stock_A7.parquet")
KS = sorted(R.k.unique())
key = ["cfg", "model", "tie"]
W = R.pivot_table(index=key + ["day"], columns="k", values="reg").sort_index()
SH = R.pivot_table(index=key + ["day"], columns="k", values="S").sort_index()
EN = R.pivot_table(index=key + ["day"], columns="k", values="E").sort_index()

pick, reg_t, sh_t, en_t = {}, {}, {}, {}
for kk, g in W.groupby(level=[0, 1, 2]):
    g = g.droplevel([0, 1, 2])
    for d in g.index:
        past = g[(g.index <= d - pd.Timedelta(days=2)) & (g.index > d - pd.Timedelta(days=30))]
        if len(past) >= 7:
            k = past.mean().idxmin()
            i = kk + (d,)
            pick[i], reg_t[i], sh_t[i], en_t[i] = k, g.loc[d, k], SH.loc[i, k], EN.loc[i, k]
T = pd.DataFrame({"pick": pd.Series(pick), "reg_tuned": pd.Series(reg_t), "S_tuned": pd.Series(sh_t), "E_tuned": pd.Series(en_t)})
T.index.names = key + ["day"]
T["reg0"], T["S0"], T["E0"] = W[0.0], SH[0.0], EN[0.0]
T = T.reset_index()
T = T[T.day >= START]
dem = R[(R.k == 0) & (R.tie == "default") & (R.day >= START)].dem.sum()

lines = []


def say(s):
    lines.append(s)
    print(s)


def red(a, b):
    m, ci, _ = boot_mean(1 - a / b)
    return f"{100 * m:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


D = {t: T[T.tie == t].groupby("day")[["reg_tuned", "reg0"]].sum() for t in ("default", "mid")}
say(f"# 安全库存对照（A7，{T.day.nunique()} 天，自 {START.date()}；{T.cfg.nunique()} 种配置 × {T.model.nunique()} 个模型）")
say("k 为安全库存（开发期小时平均需求的倍数）。日度遗憾降幅 = 1 − Σ前者 / Σ后者，95% 平稳块自助区间。")
say(f"1. 默认解 + 调好的安全库存 vs 默认解：{red(D['default'].reg_tuned, D['default'].reg0)}")
say(f"2. 中点解（不加）vs 默认解 + 调好的安全库存：{red(D['mid'].reg0, D['default'].reg_tuned)}")
say(f"3. 中点解 + 调好的安全库存 vs 中点解：{red(D['mid'].reg_tuned, D['mid'].reg0)}")
say(f"   参照：中点解 vs 默认解（同一窗口）：{red(D['mid'].reg0, D['default'].reg0)}")
for t in ("default", "mid"):
    tot = T[T.tie == t][["reg_tuned", "reg0"]].sum()
    say(f"   总遗憾（{t}）：不加 {tot.reg0:,.0f}，调好的安全库存 {tot.reg_tuned:,.0f}（之比 {tot.reg_tuned / tot.reg0:.3f}）")
# 事后最优固定 k
Wt = W.reset_index()
Wt = Wt[Wt.day >= START]
for t in ("default", "mid"):
    tot = Wt[Wt.tie == t][KS].sum()
    kb = tot.idxmin()
    say(f"4. 事后最优固定 k（{t}）= {kb}：总遗憾 / 不加 = {tot[kb] / tot[0.0]:.3f}；各 k：" + "，".join(f"{k}: {tot[k] / tot[0.0]:.3f}" for k in KS))
dd = Wt[Wt.tie == "default"].groupby("day")[KS].sum()
dm = Wt[Wt.tie == "mid"].groupby("day")[KS].sum()
kb = Wt[Wt.tie == "default"][KS].sum().idxmin()
say(f"5. 中点解（不加）vs 默认解 + 事后最优固定 k（{kb}）：{red(dm[0.0], dd[kb])}")
say(f"   中点解更优的天数（vs 默认 + 调好的安全库存）：{100 * (D['mid'].reg0 < D['default'].reg_tuned).mean():.1f}%")
say("缺供率（占需求）：" + "，".join(f"{t} 不加 {100 * T[T.tie == t].S0.sum() / dem:.2f}% / 调好 {100 * T[T.tie == t].S_tuned.sum() / dem:.2f}%" for t in ("default", "mid")))
say("购电费（含日末估值）相对默认解不加：" + "，".join(f"{t} 调好 {T[T.tie == t].E_tuned.sum() / T[T.tie == 'default'].E0.sum():.4f}" for t in ("default", "mid"))
    + f"，中点不加 {T[T.tie == 'mid'].E0.sum() / T[T.tie == 'default'].E0.sum():.4f}")
say("滚动调参选中的 k 分布：" + "；".join(f"{t}: " + T[T.tie == t].pick.value_counts(normalize=True).sort_index().round(3).to_dict().__repr__() for t in ("default", "mid")))
by = T[T.tie == "default"].groupby("model")[["reg_tuned"]].sum().join(T[T.tie == "mid"].groupby("model")[["reg0"]].sum())
say("中点解（不加）对每个模型都优于默认解 + 调好的安全库存：" + str(bool((by.reg0 < by.reg_tuned).all())) + "；各模型之比 " + ", ".join(f"{m} {r:.2f}" for m, r in (by.reg0 / by.reg_tuned).items()))
(OUT / "safety_stock_summary.txt").write_text("\n".join(lines), encoding="utf-8")
