"""表 4 等计划选择结果在利润口径下的汇总（exp_cost_basis_t4.py 的输出）。
统计量与原表 4 相同：日度遗憾降幅 1 − Σ中点 / Σ对照（对模型与配置求和后逐日计算，取均值），平稳块自助法 95% 置信区间。
输出：结果/cb_t4_summary.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
from analysis import boot_mean  # noqa: E402

OUT = ROOT / "results"
lines = []


def say(s):
    lines.append(s)
    print(s)


for tag in ("A7", "A8", "A7tight"):
    fp = OUT / f"cb_t4_{tag}.parquet"
    if not fp.exists():
        continue
    R = pd.read_parquet(fp)
    daily = R.groupby("day")[[c for c in R if c.startswith("reg_")]].sum()
    say(f"## {tag}（{R.day.nunique()} 天，{R.cfg.nunique()} 种配置 × {R.model.nunique()} 个模型）")
    seg = []
    for base in ("default", "ipm", "noise", "flat", "min"):
        m, ci, _ = boot_mean(1 - daily.reg_mid / daily[f"reg_{base}"])
        seg.append(f"中点 vs {base} {100 * m:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
    say("  " + "；".join(seg) + f"；中点更优的天数 {100 * (daily.reg_mid < daily.reg_default).mean():.1f}%")
    m, ci, _ = boot_mean(1 - daily.reg_max / daily.reg_default)
    say(f"  最大 vs 默认 {100 * m:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
    tot = R[[c for c in R if c.startswith("reg_")]].sum()
    say("  总遗憾之比（中点/默认）%.3f；（最大/默认）%.3f" % (tot.reg_mid / tot.reg_default, tot.reg_max / tot.reg_default))
    dem = R.dem.sum()
    say("  缺供率：" + "，".join(f"{t} {100 * R[f'S_{t}'].sum() / dem:.2f}%" for t in ("default", "min", "mid", "max", "ipm", "flat")))
    say(f"  最优面宽度中位 {R.width.median():.3f}（四分位 {R.width.quantile(.25):.3f}—{R.width.quantile(.75):.3f}）；默认解位置中位 {R.pos_default.median():.2f}，均值 {R.pos_default.mean():.2f}")
    say(f"  晚间（17 时后）免赔额为零的小时：默认 {100 * R.zero_eve_default.mean():.0f}%，中点 {100 * R.zero_eve_mid.mean():.0f}%；"
        f"发生缺供的小时中免赔额为零：默认 {100 * R.short_zero_default.sum() / max(R.short_hours_default.sum(), 1):.0f}%；"
        f"发生缺供的小时占比：默认 {100 * R.short_hours_default.sum() / (24 * len(R)):.1f}%，中点 {100 * R.short_hours_mid.sum() / (24 * len(R)):.1f}%")
    by = R.groupby("model")[["reg_default", "reg_mid", "reg_max"]].mean()
    say(f"  各模型平均遗憾极差：默认 {np.ptp(by.reg_default):.1f}，中点 {np.ptp(by.reg_mid):.1f}，最大 {np.ptp(by.reg_max):.1f}；中点对每个模型都更优：{bool((by.reg_mid < by.reg_default).all())}")
    if "sc0.8_default" in R:
        g = R[R.model == "lgb_local"]
        say("  误差缩放（LightGBM，遗憾 / 原预测遗憾）：" + "，".join(
            f"k={k}: 默认 {g[f'sc{k}_default'].sum() / g.reg_default.sum():.2f}、中点 {g[f'sc{k}_mid'].sum() / g.reg_mid.sum():.2f}" for k in (0.8, 0.5, 0.2)))
        # 中点计划在原始预测下的遗憾，相当于默认计划在误差缩小多少时的遗憾（线性插值）
        ks = np.array([1.0, 0.8, 0.5, 0.2, 0.0])
        rd = np.array([g.reg_default.sum(), g["sc0.8_default"].sum(), g["sc0.5_default"].sum(), g["sc0.2_default"].sum(), 0.0])
        k_eq = float(np.interp(g.reg_mid.sum(), rd[::-1], ks[::-1]))
        say(f"  中点计划（原始预测）的遗憾 ≈ 默认计划在误差缩为 {k_eq:.2f} 倍时的遗憾，即误差缩小 {100 * (1 - k_eq):.0f}%")
(OUT / "cb_t4_summary.txt").write_text("\n".join(lines), encoding="utf-8")
