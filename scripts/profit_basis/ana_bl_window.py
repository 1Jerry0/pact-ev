"""电池级"滚动调参管线 vs 滚动规则"对调参窗口长度的敏感性：窗口 14 / 28 / 56 天（两者同用一个窗口；都滞后 2 天）。
只用已存结果（bl_rule.parquet、bl_fc.parquet），比较样本统一从 2024-06-01 起。用法：python ana_bl_window.py"""
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "formal" / "code"))
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
import cfg
from exp_packs_value import rel_ci

OUT = HERE.parents[2] / "results" / "profit_basis"
RU = pd.read_parquet(OUT / "bl_rule.parquet")
FC = pd.read_parquet(OUT / "bl_fc.parquet")
T0 = pd.Timestamp("2024-06-01")


def roll(M, days, W, fb):
    out = []
    for d in days:
        past = M.loc[d - pd.Timedelta(days=W + 1): d - pd.Timedelta(days=2)]
        out.append(M.loc[d, past.sum().idxmin() if len(past) >= 7 else fb])
    return np.array(out)


lines = []
for (c, pc), g in RU.groupby(["cfg", "pc"], sort=False):
    Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
    h = FC[(FC.cfg == c) & (FC.pc == pc) & ~FC.cand.str.startswith("pi@")]
    Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
    days = [d for d in Fc.index if d >= T0]
    k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
    s = f"{c} {pc:.0f} kW（{len(days)} 天）：调参管线相对规则 %"
    for W in (14, 28, 56):
        ref, x = roll(Mc, days, W, k0), roll(Fc, days, W, "mean@0|mid")
        m, lo, hi = rel_ci(x, ref)
        s += f"；窗口 {W} 天 {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    lines.append(s)
txt = "\n".join(lines)
(OUT / "ana_bl_window.txt").write_text(txt, encoding="utf-8")
print(txt)
