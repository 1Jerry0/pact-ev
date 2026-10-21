"""表 8 用：电池级（A7，6 h，2.5 倍功率，LightGBM 均值预测）最大缓冲相对最小缓冲的成本差，三种缺供计价。
利润口径读 results/profit_basis/bl_fc.parquet；×3、×10 读 results/bl_fc.parquet（预先设定口径，scripts/exp_bl.py 的输出；cfg 第三位为倍数）。
输出：结果/maxmin_bases.txt"""
import pathlib
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from exp_packs_value import rel_ci  # noqa: E402

TUNE0 = pd.Timestamp("2024-05-01")
src = {"profit_basis": (ROOT / "results" / "profit_basis" / "bl_fc.parquet", "(6, 2.5, 3)"),
       "×3": (ROOT / "results" / "bl_fc.parquet", "(6, 2.5, 3)"),
       "×10": (ROOT / "results" / "bl_fc.parquet", "(6, 2.5, 10)")}
lines = []
for lab, (fp, c) in src.items():
    F = pd.read_parquet(fp)
    for pc in (12.0, 25.0):
        g = F[(F.cfg == c) & (F.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
        g = g[g.index >= TUNE0]
        m, lo, hi = rel_ci(g["mean@0|max"].to_numpy(), g["mean@0|min"].to_numpy())
        lines.append(f"{lab} {pc:.0f} kW：最大 vs 最小 {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]（{len(g)} 天）")
txt = "\n".join(lines)
(ROOT / "results" / "maxmin_bases.txt").write_text(txt, encoding="utf-8")
print(txt)
