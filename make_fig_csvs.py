"""Build the two small CSV inputs of fig_v8.py from text outputs of the profit-basis runs:
- results/profit_basis/e4_dispersion.csv   (Fig. 4c, battery-level rows) from exp_packs_e4.txt
- results/profit_basis/exp_bl_a8_profile_fc.csv   (Fig. 7b, A8 row) from exp_bl_a8_profile.txt
Usage: python make_fig_csvs.py"""
import pathlib
import re
import pandas as pd

PR = pathlib.Path(__file__).resolve().parents[1] / "results" / "profit_basis"

# Fig. 4(c): dispersion of same-day cost (P90 - P10 range / mean cost of errors) under the solver's plan
rows = []
for line in (PR / "exp_packs_e4.txt").read_text(encoding="utf-8").splitlines():
    m = re.match(r"\((\d+), [\d.]+, \d+\) (\d+) kW", line)
    if not m:
        continue
    stock, kw = int(m.group(1)), int(m.group(2))
    d = re.search(r"\[默认解\].*?的 ([\d.]+) 倍 \[([\d.]+), ([\d.]+)\]", line)
    rows.append((f"A7 battery, {kw} kW, {stock} h", float(d.group(1)), float(d.group(2)), float(d.group(3)), stock, kw))
E = pd.DataFrame(rows, columns=["label", "m", "lo", "hi", "stock", "kw"]).sort_values(["kw", "stock"])
E[["label", "m", "lo", "hi"]].to_csv(PR / "e4_dispersion.csv", index=False, float_format="%.2f")

# Fig. 7(b): contribution of the forecast at A8 (LightGBM pipeline vs recent-profile plan, %)
line = (PR / "exp_bl_a8_profile.txt").read_text(encoding="utf-8").splitlines()[0]
m = re.search(r"LightGBM vs 近期曲线 ([-+\d.]+) \[([-+\d.]+), ([-+\d.]+)\]", line)
pd.DataFrame([("A8, 15 batteries", float(m.group(1)), float(m.group(2)), float(m.group(3)))],
             columns=["label", "m", "lo", "hi"]).to_csv(PR / "exp_bl_a8_profile_fc.csv", index=False, float_format="%.1f")
print((PR / "e4_dispersion.csv").read_text(encoding="utf-8"))
print((PR / "exp_bl_a8_profile_fc.csv").read_text(encoding="utf-8"))
