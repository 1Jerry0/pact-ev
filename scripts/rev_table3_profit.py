"""表 3（预测精度与调度遗憾，中点计划）的利润口径版本：7 个储能站（9 种储能情景，超限按服务费计价）与换电站 A7（小时模型，数据锚定 6 种配置，缺供按逐小时零售价计价）。
各行：模型 MAE 与遗憾的 Kendall τ（各情景中位数）；MAE 最优≠遗憾最低的情景比例；TimesFM-3.0、Chronos-2、线性分位数回归相对单站 LightGBM 的总遗憾之比及平稳块自助 95% 区间。
另给求解器默认计划下 A7 的同一比较。输出：结果/table3_profit.md"""
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
ST = pd.read_parquet(OUT / "cb_storage_sched.parquet")
ST["reg"] = ST.E + ST.Pen - ST.cost_opt
ST = ST.rename(columns={"station": "unit"})
A7 = pd.read_parquet(OUT / "cb_hourly_plans.parquet", filters=[("inp", "==", "mean")])
A7 = A7[A7.model != "check_x1max"].copy()
A7["reg"] = A7.E + A7.Pen - A7.E_opt - A7.Pen_opt
A7["unit"] = "A7"


def col(g):
    taus, bd = [], []
    for c, h in g.groupby("cfg"):
        a = h.groupby("model").agg(mae=("mae", "mean"), reg=("reg", "sum"))
        taus.append(kendalltau(a.mae, a.reg)[0])
        bd.append(a.mae.idxmin() != a.reg.idxmin())
    d = g.pivot_table(index="day", columns="model", values="reg", aggfunc="sum").sort_index()
    out = {"τ": f"{np.median(taus):.2f}", "最准≠最省": f"{100 * np.mean(bd):.0f}%"}
    for m, lab in (("timesfm3_cov", "TimesFM-3.0 / LightGBM"), ("chronos2_cov", "Chronos-2 / LightGBM"), ("lqr", "线性分位数回归 / LightGBM")):
        r, lo, hi = rel_ci(d[m], d["lgb_local"])
        sig = lo > 0 or hi < 0
        txt = f"{1 + r:.2f} [{1 + lo:.2f}, {1 + hi:.2f}]"
        out[lab] = f"**{txt}**" if sig else txt
    return out


cols = {}
for u, g in ST[ST.tie == "mid"].groupby("unit"):
    cols[u] = col(g)
cols["A7"] = col(A7[A7.tie == "mid"])
cols["A7（求解器默认计划）"] = col(A7[A7.tie == "default"])
T = pd.DataFrame(cols)
md = T.to_markdown()
(OUT / "table3_profit.md").write_text(md, encoding="utf-8")
print(md)
mae = A7[A7.tie == "mid"].groupby("model").mae.mean().sort_values()
print("A7 各模型 MAE：", mae.round(1).to_dict())
for u, g in ST[ST.tie == "mid"].groupby("unit"):
    print(u, "MAE 最小：", g.groupby("model").mae.mean().idxmin())
