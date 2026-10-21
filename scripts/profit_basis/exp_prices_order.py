"""两个小检验（A7，LightGBM，只读冻结流程）：
  1. 最大库存解的计划库存是否逐时不低于最小库存解（"缺口随缓冲比例单调"命题的前提）；
  2. 功率锚定后误差价格的量级：默认解与中点解下，低估 / 高估 1 kWh 的平均价格（每 10 天取 1 天）。
用法：python exp_prices_order.py"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
import sim

S = cfg.BSS
GRID = [(4, 1.3, 3), (4, 2.5, 3), (6, 2.5, 3), (6, 4.0, 3), (8, 2.5, 3), (4, 2.5, 10)]


def task(d, y, price, tot, hmean, f, do_price):
    rows = []
    for par in GRID:
        P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
        lo, hi = P.plan(f, "min"), P.plan(f, "max")
        gap = float((hi[2] - lo[2]).min() / P.cap)
        r = dict(day=d, cfg=str(par), min_gap=gap, mean_gap=float((hi[2] - lo[2]).mean() / P.cap))
        if do_price:
            for tie in ("default", "mid"):
                wu, wo = P.dswe_weights("current", tie)
                r[f"pu_{tie}"], r[f"po_{tie}"] = float(wu.mean()), float(wo.mean())
                r[f"pu_eve_{tie}"] = float(wu[20:24].mean())
        rows.append(r)
    return rows


if __name__ == "__main__":
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    T = []
    for i, d in enumerate(pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D")):
        idx = pd.date_range(d, periods=24, freq="h")
        y, f = H[S].reindex(idx).to_numpy(), np.maximum(G.reindex(idx).to_numpy(), 0)
        if np.isfinite(y).all() and np.isfinite(f).all():
            T.append((d, y, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, f, i % 10 == 0))
    R = pd.DataFrame([r for rr in Parallel(n_jobs=6, batch_size=4)(delayed(task)(*t) for t in T) for r in rr])
    pd.set_option("display.width", 220)
    print("最大库存解逐时不低于最小库存解的比例（日×组合）：", round(float((R.min_gap >= -1e-7).mean()), 4), "；最小差 / 库容：", R.min_gap.min())
    print("\n误差价格（元/kWh，日均值的平均）：")
    print(R.dropna(subset=["pu_default"]).groupby("cfg")[["pu_default", "pu_mid", "po_default", "po_mid", "pu_eve_default", "pu_eve_mid"]].mean().round(3).to_string())
