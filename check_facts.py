"""核对正文里两处事实：A7 并发连接数 / 并发充电功率的时间加权分位数；各模型在 A7 的预测偏差。"""
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
import numpy as np
import pandas as pd
import cfg

RAW = ROOT / "data" / "raw" / "mp_evdata"
d = pd.read_excel(RAW / "charging_session.xlsx", sheet_name="A7")
d.columns = ["conn", "disc", "energy"]
d["energy"] = pd.to_numeric(d.energy, errors="coerce")
d["conn"], d["disc"] = pd.to_datetime(d.conn, errors="coerce"), pd.to_datetime(d.disc, errors="coerce")
d = d.dropna()
d = d[(d.energy > 0) & (d.disc > d.conn)]
mass = (d.disc >= "2024-07-26 10:00") & (d.disc < "2024-07-26 11:00")
d = d[~mass]
d["kw"] = d.energy / ((d.disc - d.conn).dt.total_seconds() / 3600)
grid = pd.date_range("2024-01-01", "2024-12-31 23:55", freq="5min")
ev_n = pd.concat([pd.Series(1, index=d.conn), pd.Series(-1, index=d.disc)]).sort_index().cumsum()
ev_p = pd.concat([pd.Series(d.kw.to_numpy(), index=d.conn), pd.Series(-d.kw.to_numpy(), index=d.disc)]).sort_index().cumsum()
n = ev_n[~ev_n.index.duplicated(keep="last")].reindex(grid, method="ffill").fillna(0)
p = ev_p[~ev_p.index.duplicated(keep="last")].reindex(grid, method="ffill").fillna(0)
hm = d.energy.sum() / (366 * 24)
print("时间加权（5 分钟采样）并发连接电池数：中位数 %d，p90 %d，p99 %d，最大 %d" % (n.median(), n.quantile(.9), n.quantile(.99), n.max()))
print("时间加权并发充电功率 / 均值需求（%.1f kWh/h）：中位数 %.2f，p90 %.2f，p99 %.2f，最大 %.2f" % (hm, p.median() / hm, p.quantile(.9) / hm, p.quantile(.99) / hm, p.max() / hm))
print("每次会话电量 中位数 %.1f kWh；库存折算：p90 %.1f h，p99 %.1f h（按中位电量）" % (d.energy.median(), n.quantile(.9) * d.energy.median() / hm, n.quantile(.99) * d.energy.median() / hm))
sh = d.groupby(d.conn.dt.hour).energy.sum() / d.energy.sum()
import data as D
pr = D.price(cfg.BSS, 6)
lv = np.unique(pr)
print("按接入时刻的电量占比：谷段 %.3f（时长占比 %.3f），其余 %.3f" % (sh[pr == lv[0]].sum(), (pr == lv[0]).mean(), sh[pr != lv[0]].sum()))
for m in ["lqr", "lgb_local", "lgb_global", "chronos2_cov", "timesfm3_cov", "combo5", "seasonal_q", "naive"]:
    F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet", columns=["series", "t", "y", "mean"])
    F = F[(F.series == cfg.BSS) & (F.t >= cfg.MP_EVAL_START)].dropna()
    print(f"{m:14s} 偏差 {100 * (F['mean'] - F.y).sum() / F.y.sum():+.1f}%  MAE {np.abs(F['mean'] - F.y).mean():.1f}")
