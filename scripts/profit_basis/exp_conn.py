"""A7 需求时刻口径的稳健性：把"按断开时刻"改成"按接入时刻"统计补能量（同一批清洗后的会话），
用冻结的特征与超参数重做三个简单预测（发布时刻可得的季节朴素、季节分位数均值、LightGBM 单站均值，每 7 天扩张重训），
再算计划选择（默认 / 中点 / 最大缓冲）的遗憾。只为回答：P4 是否依赖需求时刻的定义。用法：python exp_conn.py"""
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
import fc_stat
from analysis import boot_mean
from exp_bss import mix

OUT = HERE.parents[2] / "results" / "profit_basis"
RAW = ROOT / "data" / "raw" / "mp_evdata"
GRID = list(cfg.BSS_REP) + [(c, p, 3) for c in (4, 6, 8) for p in (2.5, 4.0)]


def build_series():
    d = pd.read_excel(RAW / "charging_session.xlsx", sheet_name="A7")
    d.columns = ["conn", "disc", "energy"]
    d["energy"] = pd.to_numeric(d.energy, errors="coerce")
    d["conn"], d["disc"] = pd.to_datetime(d.conn, errors="coerce"), pd.to_datetime(d.disc, errors="coerce")
    a = d[d.energy.gt(0) & d.disc.notna() & d.conn.notna()].copy()
    onhour = (a.disc.dt.minute == 0) & (a.disc.dt.second == 0)
    mass = (a.disc >= "2024-07-26 10:00") & (a.disc < "2024-07-26 11:00")
    a = a[~onhour & ~mass]
    idx = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    yd = a.groupby(a.disc.dt.floor("h")).energy.sum().reindex(idx, fill_value=0.0)
    yc = a.groupby(a.conn.dt.floor("h")).energy.sum().reindex(idx, fill_value=0.0)
    return yd, yc


def lgb_mean(y, r0, r1):
    F = D.features(y)
    cutoff = r0 - pd.Timedelta(days=2) + pd.Timedelta(hours=23)
    tr = F.loc[:cutoff].assign(y=y.loc[:cutoff]).dropna(subset=["lag_d7", "mean_d2_d8", "y"])
    idx = pd.date_range(r0, r1 + pd.Timedelta(hours=23), freq="h")
    te = F.reindex(idx)
    return pd.Series(np.maximum(fc_stat.lgb_predict(tr[D.FEATS], tr.y, te[D.FEATS], len(cfg.QS)), 0), index=idx)


def task(d, y, price, tot, hmean, fcs):
    rows = []
    for par in GRID:
        P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
        for m, f in fcs.items():
            lo, hi = P.plan(f, "min"), P.plan(f, "max")
            rows.append(dict(day=d, cfg=str(par), model=m, mae=float(np.abs(y - f).mean()), reg_default=P.cost(P.plan(f, "default")) - P.opt,
                             reg_mid=P.cost(mix(lo, hi, 0.5)) - P.opt, reg_max=P.cost(hi) - P.opt))
    return rows


if __name__ == "__main__":
    yd, yc = build_series()
    H = D.mp_hourly()
    print("与正式数据 A7_kwh 的一致性（按断开时刻重建）：最大绝对差", float((yd - H[cfg.BSS].reindex(yd.index)).abs().max()))
    print("日总量相关（接入 vs 断开）", round(float(np.corrcoef(yd.resample("D").sum(), yc.resample("D").sum())[0, 1]), 4),
          "；逐时相关", round(float(np.corrcoef(yd, yc)[0, 1]), 3))
    print("各小时占比（断开）", (yd.groupby(yd.index.hour).sum() / yd.sum()).round(3).tolist())
    print("各小时占比（接入）", (yc.groupby(yc.index.hour).sum() / yc.sum()).round(3).tolist())
    res = {}
    for name, y in (("disc", yd), ("conn", yc)):
        W = fc_stat.windows(cfg.MP_REFITS, cfg.MP_FC_END)
        parts = Parallel(n_jobs=6)(delayed(lgb_mean)(y, r0, r1) for r0, r1 in W)
        lg = pd.concat(parts)
        B = fc_stat.baselines("x", y, cfg.MP_FC_START, cfg.MP_FC_END)
        FC = {"lgb_local": lg, "naive": B[B.model == "naive"].set_index("t")["mean"], "seasonal_q": B[B.model == "seasonal_q"].set_index("t")["mean"]}
        hmean = float(y.loc[:cfg.MP_DEV_END].mean())
        T = []
        for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
            idx = pd.date_range(d, periods=24, freq="h")
            yy = y.reindex(idx).to_numpy()
            fcs = {m: np.maximum(v.reindex(idx).to_numpy(), 0) for m, v in FC.items()}
            if np.isfinite(yy).all() and all(np.isfinite(v).all() for v in fcs.values()):
                T.append((d, yy, D.price(cfg.BSS, d.month), HK(D.price("A7", d.month, "total")), hmean, fcs))
        R = pd.DataFrame([r for rr in Parallel(n_jobs=6, batch_size=4)(delayed(task)(*t) for t in T) for r in rr])
        R["series"] = name
        res[name] = R
        P = R.cfg.map(eval)
        R["set"] = np.where(P.map(lambda t: t[1]) > 2, "anchor", "paper")
        for st, g in R.groupby("set"):
            dd = g.groupby("day")[["reg_default", "reg_mid", "reg_max"]].sum()
            m, ci, _ = boot_mean(1 - dd.reg_mid / dd.reg_default)
            print(f"[{name} | {st}] 中点 vs 默认 日度降幅 {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]；各模型 MAE 与日均遗憾：")
            print(g.groupby("model")[["mae", "reg_default", "reg_mid", "reg_max"]].mean().round(1).to_string())
    pd.concat(res.values()).to_parquet(OUT / "bss_conn.parquet")
