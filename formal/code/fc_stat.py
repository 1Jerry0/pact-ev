"""统计与机器学习模型的滚动预测（CPU 并行）。用法：python fc_stat.py mp | uev
每月月初扩张重训；训练样本只含目标日 ≤ D−2 的完整观测日；预测发布于 D−1 日 08:00。
输出：out/fc/{mp,uev}_{model}.parquet，列 series, model, t, y, mean, q05..q95。"""
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import QuantileRegressor, LinearRegression
import cfg
import data as D
from util import log, pmap, save, done

warnings.filterwarnings("ignore")


def windows(refits, end):
    """每次重训覆盖的目标日区间。"""
    out = []
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] - pd.Timedelta(days=1) if i + 1 < len(refits) else end
        out.append((r0, min(r1, end)))
    return out


def pack(series, model, idx, y, mean, Q):
    Q = np.sort(np.maximum(np.asarray(Q, float), 0), axis=1)
    df = pd.DataFrame(Q, columns=cfg.QC)
    df.insert(0, "mean", np.maximum(np.asarray(mean, float), 0))
    df.insert(0, "y", y.reindex(idx).to_numpy())
    df.insert(0, "t", idx)
    df.insert(0, "model", model)
    df.insert(0, "series", series)
    return df


def baselines(series, y, start, end, hist=28):
    """季节朴素（发布时刻可得）、周季节朴素：点预测 + 近 28 天同小时样本外残差的经验分位数；
    季节经验分位数：近 28 天同小时的经验分布。历史只取目标日 ≤ D−2。"""
    out = []
    days = D.days(start, end)
    hours = y.index.hour.to_numpy()
    dayidx = y.index.normalize()
    for name, pt in [("naive", D.asof_naive(y)), ("naive7", y.shift(168))]:
        res = (y - pt)
        rows_m, rows_q, idxs = [], [], []
        for d in days:
            m = (dayidx <= d - pd.Timedelta(days=2)) & (dayidx > d - pd.Timedelta(days=2 + hist))
            past = res[m]
            idx = pd.date_range(d, periods=24, freq="h")
            p = pt.reindex(idx).to_numpy()
            Q = np.vstack([p[h] + np.nanquantile(past[hours[m] == h], cfg.QS) if np.isfinite(past[hours[m] == h]).sum() > 2
                           else np.full(len(cfg.QS), p[h]) for h in range(24)])
            rows_m.append(p); rows_q.append(Q); idxs.append(idx)
        out.append(pack(series, name, idxs[0].append(idxs[1:]), y, np.concatenate(rows_m), np.vstack(rows_q)))
    rows_m, rows_q, idxs = [], [], []
    for d in days:
        m = (dayidx <= d - pd.Timedelta(days=2)) & (dayidx > d - pd.Timedelta(days=2 + hist))
        past = y[m]
        hh = hours[m]
        idx = pd.date_range(d, periods=24, freq="h")
        rows_q.append(np.vstack([np.nanquantile(past[hh == h], cfg.QS) for h in range(24)]))
        rows_m.append(np.array([np.nanmean(past[hh == h]) for h in range(24)]))
        idxs.append(idx)
    out.append(pack(series, "seasonal_q", idxs[0].append(idxs[1:]), y, np.concatenate(rows_m), np.vstack(rows_q)))
    return pd.concat(out, ignore_index=True)


def lgb_predict(Xtr, ytr, Xte, qi, cat="auto"):
    """qi < 19：第 qi 个分位数；qi = 19：均值（L2）。"""
    obj = dict(objective="l2") if qi == len(cfg.QS) else dict(objective="quantile", alpha=float(cfg.QS[qi]))
    return lgb.LGBMRegressor(**obj, **cfg.LGB).fit(Xtr, ytr, categorical_feature=cat).predict(Xte)


def lin_design(F):
    X = F[D.LIN].copy()
    for h in range(24):
        X[f"h{h}"] = (F.hour == h).astype(float)
    return X


def task_local(series, y, wfc, wact, r0, r1, model):
    """单站 LightGBM（含天气变体）或线性分位回归，一个重训窗口。"""
    F = D.features(y)
    feats = list(D.FEATS)
    if model == "lgb_wx":
        F, feats = D.add_weather(F, wfc), feats + D.WXF
    elif model == "lgb_wx_perfect":
        F, feats = D.add_weather(F, wact), feats + D.WXF
    cutoff = r0 - pd.Timedelta(days=2) + pd.Timedelta(hours=23)
    tr = F.loc[:cutoff].assign(y=y.loc[:cutoff]).dropna(subset=["lag_d7", "mean_d2_d8", "y"])
    idx = pd.date_range(r0, r1 + pd.Timedelta(hours=23), freq="h")
    te = F.reindex(idx)
    if len(tr.dropna(subset=D.LIN) if model == "lqr" else tr) < 24 * 7:   # 训练样本不足一周（仅 A8 的 2 月预热窗口），跳过
        return pd.DataFrame()
    if model == "lqr":
        if cfg.LQR_WINDOW_DAYS:
            tr = tr[tr.index >= cutoff - pd.Timedelta(days=cfg.LQR_WINDOW_DAYS)]
        tr = tr.dropna(subset=D.LIN)
        Xtr, Xte = lin_design(tr), lin_design(te.fillna(tr[D.LIN].mean()))
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        Xtr, Xte = ((Xtr - mu) / sd).to_numpy(), ((Xte - mu) / sd).fillna(0).to_numpy()
        Q = np.column_stack([QuantileRegressor(quantile=float(t), alpha=0.0, solver="highs").fit(Xtr, tr.y).predict(Xte)
                             for t in cfg.QS])
        mean = LinearRegression().fit(Xtr, tr.y).predict(Xte)
    else:
        Q = np.column_stack([lgb_predict(tr[feats], tr.y, te[feats], qi) for qi in range(len(cfg.QS))])
        mean = lgb_predict(tr[feats], tr.y, te[feats], len(cfg.QS))
    return pack(series, model, idx, y, mean, Q)


def global_frame(Y, cutoff):
    """全局模型：各序列按训练期均值缩放后合并，序列编号作类别特征。"""
    rows = []
    for k, s in enumerate(Y.columns):
        y = Y[s]
        F = D.features(y)
        sc = y.loc[:cutoff].mean()
        sc = sc if np.isfinite(sc) and sc > 0 else 1.0
        F[D.LAGC] = F[D.LAGC] / sc
        F["sid"], F["y"], F["sc"], F["series"] = k, y / sc, sc, s
        rows.append(F)
    A = pd.concat(rows)
    A["sid"] = pd.Categorical(A.sid, categories=range(len(Y.columns)))
    return A


def task_global(Y, r0, r1, qi):
    cutoff = r0 - pd.Timedelta(days=2) + pd.Timedelta(hours=23)
    A = global_frame(Y, cutoff)
    tr = A[A.index <= cutoff].dropna(subset=["lag_d7", "mean_d2_d8", "y"])
    te = A[(A.index >= r0) & (A.index <= r1 + pd.Timedelta(hours=23))]
    pred = lgb_predict(tr[D.FEATS + ["sid"]], tr.y, te[D.FEATS + ["sid"]], qi, cat=["sid"]) * te.sc.to_numpy()
    return r0, qi, pd.DataFrame({"series": te.series.to_numpy(), "t": te.index, "pred": pred})


def assemble_global(results, Y, name):
    parts = {}
    for r0, qi, df in results:
        parts.setdefault(r0, {})[qi] = df
    out = []
    for r0, qd in sorted(parts.items()):
        base = qd[0][["series", "t"]].copy()
        Q = np.column_stack([qd[i].pred.to_numpy() for i in range(len(cfg.QS))])
        mean = qd[len(cfg.QS)].pred.to_numpy()
        for s, g in base.groupby("series", sort=False):
            ix = g.index.to_numpy()
            out.append(pack(s, name, pd.DatetimeIndex(g.t), Y[s], mean[ix], Q[ix]))
    return pd.concat(out, ignore_index=True)


def run(ds):
    if ds == "mp":
        H = D.mp_hourly()
        Y = H[cfg.MP_SERIES]
        start, end, refits = cfg.MP_FC_START, cfg.MP_FC_END, cfg.MP_REFITS
        wfc, wact = D.weather()
        local_models = ["lqr", "lgb_local", "lgb_wx", "lgb_wx_perfect"]
        glob_series = cfg.MP_SERIES
    else:
        Y = D.uev_volume()
        start, end, refits = cfg.UEV_FC_START, cfg.UEV_FC_END, cfg.UEV_REFITS
        wfc = wact = None
        local_models = ["lqr"]
        glob_series = list(Y.columns)
    W = windows(refits, end)
    if not done(cfg.OUT / "fc" / f"{ds}_naive.parquet"):
        res = pmap(lambda s: baselines(s, Y[s], start, end), [(s,) for s in Y.columns], desc=f"{ds} 基准", name=f"fc_{ds}")
        B = pd.concat(res, ignore_index=True)
        for m, g in B.groupby("model"):
            save(g, cfg.OUT / "fc" / f"{ds}_{m}.parquet")
    for m in local_models:
        fp = cfg.OUT / "fc" / f"{ds}_{m}.parquet"
        if done(fp):
            continue
        tasks = [(s, Y[s], wfc, wact, r0, r1, m) for s in Y.columns for r0, r1 in W]
        save(pd.concat(pmap(task_local, tasks, desc=f"{ds} {m}", name=f"fc_{ds}"), ignore_index=True), fp)
    name = "lgb_global"
    fp = cfg.OUT / "fc" / f"{ds}_{name}.parquet"
    if not done(fp):
        tasks = [(Y[glob_series], r0, r1, qi) for r0, r1 in W for qi in range(len(cfg.QS) + 1)]
        save(assemble_global(pmap(task_global, tasks, desc=f"{ds} {name}", name=f"fc_{ds}", batch_size=1), Y, name), fp)
    log(f"{ds} 统计与机器学习模型预测完成", f"fc_{ds}")


if __name__ == "__main__":
    run(sys.argv[1])
