"""时序基础模型零样本（GPU）。用法：python fc_tsfm.py mp | uev
D−1 日 08:00 发布，上下文截至 D−1 日 07:00，预测 40 小时后取 D 日 24 小时。
MP：chronos2（单变量）、chronos2_cov（节假日/工作日）、chronos2_wx（再加天气预报）、timesfm3、timesfm3_cov；
UEV：chronos2_cov、timesfm3_cov。按月分块保存，可续跑。"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import cfg
import data as D
from util import log, save, done
from fc_stat import pack

H = 40
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TFM_LEVELS = np.round(np.arange(0.1, 0.91, 0.1), 1)
CHUNK = 50


def to19(Q):
    """TimesFM 的 0.1–0.9 九个分位数插值到 19 个，两端线性外推。"""
    out = np.empty((Q.shape[0], len(cfg.QS)))
    lv = TFM_LEVELS
    for i, row in enumerate(np.sort(Q, axis=1)):
        out[i] = np.interp(cfg.QS, lv, row)
        out[i, 0] = row[0] - (lv[0] - cfg.QS[0]) * (row[1] - row[0]) / (lv[1] - lv[0])
        out[i, -1] = row[-1] + (cfg.QS[-1] - lv[-1]) * (row[-1] - row[-2]) / (lv[-1] - lv[-2])
    return out


def covs(idx, wfc=None):
    hol, work = D.calendar(idx)
    c = {"holiday": np.asarray(hol, float), "workday": np.asarray(work, float)}
    if wfc is not None:
        w = wfc.reindex(idx)
        c["wx_t"] = w.temperature_2m.to_numpy()
        c["wx_p"] = w.precipitation.to_numpy()
    return c


def context(y, last):
    y = y.loc[:last]
    y = y.loc[y.first_valid_index():]
    return y.fillna(0.0)


def run_chronos(Y, days, kind, wfc):
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEV)
    frames = []
    for d in days:
        last = d - pd.Timedelta(days=1) + pd.Timedelta(hours=7)
        fidx = pd.date_range(last + pd.Timedelta(hours=1), periods=H, freq="h")
        cols = list(Y.columns)
        for k in range(0, len(cols), CHUNK):
            ctx, fut = [], []
            for s in cols[k:k + CHUNK]:
                y = context(Y[s], last)
                c = pd.DataFrame({"id": s, "timestamp": y.index, "target": y.to_numpy()})
                f = pd.DataFrame({"id": s, "timestamp": fidx})
                if kind != "uni":
                    for name, v in covs(y.index, wfc if kind == "wx" else None).items():
                        c[name] = v
                    for name, v in covs(fidx, wfc if kind == "wx" else None).items():
                        f[name] = v
                ctx.append(c)
                fut.append(f)
            pred = pipe.predict_df(pd.concat(ctx), future_df=None if kind == "uni" else pd.concat(fut), prediction_length=H,
                                   quantile_levels=list(cfg.QS), id_column="id", timestamp_column="timestamp",
                                   target="target", batch_size=CHUNK)
            qcols = [str(t) if str(t) in pred.columns else f"{t:g}" for t in cfg.QS]
            for s, p in pred.groupby("id", sort=False):
                p = p.sort_values("timestamp").tail(24)
                frames.append((str(s), d, p["predictions"].to_numpy(), p[qcols].to_numpy()))
    return frames


def run_timesfm(Y, days, kind):
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    bs = int(os.environ.get("TFM_BATCH", "128" if DEV == "cuda" else "16"))
    fc = TimesFM3Evaluator(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=bs, device=DEV))
    frames = []
    for d in days:
        last = d - pd.Timedelta(days=1) + pd.Timedelta(hours=7)
        ctx, cv, names = [], [], []
        for s in Y.columns:
            y = context(Y[s], last)
            ctx.append(y.to_numpy().astype(np.float32)[None, :])
            if kind == "cov":
                full = pd.date_range(y.index[0], periods=len(y) + H, freq="h")
                cv.append(np.vstack(list(covs(full).values())).astype(np.float32))
            names.append(s)
        kw = dict(contexts=ctx, horizon=H, return_quantiles=True)
        if kind == "cov":
            kw["past_future_covariates"] = cv
        for s, o in zip(names, fc.predict_batch(**kw)):
            mean = np.asarray(o.forecast).reshape(-1)[-24:]
            Q = np.asarray(o.quantiles).reshape(-1, len(TFM_LEVELS))[-24:]
            frames.append((s, d, mean, to19(Q)))
    return frames


def run(ds):
    if ds == "mp":
        Y = D.mp_hourly()[cfg.MP_SERIES]
        start, end = cfg.MP_FC_START, cfg.MP_FC_END
        wfc, _ = D.weather()
        jobs = [("chronos2", "uni"), ("chronos2_cov", "cov"), ("chronos2_wx", "wx"), ("timesfm3", "uni"), ("timesfm3_cov", "cov")]
    else:
        Y = D.uev_volume()
        start, end, wfc = cfg.UEV_FC_START, cfg.UEV_FC_END, None
        jobs = [("chronos2_cov", "cov"), ("timesfm3_cov", "cov")]
    (cfg.OUT / "fc" / "parts").mkdir(exist_ok=True)
    for model, kind in jobs:
        fp = cfg.OUT / "fc" / f"{ds}_{model}.parquet"
        if done(fp):
            continue
        months = pd.period_range(start, end, freq="M")
        for mo in months:
            part = cfg.OUT / "fc" / "parts" / f"{ds}_{model}_{mo}.parquet"
            if done(part):
                continue
            days = D.days(max(start, mo.start_time), min(end, mo.end_time.normalize()))
            t0 = time.time()
            res = run_chronos(Y, days, kind, wfc) if model.startswith("chronos") else run_timesfm(Y, days, kind)
            out = [pack(s, model, pd.date_range(d, periods=24, freq="h"), Y[s], m, Q) for s, d, m, Q in res]
            save(pd.concat(out, ignore_index=True), part)
            log(f"{ds} {model} {mo} 完成，用时 {time.time() - t0:.0f} s", f"tsfm_{ds}")
        parts = sorted((cfg.OUT / "fc" / "parts").glob(f"{ds}_{model}_*.parquet"))
        save(pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True), fp)
    log(f"{ds} 基础模型预测完成", f"tsfm_{ds}")


if __name__ == "__main__":
    run(sys.argv[1])
