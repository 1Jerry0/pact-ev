"""数据读取与发布时刻可得的特征。输入文件由 prepare_data.py 在本机生成后随代码上传。"""
import json
import numpy as np
import pandas as pd
import cfg

_PRICES = None


def mp_hourly():
    """MP-EVData 全年逐时：常规站 kWh/h；A7_kwh 为清洗后的换电补能量（按断开时刻）；A8 为换电次数。"""
    return pd.read_parquet(cfg.DATA / "mp_hourly.parquet").set_index("t").asfreq("h")


def uev_volume(version="volume"):
    V = pd.read_csv(cfg.DATA / "urbanev" / f"{version}.csv", index_col=0, parse_dates=True)
    V = V.loc[:cfg.UEV_FC_END + pd.Timedelta(hours=23)].astype(float)
    V = V.loc[:, V.loc[:cfg.UEV_DEV_END].mean() > 1.0]
    V.columns = [str(c) for c in V.columns]
    if cfg.SMOKE:
        V = V.iloc[:, :10]
    return V


def price(station, month, kind="elec"):
    """24 小时电价（元/kWh）。免费站用同类站代理；kind: elec 电度电价 / total 总价。"""
    global _PRICES
    if _PRICES is None:
        _PRICES = json.loads((cfg.DATA / "prices.json").read_text(encoding="utf-8"))
    s = cfg.PRICE_PROXY.get(station, station)
    return np.array(_PRICES[f"{s}|{month}|{kind}"])


def weather():
    """上海 2024 JMA GSM 历史预报（Open-Meteo）。返回 (发布时刻可得的预报, 实际天气)。"""
    W = pd.DataFrame(json.loads((cfg.DATA / "openmeteo_jma_gsm_shanghai_2024.json").read_text(encoding="utf-8"))["hourly"])
    W["time"] = pd.to_datetime(W.time)
    W = W.set_index("time")
    early = W.index.hour < 8
    fc = pd.DataFrame(index=W.index)
    for v in ["temperature_2m", "precipitation", "relative_humidity_2m"]:
        fc[v] = np.where(early, W[f"{v}_previous_day1"], W[f"{v}_previous_day2"])
    return fc, W[["temperature_2m", "precipitation", "relative_humidity_2m"]].copy()


def asof_naive(y):
    """D−1 日 08:00 发布：08:00 前的时段用 D−1 同时段，08:00 起用 D−2 同时段。"""
    return pd.Series(np.where(y.index.hour < 8, y.shift(24), y.shift(48)), index=y.index)


def calendar(idx):
    day = idx.normalize()
    hol = day.isin(cfg.HOLIDAYS)
    work = (~hol & (idx.dayofweek < 5)) | day.isin(cfg.MAKEUP)
    return hol.astype(float), work.astype(float)


FEATS = ["hour", "dow", "holiday", "workday", "hol_prev", "hol_next", "lag_asof", "lag_d2", "lag_d3", "lag_d7",
         "lag_d14", "mean_d2_d8", "daily_d2", "daily_ratio_w", "early_d1", "early_d2"]
LAGC = ["lag_asof", "lag_d2", "lag_d3", "lag_d7", "lag_d14", "mean_d2_d8", "daily_d2", "early_d1", "early_d2"]
LIN = ["lag_asof", "lag_d2", "lag_d7", "mean_d2_d8", "early_d1", "workday", "holiday"]
WXF = ["wx_t", "wx_p", "wx_rh", "wx_t_mean", "wx_t_min", "wx_p_sum"]


def features(y):
    """逐时特征，全部满足发布时刻可得性（目标日 D 的特征只用 D−1 日 08:00 前的数据）。"""
    idx = y.index
    day = idx.normalize()
    f = pd.DataFrame(index=idx)
    f["hour"], f["dow"] = idx.hour, idx.dayofweek
    hol, work = calendar(idx)
    f["holiday"], f["workday"] = hol, work
    f["hol_prev"] = (day - pd.Timedelta(days=1)).isin(cfg.HOLIDAYS).astype(int)
    f["hol_next"] = (day + pd.Timedelta(days=1)).isin(cfg.HOLIDAYS).astype(int)
    f["lag_asof"] = asof_naive(y)
    for k in (2, 3, 7, 14):
        f[f"lag_d{k}"] = y.shift(24 * k)
    f["mean_d2_d8"] = sum(y.shift(24 * k) for k in range(2, 9)) / 7.0
    daily = y.groupby(day).sum()
    d2, d9 = daily.shift(2).reindex(day).to_numpy(), daily.shift(9).reindex(day).to_numpy()
    f["daily_d2"] = d2
    f["daily_ratio_w"] = d2 / np.where(d9 > 0, d9, np.nan)
    e = y[idx.hour < 8]
    early = e.groupby(e.index.normalize()).sum()
    f["early_d1"] = early.shift(1).reindex(day).to_numpy()
    f["early_d2"] = early.shift(2).reindex(day).to_numpy()
    return f


def add_weather(f, w):
    w = w.reindex(f.index)
    f = f.copy()
    f["wx_t"], f["wx_p"], f["wx_rh"] = w.temperature_2m.to_numpy(), w.precipitation.to_numpy(), w.relative_humidity_2m.to_numpy()
    d = w.groupby(w.index.normalize()).agg(t_mean=("temperature_2m", "mean"), t_min=("temperature_2m", "min"),
                                           p_sum=("precipitation", "sum"))
    for c in d.columns:
        f[f"wx_{c}"] = d[c].reindex(f.index.normalize()).to_numpy()
    return f


def days(start, end):
    return pd.date_range(start, end, freq="D")
