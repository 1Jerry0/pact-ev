"""Descriptive statistics of hourly demand (Table 1) and within-day features of the evaluation period (Table S2).
Reads formal/data (prepared by formal/code/prepare_data.py). Outputs results/descriptive_stats.csv and
results/within_day_features.csv. Usage: python desc_stats.py"""
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "formal" / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

H = pd.read_parquet(DATA / "mp_hourly.parquet").set_index("t").asfreq("h")
SERIES = [("A1", "A1", "taxi", "kWh"), ("A2", "A2", "taxi", "kWh"), ("A3", "A3", "bus depot", "kWh"),
          ("A4", "A4", "bus depot", "kWh"), ("A5", "A5", "residential", "kWh"), ("A6", "A6", "residential", "kWh"),
          ("A9", "A9", "heavy truck", "kWh"), ("A7", "A7_kwh", "swapping", "kWh"), ("A8", "A8", "swapping", "swaps")]
PERIODS = {"full sample": ("2024-01-01", "2024-12-31 23:00"), "evaluation": ("2024-04-01", "2024-12-31 23:00")}
A8_END = "2024-10-31 23:00"


def describe(x):
    x = pd.Series(x).dropna().to_numpy(float)
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1), min=x.min(), p25=np.percentile(x, 25), median=np.median(x),
                p75=np.percentile(x, 75), max=x.max(), skew=stats.skew(x), kurt=stats.kurtosis(x),  # excess kurtosis
                zero=np.mean(x == 0))


rows = []
for pname, (a, b) in PERIODS.items():
    for name, col, kind, unit in SERIES:
        end = min(pd.Timestamp(b), pd.Timestamp(A8_END)) if name == "A8" else pd.Timestamp(b)
        y = H.loc[a:end, col]
        d = describe(y)
        d.update(period=pname, station=name, kind=kind, unit=unit, start=y.first_valid_index(), end=y.last_valid_index(),
                 missing=int(y.isna().sum()))
        rows.append(d)

# UrbanEV: same zone filter as formal/code/data.py (development-period mean > 1 kWh/h); medians of zone-level statistics
V = pd.read_csv(DATA / "urbanev" / "volume.csv", index_col=0, parse_dates=True).astype(float)
V = V.loc[:"2023-02-28 23:00"]
V = V.loc[:, V.loc[:"2022-10-31 23:00"].mean() > 1.0]
for pname, (a, b) in {"full sample": ("2022-09-01", "2023-02-28 23:00"), "evaluation": ("2022-11-01", "2023-02-28 23:00")}.items():
    X = V.loc[a:b]
    Z = pd.DataFrame([describe(X[c]) for c in X.columns])
    d = Z.median().to_dict()
    d["n"] = len(X)
    zm = X.mean()
    d.update(period=pname, station="UrbanEV", kind=f"median over {V.shape[1]} zones", unit="kWh", start=X.index.min(),
             end=X.index.max(), missing=int(X.isna().sum().sum()), zone_mean_min=zm.min(), zone_mean_med=zm.median(),
             zone_mean_max=zm.max(), zone_n=X.shape[1])
    rows.append(d)
T = pd.DataFrame(rows)
T.to_csv(OUT / "descriptive_stats.csv", index=False, encoding="utf-8-sig")

# Table S2: evaluation period; hourly coefficient of variation = CV across days for each hour of the day, averaged over hours
feat = []
for name, col, kind, unit in SERIES:
    end = A8_END if name == "A8" else "2024-12-31 23:00"
    y = H.loc["2024-04-01":end, col]
    M = y.to_frame("y").assign(d=y.index.normalize(), h=y.index.hour).pivot(index="d", columns="h", values="y")
    feat.append(dict(station=name, kind=kind, unit=unit, daily_mean=y.groupby(y.index.normalize()).sum().mean(),
                     zero_share=float((y == 0).mean()), hourly_cv=float((M.std(ddof=1) / M.mean()).mean())))
y = H.loc["2024-04-01":"2024-12-31 23:00", "A7_n"]
feat.append(dict(station="A7", kind="swapping", unit="swaps", daily_mean=y.groupby(y.index.normalize()).sum().mean(),
                 zero_share=float((y == 0).mean()), hourly_cv=np.nan))
F = pd.DataFrame(feat)
F.to_csv(OUT / "within_day_features.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(T[T.period == "full sample"][["station", "kind", "n", "mean", "sd", "min", "median", "max", "skew", "kurt"]].round(2).to_string(index=False))
print(F.round(3).to_string(index=False))
