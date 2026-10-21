"""Build the inputs of the formal run in formal/data/ from the public raw files in data/raw/ (see data/README.md):
- mp_hourly.parquet: hourly series of the MP-EVData stations. Conventional stations: sum of 15-min power / 4 (kWh/h);
  swapping stations A7 and A8: hourly sums of the 15-min profile; A7_n / A7_kwh: number and energy of A7 swaps by
  hour of disconnection after cleaning (zero-energy records, records disconnected exactly on the hour, and the
  259 records disconnected together between 10:00 and 11:00 on 26 July 2024 removed).
- prices.json: hourly electricity price ("elec") and retail price ("total") of each station for months 1-12.
- urbanev/volume.csv, urbanev/volume-11kW.csv: UrbanEV hourly zone energy (two estimates).
- openmeteo_jma_gsm_shanghai_2024.json: historical weather forecasts (only used by weather-feature models).
Usage: python prepare_data.py"""
import json
import pathlib
import re
import shutil
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]          # formal/
DATA = ROOT / "data"
RAWDIR = ROOT.parent / "data" / "raw"
RAW = RAWDIR / "mp_evdata"
(DATA / "urbanev").mkdir(parents=True, exist_ok=True)


def find(folder, stem):
    """MP-EVData file names differ between releases (e.g. 'station-level_load_Profile_15min.xlsx'); match loosely."""
    key = re.sub(r"[^a-z0-9]", "", stem.lower())
    for p in folder.glob("*.xlsx"):
        if re.sub(r"[^a-z0-9]", "", p.stem.lower()) == key:
            return p
    raise FileNotFoundError(f"{stem}.xlsx not found in {folder}")


def hourly_price(p, month, kind):
    rows = p[p.month == month]
    price = np.zeros(24)
    for name in ["Off", "Shoulder", "Peak", "Sharp"]:                 # later periods overwrite earlier ones (sharp over peak)
        for _, r in rows.iterrows():
            if not str(r.period).startswith(name):
                continue
            for sp in str(r.period).split("(")[1].rstrip(")").split(","):
                a, b = sp.split("-")
                price[int(a[:2]):int(b[:2]) or 24] = r[kind]
    if (price == 0).any():                                             # hours not covered take the flat (shoulder) price
        sh = rows[rows.period.str.startswith("Shoulder")][kind]
        price[price == 0] = sh.iloc[0] if len(sh) else price[price > 0].max()
    return price


def hourly_series():
    f15 = find(RAW, "station_level_load_profile_15min")
    cols = {}
    for s in [f"A{i}" for i in range(1, 11)]:
        d = pd.read_excel(f15, sheet_name=s)
        d.columns = ["t", "y"]
        d["t"] = pd.to_datetime(d.t)
        y = d.set_index("t").y.asfreq("15min").resample("h").sum()
        cols[s] = y if s in ("A7", "A8") else y / 4.0
    H = pd.DataFrame(cols)
    a7 = pd.read_excel(find(RAW, "charging_session"), sheet_name="A7")
    a7.columns = ["conn", "disc", "energy"]
    a7["energy"] = pd.to_numeric(a7.energy, errors="coerce")
    a7["disc"] = pd.to_datetime(a7.disc, errors="coerce")
    a7 = a7[a7.energy.gt(0) & a7.disc.notna()]
    onhour = (a7.disc.dt.minute == 0) & (a7.disc.dt.second == 0)
    mass = (a7.disc >= "2024-07-26 10:00") & (a7.disc < "2024-07-26 11:00")
    a7 = a7[~onhour & ~mass]
    hr = a7.disc.dt.floor("h")
    H["A7_n"] = a7.groupby(hr).size().reindex(H.index, fill_value=0).astype(float)
    H["A7_kwh"] = a7.groupby(hr).energy.sum().reindex(H.index, fill_value=0.0)
    return H


if __name__ == "__main__":
    H = hourly_series()
    H.reset_index().rename(columns={"index": "t"}).to_parquet(DATA / "mp_hourly.parquet", index=False)
    out = {}
    for s in ["A1", "A2", "A3", "A5", "A6", "A7", "A8"]:            # A4 and A9 have no tariff; config proxies them by A3 and A1
        p = pd.read_excel(find(RAW, "price"), sheet_name=s)
        p.columns = ["month", "period", "total", "elec", "service"]
        for m in range(1, 13):
            for kind in ("elec", "total"):
                out[f"{s}|{m}|{kind}"] = hourly_price(p, m, kind).round(6).tolist()
    (DATA / "prices.json").write_text(json.dumps(out), encoding="utf-8")
    for f in ["volume.csv", "volume-11kW.csv"]:
        shutil.copy(RAWDIR / "urbanev" / f, DATA / "urbanev" / f)
    wx = RAWDIR / "weather" / "openmeteo_jma_gsm_shanghai_2024.json"
    if wx.exists():
        shutil.copy(wx, DATA / wx.name)
    else:
        print("weather file missing: run python data/fetch_weather.py (needed only by the weather-feature models)")
    print("data ready:", sorted(p.name for p in DATA.rglob("*") if p.is_file()))
