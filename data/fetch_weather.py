"""Download the historical JMA GSM weather forecasts for Shanghai (2024) from the Open-Meteo Previous Runs API
(https://open-meteo.com/en/docs/previous-runs-api; data licence CC BY 4.0) into data/raw/weather/.
The *_previous_day1/2 fields are the forecasts issued one and two days earlier, so that only information available
at forecast issue time is used. These data feed only the weather-feature LightGBM variants of the formal run,
which are not part of the paper's model pool. Usage: python fetch_weather.py"""
import json
import pathlib
import urllib.parse
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent / "raw" / "weather"
OUT.mkdir(parents=True, exist_ok=True)
VARS = ["temperature_2m", "precipitation", "relative_humidity_2m"]
params = {
    "latitude": 31.0, "longitude": 121.5, "start_date": "2024-01-01", "end_date": "2024-12-31",
    "hourly": ",".join(VARS + [f"{v}_previous_day{k}" for v in VARS for k in (1, 2)]),
    "models": "jma_gsm", "timezone": "Asia/Shanghai",
}
url = "https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
with urllib.request.urlopen(url, timeout=120) as f:
    data = json.loads(f.read().decode("utf-8"))
(OUT / "openmeteo_jma_gsm_shanghai_2024.json").write_text(json.dumps(data), encoding="utf-8")
print("saved", len(data["hourly"]["time"]), "hours to", OUT)
