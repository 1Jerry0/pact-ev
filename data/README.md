# Data

The raw data are public and are not redistributed here. Place them as follows:

```
data/raw/mp_evdata/charging_session.xlsx
data/raw/mp_evdata/station_level_load_profile_15min.xlsx   (any spelling, e.g. station-level_load_Profile_15min.xlsx)
data/raw/mp_evdata/price.xlsx
data/raw/urbanev/volume.csv
data/raw/urbanev/volume-11kW.csv
data/raw/weather/openmeteo_jma_gsm_shanghai_2024.json      (python data/fetch_weather.py)
```

- **MP-EVData**, version 3: https://doi.org/10.6084/m9.figshare.29882366 — Liu R, et al. *Scientific Data* 2026;13:924. https://doi.org/10.1038/s41597-026-07273-5. The study uses stations A1–A9 (2024).
- **UrbanEV**: https://doi.org/10.5061/dryad.np5hqc04z — Li H, et al. *Scientific Data* 2025;12:523. https://doi.org/10.1038/s41597-025-04874-4. The study uses the hourly zone energy files `volume.csv` and `volume-11kW.csv`.
- **Weather**: `python data/fetch_weather.py` downloads JMA GSM historical forecasts for Shanghai from the Open-Meteo Previous Runs API (CC BY 4.0). They feed only weather-feature models of the formal run that are not part of the paper's model pool, but the formal run expects the file.

Then run `python formal/code/prepare_data.py`, which writes `formal/data/mp_hourly.parquet`, `formal/data/prices.json` and copies of the UrbanEV and weather files.
