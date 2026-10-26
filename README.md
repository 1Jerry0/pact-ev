# Forecast error prices and equal-cost charging plans for battery-swapping and charging stations

Replication code for

> Conglin Ye, Youyuan Wang. *Forecast error prices and equal-cost charging plans: linking demand-forecast accuracy to the operating cost of battery-swapping and charging stations.* Manuscript under review, 2026.

The code reproduces every table and figure of the paper and its Supplementary Material from two public datasets. `PAPER_MAP.md` lists, for each table, figure and key number, the script that produces it and the output file to read.

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | Download instructions (`data/README.md`) and `fetch_weather.py`; raw files go to `data/raw/` (not distributed here) |
| `formal/` | Stage 1, the pre-registered forecasting and decision run: data preparation, nine forecasting models (seasonal naive to Chronos-2 and TimesFM-3.0), calibration, hourly decision models, pre-specified hypothesis tests |
| `scripts/` | Stage 2 analyses on the profit basis (hourly model, error prices, equal-cost plans, storage stations, figures and tables) and the original pre-specified shortfall pricing used for sensitivity analyses |
| `scripts/profit_basis/` | Stage 2 analyses whose shortfall cost is the hourly retail price (battery-level simulation, decision-focused forecasts, solvers, tariffs, Shenzhen zones, robustness checks) |
| `results/` | Outputs of stage 2 (created by `run_pipeline.py`) |

Code comments, console messages and the text summaries written to `results/` are in Chinese, the working language of the study; variable names, file names and this documentation are in English. Shortfall pricing is called the *profit basis* in the paper (`profit_basis` here); the pre-specified pricing priced shortfalls at 1, 3 or 10 times the day's highest retail price (the third element of a configuration tuple such as `(6, 2.5, 3)`; on the profit basis that element is a legacy label and the hourly retail price is used).

## Requirements

- Python 3.11 with the packages in `requirements.txt` (`pip install -r requirements.txt`).
- Stage 1 uses a GPU for the two foundation models (about 2–6 GB of memory) and benefits from many CPU cores; it was run on Linux with 25 cores and an RTX 4090 in about 2.5 hours (`formal/README.md`).
- Stage 2 is CPU only. The analyses of the paper took roughly 20–30 hours with 7 worker processes on a desktop computer; the most expensive steps are the battery-level simulations.
- Disk: about 3 GB for stage 1 outputs and 1 GB for stage 2 outputs.

## Reproducing the results

1. **Data.** Download MP-EVData and UrbanEV into `data/raw/` as described in `data/README.md`, and run `python data/fetch_weather.py`.
2. **Prepare inputs.** `python formal/code/prepare_data.py` writes `formal/data/`.
3. **Stage 1.** `bash formal/setup.sh` (installs the foundation-model weights) and `bash formal/run_all.sh`; outputs go to `formal/out/`. `SMOKE=1 bash formal/run_all.sh` runs a small smoke test in a few minutes.
4. **Stage 2.** `python run_pipeline.py` runs all analyses in order (`--list` shows the 112 steps, `--group` selects `prespecified`, `hourly`, `battery` or `report`, `--from N` resumes, `--jobs N` sets the number of worker processes). Logs go to `results/logs/`.
5. **Figures and tables.** The last group (`report`) writes the figures and the graphical abstract as submitted to `results/figures_ae/` (`scripts/fig_ae.py`; vector PDF and PNG), the same figures in an earlier layout to `results/figures/`, and the appendix tables to `results/appendix_profit.md`; `PAPER_MAP.md` gives the source of every other number.

Forecasting models are retrained on expanding windows with fixed seeds; LightGBM is run with `deterministic=True`. Foundation-model inference on a different GPU can change the last digits of the forecasts.

## Data

- **MP-EVData** (Shanghai, 10 stations, 2024): Liu R, et al. An AI-augmented dataset of multi-prototype electric vehicle charging load profiles in China. *Scientific Data* 2026;13:924. https://doi.org/10.1038/s41597-026-07273-5. Data: https://doi.org/10.6084/m9.figshare.29882366 (version 3).
- **UrbanEV** (Shenzhen, 275 zones, Sep 2022–Feb 2023): Li H, et al. UrbanEV: An open benchmark dataset for urban electric vehicle charging demand prediction. *Scientific Data* 2025;12:523. https://doi.org/10.1038/s41597-025-04874-4. Data: https://doi.org/10.5061/dryad.np5hqc04z.
- **Weather** (only used by weather-feature models that are not part of the paper's model pool): Open-Meteo Previous Runs API, JMA GSM, CC BY 4.0.

The datasets are not redistributed in this repository; please cite them when using the code.

## Licence

The code is released under the MIT licence (`LICENSE`). The datasets remain under the licences of their providers.

## Contact

Youyuan Wang (corresponding author), School of Economics and Management, Nanchang Hangkong University, Nanchang 330063, China.
