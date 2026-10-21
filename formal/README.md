# Stage 1: pre-registered forecasting and decision run

All settings are fixed in `code/cfg.py` (frozen before the formal run). Paths are relative to this folder.

```bash
python code/prepare_data.py      # once, after placing the raw data in ../data/raw (see ../data/README.md)
bash setup.sh                    # installs requirements and downloads the Chronos-2 and TimesFM-3.0 weights
nohup bash run_all.sh > logs/run_all.out 2>&1 &
```

`run_all.sh` runs, in order: the foundation models on the GPU (`fc_tsfm.py`) in parallel with the statistical and machine-learning models on the CPU (`fc_stat.py`); combination and online calibration (`fc_post.py`); the MP-EVData decision models (`dec_mp.py`); the mechanism experiment (`mech.py`); the UrbanEV decisions (`uev.py`); aggregation (`agg.py`); integrity checks (`check.py`: non-negative regret, zero regret for perfect forecasts, invariance of negative controls, complete forecasts); and the pre-registered analysis (`analysis.py`). A run that stops can be restarted; finished parts are skipped. `SMOKE=1 bash run_all.sh` runs a small subset into `out_smoke/` in a few minutes.

Hardware used: Linux, 25 CPU cores, 60 GB memory, one RTX 4090 (the foundation models need 2–6 GB of GPU memory); about 2.5 hours. Outputs: `out/fc/` (forecasts, about 2 GB), `out/dec/`, `out/mech/`, `out/uev/`, `out/agg/`, `out/tables/` (summary in `out/tables/summary.md`).

Comments and log messages are in Chinese.
