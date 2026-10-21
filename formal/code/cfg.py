"""正式实验配置（预注册冻结）。所有路径相对项目根目录；SMOKE=1 时用小子集做冒烟测试。"""
import os
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / ("out_smoke" if os.environ.get("SMOKE") == "1" else "out")
LOGS = ROOT / "logs"
for p in (OUT, LOGS, OUT / "fc", OUT / "dec", OUT / "mech", OUT / "uev", OUT / "agg", OUT / "tables"):
    p.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"
SEED = 20261003

# ---------------------------------------------------------------- MP-EVData
POWER = ["A1", "A2", "A3", "A4", "A5", "A6", "A9"]       # 常规站（储能调度、申报）
BSS = "A7_kwh"                                          # 换电库存（补能量，清洗后按断开时刻）
EXTRA = ["A8"]                                          # 只做精度（间歇需求）
MP_SERIES = POWER + [BSS] + EXTRA
PRICE_PROXY = {"A4": "A3", "A9": "A1", "A7_kwh": "A7", "A8": "A8"}
MP_FC_START, MP_FC_END = pd.Timestamp("2024-02-01"), pd.Timestamp("2024-12-31")   # 2–3 月为校准与情景历史的预热
MP_EVAL_START = pd.Timestamp("2024-04-01")
MP_DEV_END = pd.Timestamp("2024-03-31 23:00")
MP_REFITS = pd.date_range("2024-02-01", "2024-12-31", freq="7D")                    # 每 7 天扩张重训（按月重训跟不上水平变化）

# ---------------------------------------------------------------- UrbanEV
UEV_FC_START, UEV_FC_END = pd.Timestamp("2022-10-01"), pd.Timestamp("2023-02-28")
UEV_EVAL_START = pd.Timestamp("2022-11-01")
UEV_DEV_END = pd.Timestamp("2022-10-31 23:00")
UEV_REFITS = pd.date_range("2022-10-01", "2023-02-01", freq="MS")
UEV_COVID = (pd.Timestamp("2022-12-01"), pd.Timestamp("2023-01-31"))                # 防疫调整期标记
UEV_PRICE_MONTH = {10: 10, 11: 11, 12: 12, 1: 1, 2: 2}                              # 统一分时电价（上海 A1 电度电价，同月）

HOLIDAYS = pd.to_datetime(
    ["2022-09-10", "2022-09-11", "2022-09-12"] + [f"2022-10-0{d}" for d in range(1, 8)]
    + ["2022-12-31", "2023-01-01", "2023-01-02"] + [f"2023-01-{d}" for d in range(21, 28)]
    + ["2024-01-01"] + [f"2024-02-{d:02d}" for d in range(10, 18)]
    + ["2024-04-04", "2024-04-05", "2024-04-06"] + [f"2024-05-0{d}" for d in range(1, 6)]
    + ["2024-06-10", "2024-09-15", "2024-09-16", "2024-09-17"] + [f"2024-10-0{d}" for d in range(1, 8)])
MAKEUP = pd.to_datetime(["2022-10-08", "2022-10-09", "2023-01-28", "2023-01-29",
                         "2024-02-04", "2024-02-18", "2024-04-07", "2024-04-28", "2024-05-11",
                         "2024-09-14", "2024-09-29", "2024-10-12"])

# ---------------------------------------------------------------- 预测
QS = np.round(np.arange(0.05, 0.951, 0.05), 2)
QC = [f"q{int(round(t * 100)):02d}" for t in QS]
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=10, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, num_threads=1, deterministic=True, force_row_wise=True, verbose=-1,
           random_state=7)                                                          # 调参期小网格选定后冻结
LQR_WINDOW_DAYS = None                                                              # 与其他模型一致，扩张窗口（8 周窗口在含春节时明显变差）
CAL_WINDOW, SCEN_WINDOW = 28, 28                                                    # 在线校准与情景构造的历史天数
MP_MODELS_STAT = ["naive", "naive7", "seasonal_q", "lqr", "lgb_local", "lgb_global", "lgb_wx", "lgb_wx_perfect"]
MP_MODELS_TSFM = ["chronos2", "chronos2_cov", "chronos2_wx", "timesfm3", "timesfm3_cov"]
COMBO5 = ["lgb_local", "lqr", "seasonal_q", "chronos2_cov", "timesfm3_cov"]
UEV_MODELS_STAT = ["naive", "naive7", "seasonal_q", "lqr", "lgb_global"]
UEV_MODELS_TSFM = ["chronos2_cov", "timesfm3_cov"]
UEV_COMBO5 = ["lgb_global", "lqr", "seasonal_q", "chronos2_cov", "timesfm3_cov"]

# ---------------------------------------------------------------- 决策
ETA = 0.95
TAUS = [0.5, 0.67, 0.83, 0.91]
BANDS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
ST_GRID = [(e, q, lam) for e in (0.1, 0.25, 0.5) for q in (0.3, 0.5, 0.7) for lam in (1.0, 3.0, 10.0)]
BSS_GRID = [(c, p, m) for c in (2, 4, 8) for p in (1.1, 1.3, 1.6) for m in (1, 3, 10)]
ST_REP = [g for g in ST_GRID if g[2] == 3.0]                                        # 代表性 9 组（MPC、开环、情景优化）
BSS_REP = [g for g in BSS_GRID if g[2] == 3]
SCHED_MODELS_REP = ["naive", "lgb_local", "chronos2_cov", "combo5"]                 # 执行方式对比的模型
SAA_MODELS = ["lgb_local", "chronos2_cov", "combo5"]
SAA_QUANTS = ["mean", 0.5, 0.6, 0.7, 0.8, 0.9]
UEV_E = [0.1, 0.25, 0.5]
UEV_LQ, UEV_LAM = 0.5, 3.0
SESOI_REGRET, SESOI_CRPS = 0.05, 0.03

# ---------------------------------------------------------------- 机制实验 E4
MECH_STATIONS = ["A1", "A3", "A7_kwh"]
MECH_SOURCE = "combo5"
MECH_SENS_SOURCES = ["lgb_local", "chronos2_cov"]
MECH_N_PER, MECH_THIN, MECH_BURN = 30, 30, 300
MECH_KAPPA = {"low": -40.0, "mid": 0.0, "high": 40.0}
MECH_MPC_EVERY, MECH_MPC_N = 4, 15                                                 # MPC 每 4 天抽 1 天，每档 15 个样本

if SMOKE:
    MP_SERIES = ["A1", "A3", "A7_kwh"]
    POWER = ["A1", "A3"]
    MP_FC_END = pd.Timestamp("2024-04-07")
    MP_REFITS = pd.date_range("2024-02-01", "2024-04-07", freq="7D")
    UEV_FC_END = pd.Timestamp("2022-11-05")
    UEV_REFITS = pd.date_range("2022-10-01", "2022-11-01", freq="MS")
    ST_GRID = [(0.25, 0.5, 3.0), (0.1, 0.3, 10.0)]
    BSS_GRID = [(4, 1.3, 3), (2, 1.1, 10)]
    ST_REP, BSS_REP = ST_GRID[:1], BSS_GRID[:1]
    UEV_E = [0.25]
    MECH_N_PER, MECH_MPC_N = 6, 3
