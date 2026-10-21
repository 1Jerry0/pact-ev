"""电池级 LP 计划的两项稳健性（沿用 exp_bl.py）：
conn：A7 换电事件改按接入时刻、需求序列按接入时刻汇总并重拟合 LightGBM 均值预测（同 exp_packs_conn.py），库容 6 小时 × {12, 18, 25, 50} kW 与 4 小时 × {12, 25} kW；
a8  ：第二座换电站 A8（单位：次；60 kW 充电机、单次约 56 kWh ≈ 1 小时充满），15 块电池各占一个充电位（11 h, 12×）与 8 块 / 2 个充电位（6 h, 2.5×），罚价 ×3 与 ×10，1 小时充电；另加 2 小时充电作对照。
用法：python exp_bl_more.py conn | a8"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + str(HERE.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
import fc_stat
from exp_conn import lgb_mean, RAW
from exp_packs import TH
from exp_bl import task, analyse, QL, ETA

OUT = HERE.parents[2] / "results" / "profit_basis"


def save(res, suf):
    RU = pd.DataFrame([(d, c, pc, n, i, cc[i], uu[i], cf, uf, dem) for b, _ in res for (d, c, pc, n, cc, uu, cf, uf, dem) in b for i in range(len(TH))],
                      columns=["day", "cfg", "pc", "n", "th", "cost", "short", "full", "short_full", "dem"])
    RU.to_parquet(OUT / f"bl_rule{suf}.parquet")
    pd.DataFrame([r for _, rr in res for r in rr], columns=["day", "cfg", "pc", "n", "cand", "cost", "short", "width", "plan_cost"]).to_parquet(OUT / f"bl_fc{suf}.parquet")
    analyse(suf)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "conn":
        S = cfg.BSS
        grid = [((6, 2.5, 3), pc) for pc in (12.0, 18.0, 25.0, 50.0)] + [((4, 2.5, 3), 12.0), ((4, 2.5, 3), 25.0)]
        d0 = pd.read_excel(RAW / "charging_session.xlsx", sheet_name="A7")
        d0.columns = ["conn", "disc", "energy"]
        d0["energy"] = pd.to_numeric(d0.energy, errors="coerce")
        d0["conn"], d0["disc"] = pd.to_datetime(d0.conn, errors="coerce"), pd.to_datetime(d0.disc, errors="coerce")
        a = d0[d0.energy.gt(0) & d0.disc.notna() & d0.conn.notna()].copy()
        onhour = (a.disc.dt.minute == 0) & (a.disc.dt.second == 0)
        mass = (a.disc >= "2024-07-26 10:00") & (a.disc < "2024-07-26 11:00")
        a = a[~onhour & ~mass].sort_values("conn")
        ebar = float(a.energy.mean())
        idx = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
        y = a.groupby(a.conn.dt.floor("h")).energy.sum().reindex(idx, fill_value=0.0)
        a["day"], a["step"] = a.conn.dt.normalize(), a.conn.dt.hour * 4 + a.conn.dt.minute // 15
        EV = {d: (g.step.to_numpy(), g.energy.to_numpy()) for d, g in a.groupby("day")}
        hmean = float(y.loc[:cfg.MP_DEV_END].mean())
        G = pd.concat(Parallel(n_jobs=6)(delayed(lgb_mean)(y, r0, r1) for r0, r1 in fc_stat.windows(cfg.MP_REFITS, cfg.MP_FC_END)))
        G = G[~G.index.duplicated()]
        Tk = []
        for d in pd.date_range("2024-01-15", cfg.MP_FC_END, freq="D"):
            ix = pd.date_range(d, periods=24, freq="h")
            yy = y.reindex(ix).to_numpy()
            if not np.isfinite(yy).all() or d not in EV:
                continue
            fcs = None
            if d >= cfg.MP_EVAL_START:
                f = G.reindex(ix).to_numpy()
                if not np.isfinite(f).all():
                    continue
                fcs = {"mean": f}
            Tk.append((d, *EV[d], yy, D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, fcs, ebar, grid))
        print(len(Tk), "tasks", flush=True)
        save(Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk), "_conn")
    else:
        ST = "A8"
        p1, p2 = 1.0 / ETA, 0.5 / ETA                             # 一块电池（1 个单位）1 小时 / 2 小时充满
        grid = [((11, 12.0, 3), p1), ((11, 12.0, 10), p1), ((6, 2.5, 3), p1), ((6, 2.5, 10), p1), ((11, 12.0, 3), p2), ((6, 2.5, 3), p2)]
        d0 = pd.read_excel(RAW / "charging_session.xlsx", sheet_name=ST)
        d0.columns = ["conn", "disc", "energy"]
        d0["disc"] = pd.to_datetime(d0.disc, errors="coerce")
        a = d0[d0.disc.notna()].sort_values("disc")
        a["day"], a["step"] = a.disc.dt.normalize(), a.disc.dt.hour * 4 + a.disc.dt.minute // 15
        EV = {d: g.step.to_numpy() for d, g in a.groupby("day")}
        H = D.mp_hourly()
        hmean = float(H[ST].loc[:cfg.MP_DEV_END].mean())
        FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
        FQ = FQ[FQ.series == ST].set_index("t")
        Tk = []
        for d in pd.date_range("2024-01-15", "2024-10-31", freq="D"):
            ix = pd.date_range(d, periods=24, freq="h")
            yy = H[ST].reindex(ix).to_numpy()
            if not np.isfinite(yy).all():
                continue
            fcs = None
            if d >= cfg.MP_EVAL_START:
                f, Q = FQ["mean"].reindex(ix).to_numpy(), FQ[cfg.QC].reindex(ix).to_numpy()
                if not (np.isfinite(f).all() and np.isfinite(Q).all()):
                    continue
                fcs = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
            ev = EV.get(d, np.array([], int))
            Tk.append((d, ev, np.ones(len(ev)), yy, D.price(ST, d.month), HK(D.price(ST, d.month, "total")), hmean, fcs, 1.0, grid))
        print(len(Tk), "tasks; hmean", round(hmean, 3), flush=True)
        save(Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk), "_a8")
