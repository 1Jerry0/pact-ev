"""A8：预测换成近 28 天同小时经验分位数（seasonal_q）后，调参管线相对规则还剩多少（与 exp_bl_more.py a8 同设定，15 块电池各占一个充电位，×3 / ×10，1 小时充满）。
用法：python exp_bl_a8_profile.py"""
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
from exp_bl import task as bl_task, QL, ETA
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
RAW = ROOT / "data" / "raw" / "mp_evdata"
ST = "A8"
p1 = 1.0 / ETA
GRID = [((11, 12.0, 3), p1), ((11, 12.0, 10), p1)]

if __name__ == "__main__":
    d0 = pd.read_excel(RAW / "charging_session.xlsx", sheet_name=ST)
    d0.columns = ["conn", "disc", "energy"]
    d0["disc"] = pd.to_datetime(d0.disc, errors="coerce")
    a = d0[d0.disc.notna()].sort_values("disc")
    a["day"], a["step"] = a.disc.dt.normalize(), a.disc.dt.hour * 4 + a.disc.dt.minute // 15
    EV = {d: g.step.to_numpy() for d, g in a.groupby("day")}
    H = D.mp_hourly()
    hmean = float(H[ST].loc[:cfg.MP_DEV_END].mean())
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_seasonal_q.parquet")
    FQ = FQ[FQ.series == ST].set_index("t")
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, "2024-10-31", freq="D"):
        ix = pd.date_range(d, periods=24, freq="h")
        yy = H[ST].reindex(ix).to_numpy()
        f, Q = FQ["mean"].reindex(ix).to_numpy(), FQ[cfg.QC].reindex(ix).to_numpy()
        if not (np.isfinite(yy).all() and np.isfinite(f).all() and np.isfinite(Q).all()):
            continue
        fcs = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        ev = EV.get(d, np.array([], int))
        Tk.append((d, ev, np.ones(len(ev)), yy, D.price(ST, d.month), HK(D.price(ST, d.month, "total")), hmean, fcs, 1.0, GRID, False, False))
    res = Parallel(n_jobs=8, batch_size=1, verbose=1)(delayed(bl_task)(*t) for t in Tk)
    SQ = pd.DataFrame([r for _, rr in res for r in rr], columns=["day", "cfg", "pc", "n", "cand", "cost", "short", "width", "plan_cost"])
    SQ.to_parquet(OUT / "bl_a8_profile.parquet")
    RU = pd.read_parquet(OUT / "bl_rule_a8.parquet")
    FC = pd.read_parquet(OUT / "bl_fc_a8.parquet")
    lines = []
    for (c, pc), g in SQ.groupby(["cfg", "pc"], sort=False):
        Mc = RU[(RU.cfg == c) & (RU.pc == pc)].pivot(index="day", columns="th", values="cost").sort_index()
        Fl = FC[(FC.cfg == c) & (FC.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
        Fq = g.pivot(index="day", columns="cand", values="cost").sort_index()
        lgb = [k for k in Fl.columns if not k.startswith("pi@")]
        days = Fq.index[Fq.index >= START].intersection(Fl.index).intersection(Mc.index)
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        rule = np.array([Mc.loc[x, pick(Mc, x, k0)] for x in days])
        lg = np.array([Fl.loc[x, pick(Fl[lgb], x, "mean@0|mid")] for x in days])
        sq = np.array([Fq.loc[x, pick(Fq, x, "mean@0|mid")] for x in days])
        s = f"A8 {c} {pc:.2f}（{len(days)} 天）：相对规则 %："
        for lab, x, y in (("近期曲线管线", sq, rule), ("LightGBM 管线", lg, rule), ("LightGBM vs 近期曲线", lg, sq)):
            m, lo, hi = rel_ci(x, y)
            s += f" {lab} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_a8_profile.txt").write_text(txt, encoding="utf-8")
    print(txt)
