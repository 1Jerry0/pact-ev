"""上海电价、最高罚价（×10）下，近期曲线管线与 LightGBM 管线的差（6 h、2.5×，12 / 25 kW；每日重启口径），补 exp_bl_tariffs2 只做了 ×3 的空缺。
用法：python exp_bl_profile_x10.py"""
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + str(HERE.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
from exp_bl import task as bl_task, load_events, QL
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[1] / "results"
S = cfg.BSS
GRID = [((6, 2.5, 10), 12.0), ((6, 2.5, 10), 25.0)]

if __name__ == "__main__":
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_seasonal_q.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        f, Q = FQ["mean"].reindex(idx).to_numpy(), FQ[cfg.QC].reindex(idx).to_numpy()
        if not (np.isfinite(y).all() and np.isfinite(f).all() and np.isfinite(Q).all()) or d not in EV:
            continue
        fcs = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        Tk.append((d, *EV[d], y, D.price(S, d.month), float(D.price("A7", d.month, "total").max()), hmean, fcs, ebar, GRID, False, False))
    res = Parallel(n_jobs=8, batch_size=1, verbose=1)(delayed(bl_task)(*t) for t in Tk)
    SQ = pd.DataFrame([r for _, rr in res for r in rr], columns=["day", "cfg", "pc", "n", "cand", "cost", "short", "width", "plan_cost"])
    SQ.to_parquet(OUT / "bl_profile_x10.parquet")
    RU = pd.read_parquet(OUT / "bl_rule.parquet")
    FC = pd.read_parquet(OUT / "bl_fc.parquet")
    lines = []
    for (c, pc), g in SQ.groupby(["cfg", "pc"], sort=False):
        Mc = RU[(RU.cfg == c) & (RU.pc == pc)].pivot(index="day", columns="th", values="cost").sort_index()
        Fl = FC[(FC.cfg == c) & (FC.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
        Fq = g.pivot(index="day", columns="cand", values="cost").sort_index()
        lgb = [k for k in Fl.columns if not k.startswith("pi@")]
        days = Fq.index[Fq.index >= START].intersection(Fl.index)
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        rule = np.array([Mc.loc[x, pick(Mc, x, k0)] for x in days])
        lg = np.array([Fl.loc[x, pick(Fl[lgb], x, "mean@0|mid")] for x in days])
        sq = np.array([Fq.loc[x, pick(Fq, x, "mean@0|mid")] for x in days])
        s = f"上海 {c} {pc:.0f} kW：相对规则 %："
        for lab, x, y in (("近期曲线管线", sq, rule), ("LightGBM 管线", lg, rule), ("LightGBM vs 近期曲线", lg, sq)):
            m, lo, hi = rel_ci(x, y)
            s += f" {lab} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_profile_x10.txt").write_text(txt, encoding="utf-8")
    print(txt)
