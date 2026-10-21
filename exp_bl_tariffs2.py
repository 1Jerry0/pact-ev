"""exp_bl_tariffs 的补充：把预测换成"近 28 天同小时经验分位数"（seasonal_q，不含任何当日信息的历史日内曲线），
看其他电价下调参管线相对规则的优势，有多少来自"按日内曲线做计划"本身、多少来自当天的预测。四种电价（含上海），6 h、2.5×、×3，12 / 25 kW。
用法：python exp_bl_tariffs2.py；python exp_bl_tariffs2.py ana"""
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
from exp_tariffs import price_of
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[1] / "results"
S = cfg.BSS
GRID = [((6, 2.5, 3), 12.0), ((6, 2.5, 3), 25.0)]
TARIFFS = ("shanghai", "guangdong", "pge_ev2a", "two_block")


def task(d, ev_step, ev_e, y, tot, hmean, fcs, ebar):
    out = []
    for name in TARIFFS:
        _, rows = bl_task(d, ev_step, ev_e, y, price_of(name, d.month), tot, hmean, fcs, ebar, GRID, rule=False, pi=False)
        out += [(name, *r) for r in rows]
    return out


def analyse():
    SQ = pd.read_parquet(OUT / "bl_tariffs_sq.parquet")
    RU = pd.concat([pd.read_parquet(OUT / "bl_tariffs_rule.parquet"), pd.read_parquet(OUT / "bl_rule.parquet").assign(tariff="shanghai")])
    FC = pd.concat([pd.read_parquet(OUT / "bl_tariffs_fc.parquet"), pd.read_parquet(OUT / "bl_fc.parquet").assign(tariff="shanghai")])
    lines = []
    for (name, c, pc), g in SQ.groupby(["tariff", "cfg", "pc"], sort=False):
        ru = RU[(RU.tariff == name) & (RU.cfg == c) & (RU.pc == pc)]
        Mc = ru.pivot(index="day", columns="th", values="cost").sort_index()
        Fq = g.pivot(index="day", columns="cand", values="cost").sort_index()
        h = FC[(FC.tariff == name) & (FC.cfg == c) & (FC.pc == pc)]
        Fl = h.pivot(index="day", columns="cand", values="cost").sort_index()
        lgb = [k for k in Fl.columns if not k.startswith("pi@")]
        days = Fq.index[(Fq.index >= START)].intersection(Fl.index)
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        rule = np.array([Mc.loc[x, pick(Mc, x, k0)] for x in days])
        sq = np.array([Fq.loc[x, pick(Fq, x, "mean@0|mid")] for x in days])
        lg = np.array([Fl.loc[x, pick(Fl[lgb], x, "mean@0|mid")] for x in days])
        s = f"[{name}] {c} {pc:.0f} kW（{len(days)} 天）：相对规则 %："
        for lab, a, b in (("季节分位数管线", sq, rule), ("LightGBM 管线", lg, rule), ("LightGBM vs 季节分位数", lg, sq)):
            m, lo, hi = rel_ci(a, b)
            s += f" {lab} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_tariffs2.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
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
        Tk.append((d, *EV[d], y, float(D.price("A7", d.month, "total").max()), hmean, fcs, ebar))
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 8)), batch_size=1, verbose=5)(delayed(task)(*t) for t in Tk)
    pd.DataFrame([r for rr in res for r in rr], columns=["tariff", "day", "cfg", "pc", "n", "cand", "cost", "short", "width", "plan_cost"]).to_parquet(OUT / "bl_tariffs_sq.parquet")
    analyse()
