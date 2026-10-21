"""规则阈值不加 θ_p ≤ θ_s 约束时是否更强（电池级、6 h、2.5×、×3、12 / 25 kW；上海与另外三种电价）。
补算 θ_p > θ_s 的 55 对阈值，与原 66 对合并后滚动选（D−29…D−2），对比原规则与两条管线。用法：python exp_bl_rule_free.py"""
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
from exp_bl import run_day_bl, nsteps, load_events, NS
from exp_tariffs import price_of
from exp_packs import TH
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((6, 2.5, 3), 12.0), ((6, 2.5, 3), 25.0)]
TARIFFS = ("shanghai", "guangdong", "pge_ev2a", "two_block")
EXTRA = [(a / 10, b / 10) for a in range(11) for b in range(11) if b > a]


def task(d, ev_step, ev_e, tot, hmean, ebar):
    rows = []
    ee = np.minimum(ev_e, 1.6 * ebar)
    for name in TARIFFS:
        price = price_of(name, d.month)
        cls = np.repeat(np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6)), 4)
        for par, pc in GRID:
            cap, pmax = par[0] * hmean, par[1] * hmean
            pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
            K, C = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1)
            for j, (a, b) in enumerate(EXTRA):
                c, u = run_day_bl(ev_step, ee, price, pen, K, C, pc, ebar, np.where(cls == 0, K, np.where(cls == 1, a * K, b * K)))
                rows.append((name, d, str(par), pc, 100 + j, c))
    return rows


if __name__ == "__main__":
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    Tk = [(d, *EV[d], HK(D.price("A7", d.month, "total")), hmean, ebar) for d in pd.date_range("2024-01-15", cfg.MP_FC_END, freq="D")
          if d in EV and np.isfinite(H[S].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()).all()]
    res = Parallel(n_jobs=int(os.environ.get("NJ", 8)), batch_size=2, verbose=2)(delayed(task)(*t) for t in Tk)
    X = pd.DataFrame([r for rr in res for r in rr], columns=["tariff", "day", "cfg", "pc", "th", "cost"])
    X.to_parquet(OUT / "bl_rule_free.parquet")
    RU = pd.concat([pd.read_parquet(OUT / "bl_tariffs_rule.parquet")[["tariff", "day", "cfg", "pc", "th", "cost"]],
                    pd.read_parquet(OUT / "bl_rule.parquet").assign(tariff="shanghai")[["tariff", "day", "cfg", "pc", "th", "cost"]]])
    FC = pd.concat([pd.read_parquet(OUT / "bl_tariffs_fc.parquet"), pd.read_parquet(OUT / "bl_fc.parquet").assign(tariff="shanghai")])
    SQ = pd.read_parquet(OUT / "bl_tariffs_sq.parquet")
    lines = []
    for (name, c, pc), g in X.groupby(["tariff", "cfg", "pc"], sort=False):
        r0 = RU[(RU.tariff == name) & (RU.cfg == c) & (RU.pc == pc)]
        M0 = r0.pivot(index="day", columns="th", values="cost").sort_index()
        M1 = pd.concat([M0, g.pivot(index="day", columns="th", values="cost").sort_index()], axis=1).loc[M0.index]
        Fl = FC[(FC.tariff == name) & (FC.cfg == c) & (FC.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
        Fq = SQ[(SQ.tariff == name) & (SQ.cfg == c) & (SQ.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
        lgb = [k for k in Fl.columns if not k.startswith("pi@")]
        days = Fq.index[Fq.index >= START].intersection(Fl.index).intersection(M0.index)
        k0 = M0[M0.index < cfg.MP_EVAL_START].sum().idxmin()
        k1 = M1[M1.index < cfg.MP_EVAL_START].sum().idxmin()
        rule0 = np.array([M0.loc[x, pick(M0, x, k0)] for x in days])
        rule1 = np.array([M1.loc[x, pick(M1, x, k1)] for x in days])
        lg = np.array([Fl.loc[x, pick(Fl[lgb], x, "mean@0|mid")] for x in days])
        sq = np.array([Fq.loc[x, pick(Fq, x, "mean@0|mid")] for x in days])
        picks = pd.Series([pick(M1, x, k1) for x in days])
        share = float((picks >= 100).mean())
        s = f"[{name}] {c} {pc:.0f} kW：不加约束的规则 vs 原规则 "
        m, lo, hi = rel_ci(rule1, rule0)
        s += f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]（选到 θp>θs 的天占 {100 * share:.0f}%）；相对不加约束的规则：LightGBM 管线 "
        m, lo, hi = rel_ci(lg, rule1)
        s += f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]，近期曲线管线 "
        m, lo, hi = rel_ci(sq, rule1)
        s += f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_rule_free.txt").write_text(txt, encoding="utf-8")
    print(txt)
