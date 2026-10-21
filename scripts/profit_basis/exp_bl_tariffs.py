"""电池级仿真在其他分时电价下的"预测 vs 滚动规则"与最优面（A7，6 h、2.5×、×3，单块 12 / 25 kW，LightGBM，每日重启口径）。
电价形状取自 exp_tariffs.shape（广东、PG&E EV2-A、两段制），按月缩放到上海电度电价均值；规则的三档阈值按电价等级（最低价 = 充满，其余两档）设定。
用法：python exp_bl_tariffs.py；python exp_bl_tariffs.py ana"""
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
from exp_bl import task as bl_task, load_events, QL
from exp_tariffs import price_of
from exp_packs import TH
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((6, 2.5, 3), 12.0), ((6, 2.5, 3), 25.0)]
TARIFFS = ("guangdong", "pge_ev2a", "two_block")


def task(d, ev_step, ev_e, y, tot, hmean, fcs, ebar):
    out = []
    for name in TARIFFS:
        price = price_of(name, d.month)
        ru, rows = bl_task(d, ev_step, ev_e, y, price, tot, hmean, fcs, ebar, GRID)
        out.append((name, ru, rows))
    return out


def analyse():
    RU = pd.read_parquet(OUT / "bl_tariffs_rule.parquet")
    FC = pd.read_parquet(OUT / "bl_tariffs_fc.parquet")
    lines = []
    for (name, c, pc), g in RU.groupby(["tariff", "cfg", "pc"], sort=False):
        Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
        h = FC[(FC.tariff == name) & (FC.cfg == c) & (FC.pc == pc)]
        Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
        lgb = [k for k in Fc.columns if not k.startswith("pi@")]
        pis = [k for k in Fc.columns if k.startswith("pi@")]
        days = Fc.index[Fc.index >= START]
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin() if (Mc.index < cfg.MP_EVAL_START).any() else Mc.columns[0]
        rule = np.array([Mc.loc[x, pick(Mc, x, k0)] for x in days])
        tun = np.array([Fc.loc[x, pick(Fc[lgb], x, "mean@0|mid")] for x in days])
        pit = np.array([Fc.loc[x, pick(Fc[pis], x, "pi@0|mid")] for x in days])
        full = g.drop_duplicates("day").set_index("day").full.loc[days].to_numpy()
        s = f"[{name}] {c} {pc:.0f} kW：规则 {rule.mean():.0f} 元/日；相对规则 %："
        for lab, a in (("调参管线", tun), ("完全信息（调参）", pit), ("默认解", Fc.loc[days, "mean@0|default"].to_numpy()), ("最大库存解", Fc.loc[days, "mean@0|max"].to_numpy()), ("换下即充", full)):
            m, lo, hi = rel_ci(a, rule)
            s += f" {lab} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
        m, lo, hi = rel_ci(Fc.loc[days, "mean@0|max"].to_numpy(), Fc.loc[days, "mean@0|min"].to_numpy())
        s += f" 最大 vs 最小 {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；最优面宽度中位 {h[h.cand == 'mean@0|mid'].width.median():.3f}"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_tariffs.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    Tk = []
    for d in pd.date_range("2024-03-01", cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all() or d not in EV:
            continue
        fcs = None
        if d >= cfg.MP_EVAL_START:
            f, Q = FQ["mean"].reindex(idx).to_numpy(), FQ[cfg.QC].reindex(idx).to_numpy()
            if not (np.isfinite(f).all() and np.isfinite(Q).all()):
                continue
            fcs = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        Tk.append((d, *EV[d], y, HK(D.price("A7", d.month, "total")), hmean, fcs, ebar))
    print(len(Tk), "tasks", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 8)), batch_size=1, verbose=5)(delayed(task)(*t) for t in Tk)
    RU = pd.DataFrame([(name, dd, c, pc, n, i, cc[i], uu[i], cf, uf, dem) for r in res for name, ru, _ in r for (dd, c, pc, n, cc, uu, cf, uf, dem) in ru for i in range(len(TH))],
                      columns=["tariff", "day", "cfg", "pc", "n", "th", "cost", "short", "full", "short_full", "dem"])
    RU.to_parquet(OUT / "bl_tariffs_rule.parquet")
    pd.DataFrame([(name, *row) for r in res for name, _, rows in r for row in rows], columns=["tariff", "day", "cfg", "pc", "n", "cand", "cost", "short", "width", "plan_cost"]).to_parquet(OUT / "bl_tariffs_fc.parquet")
    analyse()
