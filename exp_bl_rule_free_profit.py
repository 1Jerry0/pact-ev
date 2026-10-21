"""规则阈值不加 θ_p ≤ θ_s 约束时是否更强（利润口径版；电池级、6 h、2.5×、12 / 25 kW；上海与另外三种电价）。
缺供计价与 exp_bl_profit.py 相同：上海取 A7 逐小时零售总价，其他电价取“该电价 + A7 同小时服务费”。
补算 θ_p > θ_s 的 55 对阈值，与 blp_main_rule / blp_tariffs_rule 中原 66 对合并后滚动选（D−29…D−2），对比原规则与两条管线。
输出：结果\blp_rule_free.parquet、结果\blp_rule_free.txt。用法：python exp_bl_rule_free_profit.py"""
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
from exp_bl import nsteps, load_events, NS  # noqa: F401
from exp_tariffs import price_of
from exp_packs_value import pick, rel_ci
from exp_bl_profit import run_day_split, TIEBREAK

OUT = ROOT / "results"
S = cfg.BSS
GRID = [((6, 2.5), 12.0), ((6, 2.5), 25.0)]
TARIFFS = ("shanghai", "guangdong", "pge_ev2a", "two_block")
EXTRA = [(a / 10, b / 10) for a in range(11) for b in range(11) if b > a]
TUNE0 = pd.Timestamp("2024-05-01")


def task(d, ev_step, ev_e, hmean, ebar):
    rows = []
    ee = np.minimum(ev_e, 1.6 * ebar)
    for name in TARIFFS:
        if name == "shanghai":
            price = D.price(S, d.month)
            kap = D.price("A7", d.month, "total") * TIEBREAK
        else:
            price = price_of(name, d.month)
            kap = (price + D.price("A7", d.month, "total") - D.price("A7", d.month)) * TIEBREAK
        cls = np.repeat(np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6)), 4)
        for (ch, pf), pc in GRID:
            cap, pmax = ch * hmean, pf * hmean
            K, C = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1)
            for j, (a, b) in enumerate(EXTRA):
                e, u, p = run_day_split(ev_step, ee, price, kap, K, C, pc, ebar, np.where(cls == 0, K, np.where(cls == 1, a * K, b * K)))
                rows.append((name, d, f"({ch}, {pf})", pc, 100 + j, e + p))
    return rows


if __name__ == "__main__":
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    Tk = [(d, *EV[d], hmean, ebar) for d in pd.date_range("2024-01-15", cfg.MP_FC_END, freq="D")
          if d in EV and np.isfinite(H[S].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()).all()]
    res = Parallel(n_jobs=int(os.environ.get("NJ", 4)), batch_size=2, verbose=2)(delayed(task)(*t) for t in Tk)
    X = pd.DataFrame([r for rr in res for r in rr], columns=["tariff", "day", "cfg", "pc", "th", "cost"])
    X.to_parquet(OUT / "blp_rule_free.parquet")
    RU = pd.concat([pd.read_parquet(OUT / "blp_main_rule.parquet"), pd.read_parquet(OUT / "blp_tariffs_rule.parquet")])
    FC = pd.concat([pd.read_parquet(OUT / "blp_main_fc.parquet"), pd.read_parquet(OUT / "blp_tariffs_fc.parquet")])
    RU["cost"], FC["cost"] = RU.E + RU.Pen, FC.E + FC.Pen
    FC["k"] = FC.model + ":" + FC.cand
    lines = []
    for (name, c, pc), g in X.groupby(["tariff", "cfg", "pc"], sort=False):
        r0 = RU[(RU.tariff == name) & (RU.cfg == c) & (RU.pc == pc) & (RU.th >= 0)]
        M0 = r0.pivot(index="day", columns="th", values="cost").sort_index()
        M1 = pd.concat([M0, g.pivot(index="day", columns="th", values="cost").sort_index()], axis=1).loc[M0.index]
        Fc = FC[(FC.tariff == name) & (FC.cfg == c) & (FC.pc == pc)].pivot(index="day", columns="k", values="cost").sort_index()
        days = Fc.index[Fc.index >= TUNE0].intersection(M0.index)
        k0 = M0[M0.index < cfg.MP_EVAL_START].sum().idxmin()
        k1 = M1[M1.index < cfg.MP_EVAL_START].sum().idxmin()
        rule0 = np.array([M0.loc[x, pick(M0, x, k0)] for x in days])
        rule1 = np.array([M1.loc[x, pick(M1, x, k1)] for x in days])
        cl = [k for k in Fc.columns if k.startswith("lgb_local:")]
        cs = [k for k in Fc.columns if k.startswith("seasonal_q:")]
        lg = np.array([Fc.loc[x, pick(Fc[cl], x, "lgb_local:mean@0|mid")] for x in days])
        sq = np.array([Fc.loc[x, pick(Fc[cs], x, "seasonal_q:mean@0|mid")] for x in days])
        share = float((pd.Series([pick(M1, x, k1) for x in days]) >= 100).mean())
        f = lambda t: f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"
        lines.append(f"[{name}] {c} {pc:.0f} kW：不加约束的规则 vs 原规则 {f(rel_ci(rule1, rule0))}（选到 θp>θs 的天占 {100 * share:.0f}%）；"
                     f"原规则基准：LightGBM 管线 {f(rel_ci(lg, rule0))}，近期曲线 {f(rel_ci(sq, rule0))}；"
                     f"不加约束的规则基准：LightGBM 管线 {f(rel_ci(lg, rule1))}，近期曲线 {f(rel_ci(sq, rule1))}")
    txt = "\n".join(lines)
    (OUT / "blp_rule_free.txt").write_text(txt, encoding="utf-8")
    print(txt)
