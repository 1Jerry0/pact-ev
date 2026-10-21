"""深圳 60 区：预测换成近 28 天同小时经验分位数（seasonal_q，校准后）的调参管线，与 LightGBM 管线、滚动规则比较（每日重启口径，与 exp_bl_uev.py 同设定、同事件拆法）。
用法：python exp_bl_uev_profile.py；python exp_bl_uev_profile.py ana"""
import os
import pathlib
import sys
import zlib
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
from exp_bl import task, QL
from exp_bl_uev import GRID, EBAR
from exp_packs_value import pick, rel_ci

OUT = HERE.parents[2] / "results" / "profit_basis"


def ztask(z, hmean, days):
    rng = np.random.default_rng(zlib.crc32(str(z).encode()))
    rows = []
    for d, y, price, tot, f, Q in days:
        n = np.diff(np.r_[0.0, np.round(np.cumsum(y) / EBAR)]).astype(int)
        steps = np.sort(np.concatenate([4 * h + rng.integers(0, 4, n[h]) for h in range(24)])) if n.sum() else np.array([], int)
        if f is None:
            continue
        fcs = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        _, b = task(d, steps.astype(int), np.full(len(steps), EBAR), y, price, tot, hmean, fcs, EBAR, GRID, rule=False, pi=False)
        rows += [(z, *r[:3], r[4], r[5]) for r in b]
    return rows


def analyse():
    SQ = pd.read_parquet(OUT / "bl_uev_profile.parquet")
    RU = pd.read_parquet(OUT / "bl_uev_rule.parquet")
    FC = pd.read_parquet(OUT / "bl_uev_fc.parquet")
    Z = pd.read_csv(OUT / "bl_uev_zones.csv")
    Z["zone"] = Z.zone.astype(str)
    Z = Z.set_index("zone")
    lines, res = [], []
    daily = {}
    for (z, pc), g in SQ.groupby(["zone", "pc"], sort=False):
        r = RU[(RU.zone == z) & (RU.pc == pc)]
        Mc = pd.DataFrame(np.vstack(r.costs.to_list()), index=pd.DatetimeIndex(r.day)).sort_index()
        Fl = FC[(FC.zone == z) & (FC.pc == pc)].pivot(index="day", columns="cand", values="cost").sort_index()
        Fq = g.pivot(index="day", columns="cand", values="cost").sort_index()
        lgb = [k for k in Fl.columns if not k.startswith("pi@")]
        days = [d for d in Fq.index if d >= cfg.UEV_EVAL_START and d in Fl.index]
        k0 = Mc[Mc.index <= cfg.UEV_DEV_END].sum().idxmin()
        rule = np.array([Mc.loc[d, pick(Mc, d, k0)] for d in days])
        lg = np.array([Fl.loc[d, pick(Fl[lgb], d, "mean@0|mid")] for d in days])
        sq = np.array([Fq.loc[d, pick(Fq, d, "mean@0|mid")] for d in days])
        res.append(dict(zone=z, pc=pc, hmean=Z.loc[z, "hmean"], cv=Z.loc[z, "cv"], rule=rule.sum(), lgb=lg.sum(), sq=sq.sum()))
        for nm, x in (("rule", rule), ("lgb", lg), ("sq", sq)):
            daily.setdefault((pc, nm), []).append(pd.Series(x / rule.mean(), index=days))
    T = pd.DataFrame(res)
    for pc, g in T.groupby("pc"):
        d_ = {nm: pd.concat(daily[(pc, nm)], axis=1).mean(axis=1) for nm in ("rule", "lgb", "sq")}
        s = f"{pc:.0f} kW，{len(g)} 区（各区等权）："
        for lab, a, b in (("近期曲线管线 vs 规则", "sq", "rule"), ("LightGBM 管线 vs 规则", "lgb", "rule"), ("LightGBM vs 近期曲线", "lgb", "sq")):
            m, lo, hi = rel_ci(d_[a].to_numpy(), d_[b].to_numpy())
            s += f" {lab} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
        gap = 100 * (g.lgb / g.sq - 1)
        s += f" 各区 LightGBM vs 近期曲线 中位 {gap.median():+.1f}%，LightGBM 更便宜的区占 {100 * (gap < 0).mean():.0f}%；≥200 kWh/h 的区中位 {gap[g.hmean >= 200].median():+.1f}%，<100 的区 {gap[g.hmean < 100].median():+.1f}%"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_uev_profile.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    V = D.uev_volume()
    sel = pd.read_csv(OUT / "bl_uev_zones.csv")
    sel["zone"] = sel.zone.astype(str)
    sel = sel.set_index("zone")
    F = pd.read_parquet(cfg.OUT / "fc" / "uev_seasonal_q_cal.parquet")
    F = F[F.series.astype(str).isin(sel.index)]
    PR = {mo: (D.price("A1", mo), HK(D.price("A1", mo, "total"))) for mo in (9, 10, 11, 12, 1, 2)}
    Tk = []
    for z in sel.index:
        g = F[F.series.astype(str) == z].set_index("t")
        days = []
        for d in pd.date_range("2022-09-01", cfg.UEV_FC_END, freq="D"):
            idx = pd.date_range(d, periods=24, freq="h")
            y = V[z].reindex(idx).to_numpy()
            if not np.isfinite(y).all():
                continue
            f = Q = None
            if d >= pd.Timestamp("2022-10-01"):
                f, Q = g["mean"].reindex(idx).to_numpy(), g[cfg.QC].reindex(idx).to_numpy()
                if not (np.isfinite(f).all() and np.isfinite(Q).all()):
                    f = Q = None
            days.append((d, y, *PR[d.month], f, Q))
        Tk.append((z, float(sel.loc[z, "hmean"]), days))
    print(len(Tk), "zones", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 8)), batch_size=1, verbose=5)(delayed(ztask)(*t) for t in Tk)
    pd.DataFrame([r for rr in res for r in rr], columns=["zone", "day", "cfg", "pc", "cand", "cost"]).to_parquet(OUT / "bl_uev_profile.parquet")
    analyse()
