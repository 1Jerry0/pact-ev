"""其他真实分时电价下的最优面与选解（A7 小时模型、锚定 18 组、LightGBM 均值预测、评估期 275 天）。
电价形状换成别的真实分时电价，再按月缩放到与上海当月电度电价同一日均值（只改形状不改水平）；缺口罚价仍按上海当日最高总价，以便只隔离电价形状的影响。
  shanghai  原上海分时电价（核对：应与正式结果一致）
  guangdong 广东（粤发改价格〔2021〕331号）：谷 0–8 时 0.38，平 8–10、12–14、19–24 时 1，峰 10–12、14–19 时 1.7；7–9 月 11–12、15–17 时尖峰 = 峰 ×1.25
  pge_ev2a  美国 PG&E EV2-A（2026-03 起）：谷 0–15 时 0.22558；平 15–16、21–24 时 夏 0.42760 / 冬 0.39428；峰 16–21 时 夏 0.53809 / 冬 0.41099（夏 = 6–9 月）
  two_block 两段制：谷 22–6 时 1，其余 2
报告：最优面宽度、默认解在面上的位置、各选解规则的总遗憾（相对默认解）、中点相对默认的逐日遗憾降幅。用法：python exp_tariffs.py"""
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
import sim
from fastlp import FastBss
from exp_packs_value import rel_ci

OUT = HERE.parents[1] / "results"
S = cfg.BSS
GRID = [(c, p, m) for c in (4, 6, 8) for p in (2.5, 4.0) for m in (1, 3, 10)]
H24 = np.arange(24)


def shape(name, month):
    if name == "guangdong":
        s = np.where(H24 < 8, 0.38, 1.0)
        peak = ((H24 >= 10) & (H24 < 12)) | ((H24 >= 14) & (H24 < 19))
        s = np.where(peak, 1.7, s)
        if month in (7, 8, 9):
            crit = (H24 == 11) | ((H24 >= 15) & (H24 < 17))
            s = np.where(crit, 1.7 * 1.25, s)
        return s
    if name == "pge_ev2a":
        summer = month in (6, 7, 8, 9)
        s = np.full(24, 0.22558)
        part = (H24 == 15) | (H24 >= 21)
        s = np.where(part, 0.42760 if summer else 0.39428, s)
        s = np.where((H24 >= 16) & (H24 < 21), 0.53809 if summer else 0.41099, s)
        return s
    if name == "two_block":
        return np.where((H24 >= 22) | (H24 < 6), 1.0, 2.0)
    raise ValueError(name)


def price_of(name, month):
    base = D.price(S, month)
    if name == "shanghai":
        return base
    s = shape(name, month)
    return s * base.mean() / s.mean()


def task(d, y, f, tot, hmean):
    rows = []
    for name in ("shanghai", "guangdong", "pge_ev2a", "two_block"):
        price = price_of(name, d.month)
        for par in GRID:
            P = sim.Problem("bss", y, price, par, dict(hmean=hmean, tot_max=tot))
            Fb = FastBss(P)
            st = {}
            for tie in ("default", "min", "mid", "max"):
                g, s, u, fc = Fb.plan(f, tie)
                cost = sim.bss_exec(g, s, fc, P.y, P.price, P.pen, P.cap, P.pmax, "current", u)[0]
                st[tie] = (cost - P.opt, float(s.sum()))
            span = st["max"][1] - st["min"][1]
            pos = (st["default"][1] - st["min"][1]) / span if span > 1e-6 * P.cap else np.nan
            rows.append((d, name, str(par), st["default"][0], st["min"][0], st["mid"][0], st["max"][0], span / (25 * P.cap), pos, P.opt))
    return rows


if __name__ == "__main__":
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
    G = F[F.series == S].set_index("t")["mean"]
    Tk = []
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y, f = H[S].reindex(idx).to_numpy(), G.reindex(idx).to_numpy()
        if np.isfinite(y).all() and np.isfinite(f).all():
            Tk.append((d, y, np.maximum(f, 0), float(D.price("A7", d.month, "total").max()), hmean))
    res = Parallel(n_jobs=int(os.environ.get("NJ", 7)), batch_size=2, verbose=2)(delayed(task)(*t) for t in Tk)
    R = pd.DataFrame([r for rr in res for r in rr], columns=["day", "tariff", "cfg", "reg_default", "reg_min", "reg_mid", "reg_max", "width", "pos", "opt"])
    R.to_parquet(OUT / "exp_tariffs.parquet")
    R["mult"] = R.cfg.map(lambda s: eval(s)[2])
    lines = ["LightGBM，锚定 18 组，275 天；遗憾为相对默认解的总遗憾比；中点 vs 默认为逐日合计遗憾的降幅（%，95% 区间）"]
    for name, g in R.groupby("tariff", sort=False):
        dd = g.groupby("day")[["reg_default", "reg_mid", "reg_max", "reg_min"]].sum()
        m, lo, hi = rel_ci(dd.reg_mid.to_numpy(), dd.reg_default.to_numpy())
        s = (f"[{name}] 面宽中位 {100 * g.width.median():.1f}% 库容；默认解位置中位 {g.pos.median():.2f}（底部 <0.05 占 {100 * (g.pos < 0.05).mean():.0f}%）；"
             f"默认解遗憾 / 完全信息成本 {g.reg_default.sum() / g.opt.sum():.3f}；中点 vs 默认 {100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]；")
        for mult, h in g.groupby("mult"):
            b = h.reg_default.sum()
            s += f" ×{mult}: 最小 {h.reg_min.sum() / b:.2f} 中点 {h.reg_mid.sum() / b:.2f} 最大 {h.reg_max.sum() / b:.2f}；"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_tariffs.txt").write_text(txt, encoding="utf-8")
    print(txt)
