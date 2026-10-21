"""电池级 LP 计划的三项补充（沿用 exp_bl.py）：
extra ：LightGBM，再加 4 组配置（库容 8 小时、功率 4 倍；12 / 25 kW），看"12 kW 打平、25 kW 有收益"是否依赖库容与功率；
models：9 个模型各自的滚动调参管线（均值 × 裕量 {0,1,2} × 4 种选解 + 分位数 × 3 种选解），与同一批滚动规则比较——
        区分"用了逐小时计划"与"预测本身有本事"：季节朴素预测的调参管线能不能也赢规则？
a8b   ：A8 自身设备口径下电池数的敏感性（10 / 20 块，各占一个充电位，1 小时充电）。
用法：python exp_bl_models.py extra | models | a8b | ana"""
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
from analysis import MAIN_POOL
from exp_packs import TH, RAW
from exp_bl import task, analyse, load_events, QL, ETA
from exp_bl_more import save
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID_X = [((8, 2.5, 3), 12.0), ((8, 2.5, 3), 25.0), ((6, 4.0, 3), 12.0), ((6, 4.0, 3), 25.0)]
GRID_M = [((6, 2.5, 3), 12.0), ((6, 2.5, 3), 18.0), ((6, 2.5, 3), 25.0), ((4, 2.5, 3), 25.0)]


def mtask(d, ev_step, ev_e, y, price, tot, hmean, fq, ebar):
    """fq：{模型: (均值, 分位数矩阵)}。只算计划类候选（规则沿用 bl_rule.parquet）。"""
    rows = []
    for m, (f, Q) in fq.items():
        fcs = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        _, rr = task(d, ev_step, ev_e, y, price, tot, hmean, fcs, ebar, GRID_M, False, False)
        rows += [(m, *r[:7]) for r in rr]
    return rows


def ana_models():
    FC = pd.read_parquet(OUT / "bl_fc_models.parquet")
    RU = pd.read_parquet(OUT / "bl_rule.parquet")
    lines = []
    for (c, pc), g in FC.groupby(["cfg", "pc"], sort=False):
        r0 = RU[(RU.cfg == c) & (RU.pc == pc)]
        Mc = r0.pivot(index="day", columns="th", values="cost").sort_index()
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        s = f"{c} {pc:.0f} kW：各模型滚动调参管线相对滚动规则 %："
        for m in MAIN_POOL:
            Fc = g[g.model == m].pivot(index="day", columns="cand", values="cost").sort_index()
            days = [d for d in Fc.index if d >= START and d in Mc.index]
            ref = np.array([Mc.loc[d, pick(Mc, d, k0)] for d in days])
            x = np.array([Fc.loc[d, pick(Fc, d, "mean@0|mid")] for d in days])
            a, lo, hi = rel_ci(x, ref)
            b = rel_ci(Fc.loc[days, "mean@1|max"].to_numpy(), ref)
            s += f" {m} {100 * a:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]（均值+最大+1 步 {100 * b[0]:+.1f}）；"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_models.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ana":
        ana_models()
        sys.exit()
    if mode == "a8b":
        ST = "A8"
        p1 = 1.0 / ETA
        grid = [((7.2, 12.0, 3), p1), ((14.4, 16.0, 3), p1), ((7.2, 12.0, 10), p1), ((14.4, 16.0, 10), p1)]
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
        print(len(Tk), "tasks", flush=True)
        save(Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk), "_a8b")
        sys.exit()
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    pool = ["lgb_local"] if mode == "extra" else MAIN_POOL
    FQ = {}
    for m in pool:
        F = pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet")
        FQ[m] = F[F.series == S].set_index("t")
    Tk = []
    for d in pd.date_range("2024-01-15" if mode == "extra" else cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all() or d not in EV:
            continue
        price, tot = D.price(S, d.month), HK(D.price("A7", d.month, "total"))
        if d < cfg.MP_EVAL_START:
            Tk.append((d, *EV[d], y, price, tot, hmean, None, ebar, GRID_X))
            continue
        fq = {m: (FQ[m]["mean"].reindex(idx).to_numpy(), FQ[m][cfg.QC].reindex(idx).to_numpy()) for m in pool}
        if not all(np.isfinite(a).all() and np.isfinite(b).all() for a, b in fq.values()):
            continue
        if mode == "extra":
            f, Q = fq["lgb_local"]
            Tk.append((d, *EV[d], y, price, tot, hmean, {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}, ebar, GRID_X))
        else:
            Tk.append((d, *EV[d], y, price, tot, hmean, fq, ebar))
    print(mode, len(Tk), "tasks", flush=True)
    if mode == "extra":
        save(Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(task)(*t) for t in Tk), "_x")
    else:
        res = Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(mtask)(*t) for t in Tk)
        pd.DataFrame([r for rr in res for r in rr], columns=["model", "day", "cfg", "pc", "n", "cand", "cost", "short"]).to_parquet(OUT / "bl_fc_models.parquet")
        ana_models()
