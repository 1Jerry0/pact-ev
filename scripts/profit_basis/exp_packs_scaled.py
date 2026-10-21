"""电池级：换一种用预测的方式再问一次"预测值不值"。
exp_packs_value.py 里预测类是"LP 计划 + 跟踪"，它输给/打平滚动阈值规则，可能只是因为聚合 LP 与电池级执行不匹配。
这里把预测直接塞进规则类：平/峰段的库存位置阈值乘以"预测需求 / 过去 28 天同口径平均需求"，
  A 日比例：全天预测总量 / 过去 D−29…D−2 的日均实际总量；
  B 时比例：未来 4 小时预测量 / 过去同小时段 4 小时实际量的均值（逐小时变化）；
各自再用真实需求代替预测得到上限（完全预测）。阈值同样按 D−29…D−2 的电池级成本滚动选。
对照：exp_packs_value.py 存下的普通滚动规则（同一仿真、同一批天）。用法：python exp_packs_scaled.py"""
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
from exp_packs import run_day, RAW, T, TH
from exp_packs_value import pick, rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((4, 2.5, 3), 12.0, 2), ((6, 2.5, 3), 12.0, 2), ((8, 2.5, 3), 12.0, 2), ((6, 4.0, 3), 12.0, 2), ((4, 2.5, 10), 12.0, 2), ((6, 2.5, 10), 12.0, 2),
        ((4, 2.5, 3), 25.0, 1), ((6, 2.5, 3), 25.0, 1)]
W = 4


def win(v):
    """未来 W 小时（含本小时，日末截断）的需求量。"""
    return np.array([v[t:t + W].sum() for t in range(T)])


def task(d, ev_step, ev_e, price, tot, hmean, ratios, ebar):
    rows = []
    cls = np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6))
    ee = np.minimum(ev_e, 1.6 * ebar)
    for par, pc, L in GRID:
        cap, pmax = par[0] * hmean, par[1] * hmean
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(T)))
        K = max(int(round(cap / ebar)), 2)
        for nm, r in ratios.items():
            for i, (a, b) in enumerate(TH):
                tgt = np.where(cls == 0, K, np.minimum(np.where(cls == 1, a, b) * K * r, K))
                c, u, _ = run_day(ev_step, ee, price, pen, cap, pmax, pc, ebar, tgt)
                rows.append((d, str(par), pc, L, nm, i, c, u))
    return rows


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "ana"):
        d0 = pd.read_excel(RAW / "charging_session.xlsx", sheet_name="A7")
        d0.columns = ["conn", "disc", "energy"]
        d0["energy"] = pd.to_numeric(d0.energy, errors="coerce")
        d0["disc"] = pd.to_datetime(d0.disc, errors="coerce")
        a = d0[d0.energy.gt(0) & d0.disc.notna()].copy()
        onhour = (a.disc.dt.minute == 0) & (a.disc.dt.second == 0)
        mass = (a.disc >= "2024-07-26 10:00") & (a.disc < "2024-07-26 11:00")
        a = a[~onhour & ~mass].sort_values("disc")
        ebar = float(a.energy.mean())
        a["day"], a["step"] = a.disc.dt.normalize(), a.disc.dt.hour * 4 + a.disc.dt.minute // 15
        EV = {d: (g.step.to_numpy(), g.energy.to_numpy()) for d, g in a.groupby("day")}
        H = D.mp_hourly()
        hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
        FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
        G = FQ[FQ.series == S].set_index("t")["mean"]
        Y = {}
        for d in pd.date_range("2024-01-01", cfg.MP_FC_END, freq="D"):
            y = H[S].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
            if np.isfinite(y).all():
                Y[d] = y
        Tk = []
        for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
            f = np.maximum(G.reindex(pd.date_range(d, periods=24, freq="h")).to_numpy(), 0)
            past = [Y[p] for p in pd.date_range(d - pd.Timedelta(days=29), d - pd.Timedelta(days=2), freq="D") if p in Y]
            if d not in Y or d not in EV or not np.isfinite(f).all() or len(past) < 14:
                continue
            P = np.array(past)
            day_avg, win_avg = P.sum(1).mean(), np.array([win(p) for p in P]).mean(0)
            ratios = {"fc_day": np.full(T, f.sum() / day_avg), "pi_day": np.full(T, Y[d].sum() / day_avg),
                      "fc_win": win(f) / np.maximum(win_avg, 1e-9), "pi_win": win(Y[d]) / np.maximum(win_avg, 1e-9)}
            Tk.append((d, *EV[d], D.price(S, d.month), HK(D.price("A7", d.month, "total")), hmean, ratios, ebar))
        if len(sys.argv) > 1:
            Tk = Tk[:int(sys.argv[1])]
        print(len(Tk), "tasks", flush=True)
        res = Parallel(n_jobs=7, batch_size=1, verbose=2)(delayed(task)(*t) for t in Tk)
        pd.DataFrame([r for rr in res for r in rr], columns=["day", "cfg", "pc", "L", "kind", "th", "cost", "short"]).to_parquet(OUT / "bss_packs_scaled.parquet")
    SC = pd.read_parquet(OUT / "bss_packs_scaled.parquet")
    RU = pd.read_parquet(OUT / "bss_packs_value_rule.parquet")
    lines = []
    for (c, pc, L), g in SC.groupby(["cfg", "pc", "L"], sort=False):
        r0 = RU[(RU.cfg == c) & (RU.pc == pc) & (RU.L == L)]
        Mc = r0.pivot(index="day", columns="th", values="cost").sort_index()
        k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
        days = sorted(set(g.day[g.day >= START]) & set(Mc.index))
        ref = np.array([Mc.loc[d, pick(Mc, d, k0)] for d in days])
        s = f"{c} p_c={pc:.0f}kW L={L}（{len(days)} 天）：滚动规则日均成本 {ref.mean():.0f}；相对滚动规则的成本差（%）"
        for nm, h in g.groupby("kind", sort=False):
            M = h.pivot(index="day", columns="th", values="cost").sort_index()
            x = np.array([M.loc[d, pick(M, d, k0)] for d in days])
            m, lo, hi = rel_ci(x, ref)
            s += f" {nm} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_packs_scaled.txt").write_text(txt, encoding="utf-8")
    print(txt)
