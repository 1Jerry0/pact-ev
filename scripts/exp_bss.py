"""A7 换电站的验证实验（只读冻结流程，不改动 正式实验/ 下任何文件）。
回答三个问题：
  1. 不用预测的库存规则（换下即充 / 分时阈值规则）与"预测 + 计划"相比如何；
  2. 把充电功率锚定到原始会话数据支持的范围（2.5–4 倍均值需求）后，计划选择（P4）和各对照是否仍成立；
  3. 计划选择相当于把预测误差缩小多少（误差缩放扫描）。
配置集：
  paper  —— 论文原 27 组合：规则 + 代表性 9 组合上的误差缩放（计划类结果直接读正式输出）；
  anchor —— 库容 4/6/8 小时 × 功率 2.5/4.0 × 罚价 1/3/10：9 个模型的默认/最小/中点/最大/内点/电价扰动，
            LightGBM 的 MPC、情景优化、情景选 α、误差缩放。
用法：python exp_bss.py [paper] [anchor]"""
import os
import pathlib
import sys
import zlib
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
os.environ["PYTHONPATH"] = str(ROOT / "formal" / "code") + os.pathsep + os.environ.get("PYTHONPATH", "")   # 并行子进程也能导入冻结模块
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import cfg
import data as D
import sim

OUT = HERE.parents[1] / "results"
ETA = cfg.ETA
S = cfg.BSS
MAIN_POOL = ["naive", "naive7", "seasonal_q", "lqr", "lgb_local", "lgb_global", "chronos2_cov", "timesfm3_cov", "combo5"]
LEVER = "lgb_local"
LV = np.round(np.arange(0, 1.01, 0.1), 2)
TH = np.array([(a, b) for a in LV for b in LV if b <= a])          # (平段阈值, 峰/尖段阈值)，66 组
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCALES = [0.5, 0.6, 0.7, 0.8, 0.9]
GRIDS = {"paper": list(cfg.BSS_GRID),
         "anchor": [(c, p, m) for c in (4, 6, 8) for p in (2.5, 4.0) for m in (1, 3, 10)]}
_LP = sim.linprog


def rule_costs(y, price, pen, cap, pmax):
    """不用预测的目标库存规则（对 66 组阈值向量化）：谷段充满，平段保持 th_f·cap，峰/尖段保持 th_p·cap。"""
    lv = np.unique(np.round(price, 6))
    cls = np.searchsorted(lv, np.round(price, 6))                   # 0 谷 1 平 2+ 峰/尖
    s0 = cap / 2
    s = np.full(len(TH), s0)
    tot, short = np.zeros(len(TH)), np.zeros(len(TH))
    for t in range(24):
        tgt = np.full(len(TH), cap) if cls[t] == 0 else (TH[:, 0] * cap if cls[t] == 1 else TH[:, 1] * cap)
        g = np.clip((tgt - s) / ETA, 0, pmax)
        avail = s + ETA * g
        served = np.minimum(y[t], avail)
        u = y[t] - served
        s = np.minimum(avail - served, cap)
        tot += price[t] * g + pen[t] * u
        short += u
    lo = price.min()
    term = np.where(s < s0, (s0 - s) * lo / ETA, (s0 - s) * lo * ETA)
    return tot + term, short


def run_plan(P, pl, y=None):
    y = P.y if y is None else y
    return sim.bss_exec(pl[1], pl[2], pl[3], y, P.price, P.pen, P.cap, P.pmax, "current", pl[4])


def mix(lo, hi, a):
    return ("bss", (1 - a) * lo[1] + a * hi[1], (1 - a) * lo[2] + a * hi[2], lo[3], (1 - a) * lo[4] + a * hi[4])


def plan_ipm(P, f):
    sim.linprog = lambda *a, **k: _LP(*a, **{**k, "method": "highs-ipm"})
    try:
        return P.plan(f, "default")
    finally:
        sim.linprog = _LP


def task(tag, d, y, price, tot_max, hmean, fcs, lever, do_lever_mpc):
    """fcs：{模型: 均值预测}（开发期为空，只算规则）；lever：(均值, 原始分位数矩阵, 残差历史) 或 None。"""
    sc = dict(hmean=hmean, tot_max=tot_max)
    rows, rule = [], []
    z = np.random.default_rng(zlib.crc32(f"{S}{d.date()}".encode())).uniform(-1, 1, 24)
    for par in GRIDS[tag]:
        P = sim.Problem("bss", y, price, par, sc)
        rc, rs = rule_costs(y, price, P.pen, P.cap, P.pmax)
        rule.append((d, str(par), rc - P.opt, rs))
        if not fcs and lever is None:
            continue
        u_opt = sim.bss_plan(y, price, P.pen, P.cap, P.pmax, None)[3].sum()
        base = dict(day=d, cfg=str(par), opt=P.opt, dem=y.sum(), short_opt=u_opt)
        if tag == "anchor":
            Pn = sim.Problem("bss", y, price * (1 + 0.01 * z), par, sc)
            for m, f in fcs.items():
                pd_, pmin, pmax_ = P.plan(f, "default"), P.plan(f, "min"), P.plan(f, "max")
                pmid = mix(pmin, pmax_, 0.5)
                r = dict(base, model=m, mae=float(np.abs(y - f).mean()))
                for k, pl in (("default", pd_), ("min", pmin), ("mid", pmid), ("max", pmax_)):
                    c, sh = run_plan(P, pl)
                    r[f"reg_{k}"], r[f"short_{k}"] = c - P.opt, sh
                r["reg_ipm"] = P.cost(plan_ipm(P, f)) - P.opt
                r["reg_noise"] = Pn.cost(Pn.plan(f, "default")) - Pn.opt
                smin, smax = pmin[2].sum(), pmax_[2].sum()
                r["width"] = (smax - smin) / (25 * P.cap)
                r["pos_default"] = (pd_[2].sum() - smin) / (smax - smin) if smax - smin > 1e-6 else np.nan
                rows.append(r)
        if lever is not None and (tag == "anchor" or par in cfg.BSS_REP):
            mean, Q, Hres = lever
            r = dict(base, model=LEVER + "_lever")
            pmin, pmax_ = P.plan(mean, "min"), P.plan(mean, "max")
            pdef = P.plan(mean, "default")
            r["reg_default"], r["reg_mid"] = P.cost(pdef) - P.opt, P.cost(mix(pmin, pmax_, 0.5)) - P.opt
            r["reg_max"] = P.cost(pmax_) - P.opt
            for k in SCALES:                                           # 误差缩放：f_k = y − k (y − f)
                fk = np.maximum(y - k * (y - mean), 0)
                r[f"sc{k}_default"] = P.cost(P.plan(fk, "default")) - P.opt
                r[f"sc{k}_mid"] = P.cost(P.plan(fk, "mid")) - P.opt
            if tag == "anchor":
                if Hres is not None and len(Hres) >= 7:
                    Ys = np.maximum(mean[None, :] + Hres, 0)
                    # 情景选 α（只在等成本计划中选）
                    al = {a: mix(pmin, pmax_, a) for a in ALPHAS}
                    sa = {a: np.mean([run_plan(P, pl, ys)[0] for ys in Ys]) for a, pl in al.items()}
                    r["reg_alpha"] = P.cost(al[min(sa, key=sa.get)]) - P.opt
                    # 情景优化（分位数 × 选解），与 dec_mp 的 saa_qt 相同
                    cands = {("mean", "default"): pdef, ("mean", "mid"): al[0.5], ("mean", "max"): pmax_}
                    for q in cfg.SAA_QUANTS[1:]:
                        f = sim.qinterp(Q, q)
                        lo, hi = P.plan(f, "min"), P.plan(f, "max")
                        cands[(q, "default")], cands[(q, "mid")], cands[(q, "max")] = P.plan(f, "default"), mix(lo, hi, 0.5), hi
                    sq = {k: np.mean([run_plan(P, pl, ys)[0] for ys in Ys]) for k, pl in cands.items()}
                    r["reg_saa"] = P.cost(cands[min(sq, key=sq.get)]) - P.opt
                if do_lever_mpc:
                    r["mpc_default"] = P.mpc(mean, "default") - P.opt
                    r["mpc_max"] = P.mpc(mean, "max") - P.opt
            rows.append(r)
    return rows, rule


def build_tasks(tag):
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    F = {m: pd.read_parquet(cfg.OUT / "fc" / f"mp_{m}.parquet") for m in (MAIN_POOL if tag == "anchor" else [LEVER])}
    G = {m: F[m][F[m].series == S].set_index("t") for m in F}
    L = G[LEVER]
    R = (L.y - L["mean"])
    t0 = R.index.min().normalize()
    A = R.reindex(pd.date_range(t0, R.index.max().normalize() + pd.Timedelta(hours=23), freq="h")).to_numpy().reshape(-1, 24)
    ok = np.isfinite(A).all(1)
    tasks = []
    for d in pd.date_range("2024-01-08", cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        y = H[S].reindex(idx).to_numpy()
        if not np.isfinite(y).all():
            continue
        price, tot = D.price(S, d.month), float(D.price("A7", d.month, "total").max())
        fcs, lever, mpc = {}, None, False
        if d >= cfg.MP_EVAL_START:
            fcs = {m: np.maximum(G[m]["mean"].reindex(idx).to_numpy(), 0) for m in G} if tag == "anchor" else {}
            if all(np.isfinite(v).all() for v in fcs.values()) and np.isfinite(L["mean"].reindex(idx).to_numpy()).all():
                i = (d - t0).days
                hid = [j for j in range(max(0, i - 2 - cfg.SCEN_WINDOW + 1), i - 1) if ok[j]]
                lever = (np.maximum(L["mean"].reindex(idx).to_numpy(), 0), L[cfg.QC].reindex(idx).to_numpy(), A[hid] if hid else None)
                mpc = (d - cfg.MP_EVAL_START).days % 2 == 0
            else:
                fcs = {}
        tasks.append((tag, d, y, price, tot, hmean, fcs, lever, mpc))
    return tasks


if __name__ == "__main__":
    for tag in (sys.argv[1:] or ["paper", "anchor"]):
        T = build_tasks(tag)
        print(tag, len(T), "day tasks", flush=True)
        res = Parallel(n_jobs=6, batch_size=2, verbose=3)(delayed(task)(*t) for t in T)
        rows = [r for a, _ in res for r in a]
        pd.DataFrame(rows).to_parquet(OUT / f"bss_{tag}_plans.parquet")
        rule = [(d, c, i, float(reg[i]), float(sh[i])) for _, b in res for (d, c, reg, sh) in b for i in range(len(TH))]
        pd.DataFrame(rule, columns=["day", "cfg", "th", "regret", "short"]).to_parquet(OUT / f"bss_{tag}_rule.parquet")
        print(tag, "saved", len(rows), "plan rows", flush=True)
