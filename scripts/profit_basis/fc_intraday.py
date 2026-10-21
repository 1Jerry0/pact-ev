"""日内滚动更新的预测（A7 逐时换电补能量）：在日前 LightGBM 预测之上，每个整点用截至上一小时的实际需求重新预测当天剩余各小时。
样本 = （目标小时 t，发布整点 o ≤ t 所在小时，同一天）；特征：日前模型的全部特征 + 日前预测（均值、q10、q90）+ 提前量 + 发布整点
+ 发布前 1/2/3 小时实际值、近 6/24 小时合计、近 1/3/6/24 小时的日前预测误差、当天已过时段的实际 / 日前预测之比、昨天同小时实际值。
模型：LightGBM（冻结的超参数），目标：均值与 0.6–0.9 分位；每 7 天扩张重训，训练样本为目标日 ≤ 重训日前一天（日内信息实时可得，不需要 D−2 的滞后）。
输出 out/fc_intraday_a7.parquet：t, lead, mean, q0.6…q0.9, y, da；并报告分提前量的精度，以及"到达噪声地板"（泊松）估计。用法：python fc_intraday.py"""
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
import lightgbm as lgb
from joblib import Parallel, delayed
import cfg
import data as D

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
OBJ = [("mean", dict(objective="l2"))] + [(f"q{q}", dict(objective="quantile", alpha=q)) for q in (0.6, 0.7, 0.8, 0.9)]
EXTRA = ["da", "da_q10", "da_q90", "lead", "origin", "lag_d1", "r1", "r2", "r3", "s6", "s24", "e1", "e3", "e6", "e24", "sofar"]


def frame(y, da, q10, q90):
    F0 = D.features(y)
    err = y - da
    day = y.index.normalize()
    cy, cd = y.groupby(day).cumsum(), da.groupby(day).cumsum()
    parts = []
    for l in range(24):
        F = F0.copy()
        F["da"], F["da_q10"], F["da_q90"] = da, q10, q90
        F["lead"], F["origin"] = l, F.hour - l
        F["lag_d1"] = y.shift(24)
        for k in (1, 2, 3):
            F[f"r{k}"] = y.shift(l + k)
        F["s6"], F["s24"] = y.rolling(6).sum().shift(l + 1), y.rolling(24).sum().shift(l + 1)
        F["e1"] = err.shift(l + 1)
        for k in (3, 6, 24):
            F[f"e{k}"] = err.rolling(k).mean().shift(l + 1)
        sofar = (cy.shift(l + 1) / cd.shift(l + 1).where(cd.shift(l + 1) > 0)).where(F.origin > 0)
        F["sofar"] = sofar
        F["y"] = y
        parts.append(F[F.origin >= 0])
    return pd.concat(parts).sort_index(kind="stable")


def fit_predict(A, r0, r1, name, obj, feats):
    tr = A[(A.index < r0) & A.y.notna() & A.da.notna()]
    te = A[(A.index >= r0) & (A.index <= r1 + pd.Timedelta(hours=23)) & A.da.notna()]
    m = lgb.LGBMRegressor(**obj, **cfg.LGB).fit(tr[feats], tr.y)
    return r0, name, pd.Series(np.maximum(m.predict(te[feats]), 0), index=pd.MultiIndex.from_arrays([te.index, te.lead]))


def run_station(st):
    """储能站：只训练均值模型，输出 out/fc_intraday_{站}.parquet，返回分提前量的精度摘要。"""
    H = D.mp_hourly()
    y = H[st]
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == st].set_index("t")
    da, q10, q90 = (FQ[c].reindex(y.index) for c in ("mean", "q10", "q90"))
    A = frame(y, da, q10, q90)
    feats = list(D.FEATS) + EXTRA
    refits = [r for r in cfg.MP_REFITS if r >= pd.Timestamp("2024-03-28")]
    W = [(r, (refits[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(refits) else cfg.MP_FC_END) for i, r in enumerate(refits)]
    res = Parallel(n_jobs=int(os.environ.get("NJ", 6)), verbose=0)(delayed(fit_predict)(A, r0, r1, "mean", OBJ[0][1], feats) for r0, r1 in W)
    P = pd.DataFrame({"mean": pd.concat([s for _, _, s in res])})
    P.index.names = ["t", "lead"]
    P = P.reset_index()
    P["y"], P["da"] = y.reindex(P.t).to_numpy(), da.reindex(P.t).to_numpy()
    P.to_parquet(OUT / f"fc_intraday_{st}.parquet")
    E = P[P.t >= cfg.MP_EVAL_START].dropna(subset=["y", "da"])
    ybar = float(E.y.mean())
    s = f"{st}：nMAE 日内 / 日前（同样本）"
    for l in (0, 1, 3, 6, 12):
        e = E[E.lead == l]
        s += f"；提前 {l} h {np.abs(e.y - e['mean']).mean() / ybar:.3f} / {np.abs(e.y - e.da).mean() / ybar:.3f}"
    return s + f"；全部 {np.abs(E.y - E['mean']).mean() / ybar:.3f} / {np.abs(E.y - E.da).mean() / ybar:.3f}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out = [run_station(st) for st in sys.argv[1:]]
        txt = "\n".join(out)
        (OUT / "fc_intraday_storage.txt").write_text(txt, encoding="utf-8")
        print(txt)
        sys.exit()
    H = D.mp_hourly()
    y = H[S]
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    da, q10, q90 = (FQ[c].reindex(y.index) for c in ("mean", "q10", "q90"))
    A = frame(y, da, q10, q90)
    feats = list(D.FEATS) + EXTRA
    refits = [r for r in cfg.MP_REFITS if r >= pd.Timestamp("2024-03-28")]
    W = [(r, (refits[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(refits) else cfg.MP_FC_END) for i, r in enumerate(refits)]
    res = Parallel(n_jobs=int(os.environ.get("NJ", 6)), verbose=1)(delayed(fit_predict)(A, r0, r1, nm, ob, feats) for r0, r1 in W for nm, ob in OBJ)
    P = {}
    for r0, nm, s in res:
        P.setdefault(nm, []).append(s)
    P = pd.DataFrame({nm: pd.concat(v) for nm, v in P.items()})
    P.index.names = ["t", "lead"]
    P = P.reset_index()
    P["y"], P["da"] = y.reindex(P.t).to_numpy(), da.reindex(P.t).to_numpy()
    P.to_parquet(OUT / "fc_intraday_a7.parquet")
    E = P[P.t >= cfg.MP_EVAL_START]
    ybar = float(y.loc[cfg.MP_EVAL_START:cfg.MP_FC_END + pd.Timedelta(hours=23)].mean())
    lines = [f"A7 日内滚动预测，评估期 {E.t.dt.normalize().nunique()} 天；平均需求 {ybar:.1f} kWh/h；nMAE = MAE / 平均需求"]
    g = E.groupby("lead").apply(lambda d: pd.Series(dict(id=np.abs(d.y - d["mean"]).mean() / ybar, da=np.abs(d.y - d.da).mean() / ybar, n=len(d))))
    for l in (0, 1, 2, 3, 6, 12, 18, 23):
        lines.append(f"提前 {l:>2} 小时：日内 {g.loc[l, 'id']:.3f}，同样本日前 {g.loc[l, 'da']:.3f}（{int(g.loc[l, 'n'])} 个样本）")
    lines.append(f"全部（发布整点，目标）对：日内 {np.abs(E.y - E['mean']).mean() / ybar:.3f}，日前 {np.abs(E.y - E.da).mean() / ybar:.3f}")
    # 到达噪声地板：若每小时换电次数是以预测为均值的泊松、每次电量取实际分布，MAE 至少是多少
    from exp_bl import load_events
    EV, ebar = load_events()
    en = np.concatenate([e for _, e in EV.values()])
    rng = np.random.default_rng(cfg.SEED)
    Ed = E[E.lead == 0]
    lam = np.maximum(Ed.y.to_numpy(), 0)      # 用实际值的平滑版本做强度会低估噪声；这里分别用两种强度：日前预测、同小时 28 天均值
    out = {}
    for nm, rate in (("日前预测作强度", np.maximum(Ed.da.to_numpy(), 0)), ("日内预测作强度", np.maximum(Ed["mean"].to_numpy(), 0))):
        mae = []
        for _ in range(20):
            nsw = rng.poisson(rate / ebar)
            tot = np.array([rng.choice(en, size=k).sum() if k else 0.0 for k in nsw])
            mae.append(np.abs(tot - rate).mean())
        out[nm] = float(np.mean(mae)) / ybar
    cnt = pd.Series({d: len(e) for d, (s_, e) in EV.items()})
    hourly_cnt = pd.Series(np.concatenate([np.bincount(s_ // 4, minlength=24) for d, (s_, e) in EV.items() if d >= cfg.MP_EVAL_START]))
    lines.append(f"到达噪声地板（复合泊松：每小时次数 ~ Poisson(强度 / {ebar:.1f} kWh)，每次电量取实际分布；nMAE）：" + "；".join(f"{k} {v:.3f}" for k, v in out.items())
                 + f"；评估期每小时平均换电 {hourly_cnt.mean():.1f} 次，单次电量变异系数 {en.std() / en.mean():.2f}")
    txt = "\n".join(lines)
    (OUT / "fc_intraday.txt").write_text(txt, encoding="utf-8")
    print(txt)
