"""组合预测与在线校准。用法：python fc_post.py mp | uev
- combo5：5 个成员的均值与分位数等权平均；
- 校准：每个分位数用目标日 ≤ D−2、最近 28 天的样本外残差 y − q_τ 的经验 τ 分位数做加性修正（至少 7 天历史），
  输出 {ds}_{model}_cal.parquet（均值不变）。"""
import sys
import numpy as np
import pandas as pd
import cfg
from util import log, pmap, save, done


def combo(ds):
    fp = cfg.OUT / "fc" / f"{ds}_combo5.parquet"
    if done(fp):
        return
    members = cfg.COMBO5 if ds == "mp" else cfg.UEV_COMBO5
    F = [pd.read_parquet(cfg.OUT / "fc" / f"{ds}_{m}.parquet").set_index(["series", "t"]) for m in members]
    idx = F[0].index
    for f in F[1:]:
        idx = idx.intersection(f.index)
    vals = np.mean([f.loc[idx, ["mean"] + cfg.QC].to_numpy() for f in F], axis=0)
    out = pd.DataFrame(vals, columns=["mean"] + cfg.QC, index=idx).reset_index()
    out["y"] = F[0].loc[idx, "y"].to_numpy()
    out["model"] = "combo5"
    out[cfg.QC] = np.sort(out[cfg.QC].to_numpy(), axis=1)
    save(out[["series", "model", "t", "y", "mean"] + cfg.QC], fp)


def calibrate_series(g):
    g = g.sort_values("t").reset_index(drop=True)
    day = g.t.dt.normalize()
    Q = g[cfg.QC].to_numpy()
    R = g.y.to_numpy()[:, None] - Q
    Qc = Q.copy()
    for d in day.unique():
        past = ((day <= d - pd.Timedelta(days=2)) & (day > d - pd.Timedelta(days=2 + cfg.CAL_WINDOW))).to_numpy()
        past &= np.isfinite(R[:, 0])
        if past.sum() < 24 * 7:
            continue
        cur = (day == d).to_numpy()
        corr = np.array([np.quantile(R[past, i], t) for i, t in enumerate(cfg.QS)])
        Qc[cur] = np.sort(np.maximum(Q[cur] + corr, 0), axis=1)
    g[cfg.QC] = Qc
    return g


def calibrate(ds, model):
    fp = cfg.OUT / "fc" / f"{ds}_{model}_cal.parquet"
    if done(fp):
        return
    F = pd.read_parquet(cfg.OUT / "fc" / f"{ds}_{model}.parquet")
    res = pmap(calibrate_series, [(g,) for _, g in F.groupby("series")], desc=f"{ds} {model} 校准", name=f"post_{ds}")
    out = pd.concat(res, ignore_index=True)
    out["model"] = model
    save(out, fp)


def run(ds):
    combo(ds)
    models = (cfg.MP_MODELS_STAT + cfg.MP_MODELS_TSFM if ds == "mp" else cfg.UEV_MODELS_STAT + cfg.UEV_MODELS_TSFM) + ["combo5"]
    for m in models:
        calibrate(ds, m)
    log(f"{ds} 组合与校准完成", f"post_{ds}")


if __name__ == "__main__":
    run(sys.argv[1])
