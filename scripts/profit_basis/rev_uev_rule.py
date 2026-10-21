"""UrbanEV 套用换电模型时的无预测规则：把"9–10 月一次选定阈值"补成"按 D−29…D−2 滚动选阈值"，与正文其余部分的规则口径一致。
只重算规则，计划类结果沿用 out\\rev_uev.parquet。用法：python rev_uev_rule.py"""
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
from exp_bss import rule_costs, TH
from rev_uev import GRID
from exp_packs_value import rel_ci

OUT = HERE.parents[2] / "results" / "profit_basis"


def task(z, hmean, days):
    """days: 按时间排序的 [(day, y, price, tot, is_eval)]。返回评估日的 (zone, day, cap, 滚动规则成本, 静态规则成本)。"""
    out = []
    for par in GRID:
        C, idx = [], []
        for d, y, price, tot, _ in days:
            pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
            C.append(rule_costs(y, price, pen, par[0] * hmean, par[1] * hmean)[0])
            idx.append(d)
        M = pd.DataFrame(np.array(C), index=pd.DatetimeIndex(idx))
        k0 = int(M[M.index <= cfg.UEV_DEV_END].sum().idxmin())
        for d, _, _, _, ev in days:
            if ev:
                past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
                k = int(past.sum().idxmin()) if len(past) >= 7 else k0
                out.append((z, d, par[0], float(M.loc[d, k]), float(M.loc[d, k0])))
    return out


if __name__ == "__main__":
    V = D.uev_volume()
    dev_days = pd.date_range("2022-09-01", cfg.UEV_DEV_END.normalize(), freq="D")
    ev_days = D.days(cfg.UEV_EVAL_START, cfg.UEV_FC_END)
    PR = {mo: (D.price("A1", mo), HK(D.price("A1", mo, "total"))) for mo in (9, 10, 11, 12, 1, 2)}
    T = []
    for z in V.columns:
        hmean = float(V[z].loc[:cfg.UEV_DEV_END].mean())
        if hmean <= 0:
            continue
        days = []
        for d in list(dev_days) + list(ev_days):
            y = V[z].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
            if np.isfinite(y).all():
                days.append((d, y, *PR[d.month], d >= cfg.UEV_EVAL_START))
        T.append((z, hmean, days))
    print(len(T), "zones", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("NJ", 7)), batch_size=1, verbose=1)(delayed(task)(*t) for t in T)
    RU = pd.DataFrame([r for rr in res for r in rr], columns=["zone", "day", "cap", "c_roll", "c_static"])
    RU.to_parquet(OUT / "rev_uev_rule.parquet")
    R = pd.read_parquet(OUT / "rev_uev.parquet")
    L = R[R.model == "lgb_global"].merge(RU, on=["zone", "day", "cap"])
    L["rule_roll"] = L.c_roll - L.opt
    L["chk"] = (L.c_static - L.opt - L.rule_static).abs()
    tot = L[["reg_default", "reg_mid", "reg_max", "rule_static", "rule_roll", "rule_full"]].sum()
    lines = [f"UrbanEV 换电，LightGBM（全局），{L.zone.nunique()} 区 × {L.day.nunique()} 天；静态规则复算最大差 {L.chk.max():.2e}",
             "总遗憾 / 默认解：" + (tot / tot.reg_default).round(3).to_dict().__repr__()]
    dd = L.groupby("day")[["reg_mid", "reg_max", "rule_roll", "rule_static"]].sum()
    for a in ("reg_mid", "reg_max"):
        m, lo, hi = rel_ci(dd[a], dd.rule_roll)
        lines.append(f"{a} 相对滚动规则的总遗憾差 {100 * m:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    zz = L.groupby("zone")[["reg_mid", "rule_roll", "rule_static", "reg_default"]].sum()
    lines.append(f"中点解优于滚动规则的区占比 {(zz.reg_mid < zz.rule_roll).mean():.3f}（静态规则 {(zz.reg_mid < zz.rule_static).mean():.3f}）")
    bycap = L.groupby("cap")[["reg_default", "reg_mid", "rule_roll"]].sum()
    lines.append("按库容：" + (bycap.div(bycap.reg_default, axis=0)).round(3).to_dict().__repr__())
    txt = "\n".join(lines)
    (OUT / "rev_uev_rule.txt").write_text(txt, encoding="utf-8")
    print(txt)
