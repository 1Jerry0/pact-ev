"""7 个常规站（储能辅助充电）：不用预测的分时规则控制 与 预测+计划 的同口径比较。
规则族（参数在 1–3 月开发期逐站逐组合选定，评估期不再调整）：
  充电时段 ∈ {只在谷段, 谷段+平段}；放电时段 ∈ {峰及以上, 只在最高价段, 平段及以上（且不在充电时段）}；
  套利可用电量之外保留 ρ·E（ρ ∈ {0, 0.25, 0.5}）专门用于削减超过接入上限的负荷。
执行物理与冻结流程的逐时纠偏相同（sim._st_step）。完全信息最优值直接取正式输出 mp_sched.parquet 的 opt 列。
用法：python exp_storage_rule.py"""
import itertools
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "formal" / "code"))
import numpy as np
import pandas as pd
import cfg
import data as D
import sim
from dec_mp import scales
from analysis import boot_mean

OUT = HERE.parents[1] / "results"
ETA = cfg.ETA
FAM = list(itertools.product((0, 1), (0, 1, 2), (0.0, 0.25, 0.5)))


def rule_cost(y, price, E, L, lam, par):
    cmode, dmode, rho = par
    lv = np.unique(np.round(price, 6))
    cls = np.searchsorted(lv, np.round(price, 6))
    top = cls.max()
    P = E / 2
    soc, tot = E / 2, 0.0
    for t in range(24):
        ch = cls[t] == 0 or (cmode == 1 and cls[t] == 1)
        dis = (not ch) and ((dmode == 0 and cls[t] >= 2) or (dmode == 1 and cls[t] == top and top >= 2) or (dmode == 2 and cls[t] >= 1))
        c_p = P if ch else 0.0
        d_p = min(P, max(soc - rho * E, 0.0) * ETA) if dis else 0.0
        net, soc = sim._st_step(c_p, d_p, y[t], soc, E, L, True)
        tot += price[t] * net + lam * max(net - L, 0)
    return tot + sim.terminal_value(soc, E / 2, price.min())


if __name__ == "__main__":
    H = D.mp_hourly()
    SC = scales(H)
    S = pd.read_parquet(cfg.OUT / "dec" / "mp_sched.parquet")
    S = S[S.station.isin(cfg.POWER)]
    opt = S.groupby(["station", "day", "cfg"]).opt.first()
    rows = []
    for st in cfg.POWER:
        for d in pd.date_range("2024-01-08", cfg.MP_FC_END, freq="D"):
            y = H[st].reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
            if not np.isfinite(y).all():
                continue
            p = D.price(st, d.month)
            for par in cfg.ST_GRID:
                fe, fl, lam = par
                E, L = fe * SC[st]["dmean"], SC[st]["peaks"][fl]
                rows.append((st, d, str(par), *[rule_cost(y, p, E, L, lam, f) for f in FAM]))
        print(st, "done", flush=True)
    R = pd.DataFrame(rows, columns=["station", "day", "cfg"] + list(range(len(FAM))))
    dev = R[R.day < cfg.MP_EVAL_START]
    best = dev.groupby(["station", "cfg"])[list(range(len(FAM)))].sum().idxmin(axis=1)
    ev = R[R.day >= cfg.MP_EVAL_START].copy()
    pick = pd.MultiIndex.from_frame(ev[["station", "cfg"]]).map(best).to_numpy()
    ev["rule_cost"] = ev[list(range(len(FAM)))].to_numpy()[np.arange(len(ev)), pick]
    ev = ev.merge(opt.reset_index(), on=["station", "day", "cfg"])
    ev["rule"] = ev.rule_cost - ev.opt
    P = S[(S.model == "lgb_local") & (S.strategy == "point") & (S["mode"] == "current")].pivot_table(index=["station", "day", "cfg"], columns="tie", values="regret").add_prefix("lgb_").reset_index()
    N = S[(S.model == "naive") & (S.strategy == "point") & (S["mode"] == "current") & (S.tie == "mid")][["station", "day", "cfg", "regret"]].rename(columns={"regret": "naive_mid"})
    A = ev[["station", "day", "cfg", "rule", "opt"]].merge(P, on=["station", "day", "cfg"]).merge(N, on=["station", "day", "cfg"])
    A.to_parquet(OUT / "storage_rule.parquet")
    pd.set_option("display.width", 250)
    g = A.groupby("station")[["opt", "lgb_default", "lgb_mid", "naive_mid", "rule"]].mean()
    g["rule/lgb_mid"] = g.rule / g.lgb_mid
    g["lgb_mid 占成本%"] = 100 * g.lgb_mid / (g.opt + g.lgb_mid)
    g["rule 占成本%"] = 100 * g.rule / (g.opt + g.rule)
    print("== 全部 27 组合，评估期日均（元）：完全信息成本、遗憾 ==")
    print(g.round(2).to_string())
    for st, x in A.groupby("station"):
        dd = x.groupby("day")[["rule", "lgb_mid"]].sum()
        m, ci, _ = boot_mean(1 - dd.lgb_mid / dd.rule)
        print(f"{st}: LightGBM 中点解相对规则的日度遗憾降低 {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    # 同口径（代表性组合）与 MPC、情景优化比较
    M = S[(S.model == "lgb_local") & (S["mode"] == "mpc")][["station", "day", "cfg", "regret"]].rename(columns={"regret": "mpc_mid"})
    Q = S[(S.model == "lgb_local") & (S.strategy == "saa_qt")][["station", "day", "cfg", "regret"]].rename(columns={"regret": "saa_qt"})
    B = A.merge(M, on=["station", "day", "cfg"]).merge(Q, on=["station", "day", "cfg"])
    gb = B.groupby("station")[["lgb_default", "lgb_mid", "mpc_mid", "saa_qt", "rule"]].sum()
    print("\n== 代表性 9 组合、有情景历史的样本：总遗憾 / 默认解 =="); print(gb.div(gb.lgb_default, axis=0).round(3).to_string())
    print("\n选中的规则参数（充电时段, 放电时段, 保留比例）分布："); print(best.map(lambda i: FAM[i]).value_counts().head(8).to_string())
