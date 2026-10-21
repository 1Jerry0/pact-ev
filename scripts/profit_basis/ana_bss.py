"""汇总 exp_bss.py 的输出：规则基线（开发期定阈值 / 滚动 28 天定阈值）与计划类方法的同口径比较、误差缩放的等价精度。
用法：python ana_bss.py paper|anchor"""
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
import cfg
from analysis import boot_mean

OUT = HERE.parents[2] / "results" / "profit_basis"
EX = (pathlib.Path(__file__).resolve().parents[2] / "results" / "profit_basis" / "tables_extra")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
EVAL0 = cfg.MP_EVAL_START


def rule_series(tag):
    """返回每个 (cfg, day) 的规则遗憾与缺口：static = 开发期（1–3 月）选定阈值；roll = 用 D−29…D−2 的成本滚动选阈值；full = 换下即充（阈值全为 1）。"""
    R = pd.read_parquet(OUT / f"bss_{tag}_rule.parquet")
    nth = R.th.max() + 1
    out = []
    for c, g in R.groupby("cfg"):
        M = g.pivot(index="day", columns="th", values="regret").sort_index()
        Sh = g.pivot(index="day", columns="th", values="short").sort_index()
        dev = M.loc[:EVAL0 - pd.Timedelta(days=1)]
        best = int(dev.sum().idxmin())
        ev = M.loc[EVAL0:]
        roll_pick = []
        for d in ev.index:
            past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
            roll_pick.append(int(past.sum().idxmin()))
        o = pd.DataFrame({"cfg": c, "day": ev.index, "rule_static": ev[best].to_numpy(), "short_static": Sh.loc[ev.index, best].to_numpy(),
                          "rule_roll": ev.to_numpy()[np.arange(len(ev)), roll_pick], "short_roll": Sh.loc[ev.index].to_numpy()[np.arange(len(ev)), roll_pick],
                          "rule_full": ev[nth - 1].to_numpy(), "short_full": Sh.loc[ev.index, nth - 1].to_numpy(), "th_static": best})
        out.append(o)
    return pd.concat(out, ignore_index=True)


def ci(a, b):
    """日度 1 − Σa/Σb 的均值与平稳块自助法区间（与 P4 同一统计量）。"""
    m, c, _ = boot_mean(1 - a / b)
    return f"{m:.3f} [{c[0]:.3f}, {c[1]:.3f}]"


def par(R):
    P = R.cfg.map(eval)
    return R.assign(cap=P.map(lambda t: t[0]), pw=P.map(lambda t: t[1]), mult=P.map(lambda t: t[2]))


if __name__ == "__main__":
    tag = sys.argv[1]
    RU = par(rule_series(tag))
    if tag == "paper":
        PR = pd.read_parquet(EX / "plan_robust_A7.parquet")
        L = PR[PR.model == "lgb_local"][["day", "cfg", "reg_default", "reg_mid", "reg_max", "reg_ipm", "reg_noise"]]
        A = par(RU.merge(L, on=["day", "cfg"]))
        cols = ["reg_default", "reg_ipm", "reg_noise", "reg_mid", "reg_max", "rule_full", "rule_static", "rule_roll"]
        print("== 论文原 27 组合，LightGBM，日均遗憾（元）==")
        print(A[cols].mean().round(0).to_string())
        print("\n按罚价倍数："); print(A.groupby("mult")[cols].mean().round(0).to_string())
        print("\n按功率倍数："); print(A.groupby("pw")[cols].mean().round(0).to_string())
        print("\n按库容："); print(A.groupby("cap")[cols].mean().round(0).to_string())
        g = A.groupby("cfg")[cols].mean()
        print("\n规则（滚动）优于中点解的组合数：", int((g.rule_roll < g.reg_mid).sum()), "/ 27；优于最大缓冲：", int((g.rule_roll < g.reg_max).sum()))
        D = A.groupby("day")[cols].sum()
        print("滚动规则相对中点解的日度遗憾降低：", ci(D.rule_roll, D.reg_mid), "；相对默认解：", ci(D.rule_roll, D.reg_default))
        # 同口径杠杆比较：代表性 9 组合（罚价 ×3），隔日样本（mpc_ties 的日期）
        MT = pd.read_parquet(EX / "mpc_ties.parquet")
        MT = MT[(MT.station == cfg.BSS) & (MT.model == "lgb_local")]
        mp = MT.pivot_table(index=["day", "cfg"], columns="tie", values="mpc").add_prefix("mpc_").reset_index()
        SC = pd.read_parquet(cfg.OUT / "dec" / "mp_sched.parquet")
        SC = SC[(SC.station == cfg.BSS) & (SC.model == "lgb_local") & (SC.strategy == "saa_qt")][["day", "cfg", "regret"]].rename(columns={"regret": "saa_qt"})
        B = A.merge(mp, on=["day", "cfg"]).merge(SC, on=["day", "cfg"])
        cols2 = ["reg_default", "reg_mid", "reg_max", "mpc_default", "mpc_max", "saa_qt", "rule_full", "rule_static", "rule_roll"]
        print(f"\n== 同口径：代表性 9 组合（罚价×3）、隔日、有情景历史的样本 n={len(B)}；总遗憾 / 默认解 ==")
        print((B[cols2].sum() / B.reg_default.sum()).round(3).to_string())
        print("\n按功率倍数："); print(B.groupby("pw")[cols2].sum().div(B.groupby("pw").reg_default.sum(), axis=0).round(3).to_string())
        D2 = B.groupby("day")[cols2].sum()
        print("MPC+最大缓冲 相对 滚动规则：", ci(D2.mpc_max, D2.rule_roll), "；情景优化 相对 滚动规则：", ci(D2.saa_qt, D2.rule_roll))
        # 误差缩放
        LV = par(pd.read_parquet(OUT / "bss_paper_plans.parquet"))
        sc = {1.0: (LV.reg_default.sum(), LV.reg_mid.sum())}
        for k in (0.9, 0.8, 0.7, 0.6, 0.5):
            sc[k] = (LV[f"sc{k}_default"].sum(), LV[f"sc{k}_mid"].sum())
        T = pd.DataFrame(sc, index=["default", "mid"]).T / LV.reg_default.sum()
        print("\n== 误差缩放（代表性 9 组合，LightGBM；总遗憾 / 原误差默认解）=="); print(T.round(3).to_string())
    else:
        P = par(pd.read_parquet(OUT / "bss_anchor_plans.parquet"))
        M = P[~P.model.str.endswith("_lever")]
        LV = P[P.model.str.endswith("_lever")]
        L = M[M.model == "lgb_local"]
        A = L.merge(RU.drop(columns=["cap", "pw", "mult"]), on=["day", "cfg"])
        cols = ["reg_default", "reg_ipm", "reg_noise", "reg_mid", "reg_max", "rule_full", "rule_static", "rule_roll"]
        print("== 锚定网格（库容 4/6/8 h × 功率 2.5/4 × 罚价 1/3/10），LightGBM，日均遗憾（元）==")
        print(A[cols].mean().round(0).to_string())
        print("\n按功率 × 罚价："); print(A.groupby(["pw", "mult"])[cols].mean().round(0).to_string())
        sh = ["short_opt", "short_default", "short_mid", "short_max", "short_static", "short_roll", "short_full"]
        print("\n未满足需求占比："); print((A.groupby("pw")[sh].sum().div(A.groupby("pw").dem.sum(), axis=0)).round(4).to_string())
        print("\n完全信息日均成本：", round(A.opt.mean(), 0))
        # P4 同一统计量：9 个模型 × 18 组合
        D = M.groupby("day")[["reg_default", "reg_min", "reg_mid", "reg_max", "reg_ipm", "reg_noise"]].sum()
        print("\n== P4 统计量（9 模型 × 18 组合，日度降幅，95% CI）==")
        for base in ("reg_default", "reg_min", "reg_ipm", "reg_noise"):
            print(f"中点 vs {base[4:]}: {ci(D.reg_mid, D[base])}；最大缓冲 vs {base[4:]}: {ci(D.reg_max, D[base])}")
        print("默认解在最优面中的位置（均值/中位数）：", round(M.pos_default.mean(), 3), round(M.pos_default.median(), 3), "；最优面宽度中位数：", round(M.width.median(), 3))
        g = M.groupby(["pw", "mult"])[["reg_default", "reg_ipm", "reg_noise", "reg_mid", "reg_max"]].sum()
        print("\n按功率 × 罚价（总遗憾 / 默认解）："); print(g.div(g.reg_default, axis=0).round(3).to_string())
        # 模型排序：精度 vs 价值
        mm = M.groupby(["cfg", "model"])[["mae", "reg_default", "reg_mid"]].mean().reset_index()
        from scipy.stats import kendalltau
        kt = mm.groupby("cfg").apply(lambda x: pd.Series({"tau_default": kendalltau(x.mae, x.reg_default)[0], "tau_mid": kendalltau(x.mae, x.reg_mid)[0],
                                                         "mismatch_mid": float(x.loc[x.mae.idxmin(), "model"] != x.loc[x.reg_mid.idxmin(), "model"])}))
        print("\n精度—价值：Kendall τ 中位数（默认 / 中点）", kt.tau_default.median().round(3), kt.tau_mid.median().round(3), "；MAE 最优≠遗憾最优的组合比例（中点）", kt.mismatch_mid.mean().round(3))
        print("各模型日均遗憾："); print(M.groupby("model")[["mae", "reg_default", "reg_mid", "reg_max"]].mean().round(1).sort_values("mae").to_string())
        # 杠杆（LightGBM）
        B = LV.merge(RU.drop(columns=["cap", "pw", "mult"]), on=["day", "cfg"]).dropna(subset=["reg_saa"])
        c1 = ["reg_default", "reg_mid", "reg_max", "reg_alpha", "reg_saa", "rule_full", "rule_static", "rule_roll"]
        print(f"\n== 杠杆（LightGBM，有情景历史的全部评估日 n={len(B)}；总遗憾 / 默认解）==")
        print(B.groupby(["pw", "mult"])[c1].sum().div(B.groupby(["pw", "mult"]).reg_default.sum(), axis=0).round(3).to_string())
        Bm = B.dropna(subset=["mpc_max"])
        c2 = c1 + ["mpc_default", "mpc_max"]
        print(f"\n含 MPC（隔日 n={len(Bm)}）：")
        print(Bm.groupby(["pw", "mult"])[c2].sum().div(Bm.groupby(["pw", "mult"]).reg_default.sum(), axis=0).round(3).to_string())
        D2 = Bm.groupby("day")[c2].sum()
        print("中点解 相对 滚动规则：", ci(D2.reg_mid, D2.rule_roll), "；情景优化 相对 滚动规则：", ci(D2.reg_saa, D2.rule_roll), "；MPC+最大缓冲 相对 滚动规则：", ci(D2.mpc_max, D2.rule_roll))
        sc = {1.0: (LV.reg_default.sum(), LV.reg_mid.sum())}
        for k in (0.9, 0.8, 0.7, 0.6, 0.5):
            sc[k] = (LV[f"sc{k}_default"].sum(), LV[f"sc{k}_mid"].sum())
        T = pd.DataFrame(sc, index=["default", "mid"]).T / LV.reg_default.sum()
        print("\n== 误差缩放（锚定网格，LightGBM；总遗憾 / 原误差默认解）=="); print(T.round(3).to_string())
