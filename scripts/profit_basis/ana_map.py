"""汇总 exp_map.py：资源—价值图、缺口随缓冲比例的单调性、缓冲持有成本的盈亏平衡点。用法：python ana_map.py"""
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "formal" / "code"))
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
from ana_bss import rule_series, par

OUT = HERE.parents[2] / "results" / "profit_basis"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
AL = [0.0, 0.25, 0.5, 0.75, 1.0]

if __name__ == "__main__":
    P = par(pd.read_parquet(OUT / "bss_map_plans.parquet"))
    RU = rule_series("map")
    print("执行函数与冻结流程一致性（最大成本差）：", P.chk.max())
    A = P.merge(RU, on=["day", "cfg"])
    A["best_a"] = A[[f"reg_a{a}" for a in AL]].min(axis=1)
    for m in ("lgb_local", "combo5"):
        for mult in sorted(A.mult.unique()):
            X = A[(A.model == m) & (A.mult == mult)]
            g = X.groupby(["cap", "pw"])[["reg_default", "reg_a0.5", "reg_a1.0", "rule_roll", "rule_static", "rule_full", "opt"]].sum()
            print(f"\n===== {m}，罚价 ×{mult} =====")
            print("中点解 / 默认解（总遗憾）："); print((g["reg_a0.5"] / g.reg_default).unstack("pw").round(2).to_string())
            print("中点解 / 滚动规则（<1 表示预测+计划更好）："); print((g["reg_a0.5"] / g.rule_roll).unstack("pw").round(2).to_string())
            print("最大缓冲 / 滚动规则："); print((g["reg_a1.0"] / g.rule_roll).unstack("pw").round(2).to_string())
            print("默认解遗憾占总成本 %："); print((100 * g.reg_default / (g.opt + g.reg_default)).unstack("pw").round(1).to_string())
            print("中点解遗憾占总成本 %："); print((100 * g["reg_a0.5"] / (g.opt + g["reg_a0.5"])).unstack("pw").round(1).to_string())
            print("滚动规则遗憾占总成本 %："); print((100 * g.rule_roll / (g.opt + g.rule_roll)).unstack("pw").round(1).to_string())
    # 单调性：缺口是否随 α 不增
    Sh = P[[f"short_a{a}" for a in AL]].to_numpy()
    tol = 1e-6 * np.maximum(P.dem.to_numpy(), 1)[:, None]
    mono = (np.diff(Sh, axis=1) <= tol).all(axis=1)
    print("\n缺口随 α 单调不增的比例（日×组合×模型）：", round(mono.mean(), 4), "；违反时的最大增量 / 日需求：", float((np.diff(Sh, axis=1) / P.dem.to_numpy()[:, None]).max()))
    Rg = P[[f"reg_a{a}" for a in AL]].to_numpy()
    print("遗憾最小的 α 分布：", dict(zip(AL, np.round(np.bincount(Rg.argmin(1), minlength=5) / len(Rg), 3))))
    print("按罚价：", {int(k): dict(zip(AL, np.round(np.bincount(v[[f'reg_a{a}' for a in AL]].to_numpy().argmin(1), minlength=5) / len(v), 2))) for k, v in P.groupby("mult")})
    # 持有成本盈亏平衡：h* = 遗憾节省 / 额外库存·小时
    for mult in sorted(P.mult.unique()):
        X = P[(P.model == "lgb_local") & (P.mult == mult)]
        g = X.groupby("pw").apply(lambda x: pd.Series({
            "额外平均库存占库容": ((x["inv_a0.5"] - x.inv_default) / x.cap_kwh).mean(),
            "盈亏平衡持有成本 元/(kWh·h)": (x.reg_default - x["reg_a0.5"]).sum() / (24 * (x["inv_a0.5"] - x.inv_default)).sum()}))
        print(f"\n中点解相对默认解，罚价 ×{mult}："); print(g.round(4).to_string())
