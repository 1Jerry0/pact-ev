"""缓冲的持有成本盈亏平衡（利润口径）：中点计划相对求解器默认计划节省的遗憾 ÷ 多持有的库存·小时。
读 结果\利润口径\bss_map_plans.parquet（exp_map.py 利润口径副本的输出，LightGBM）。输出：结果\breakeven_profit.txt"""
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = pd.read_parquet(ROOT / "results" / "profit_basis" / "bss_map_plans.parquet")
P = P[P.model == "lgb_local"].copy()
P[["cap", "pw"]] = P.cfg.str.strip("()").str.split(",", expand=True).iloc[:, :2].astype(float)
CNY_PER_USD = 7.1217        # 国家统计局《2024 年统计公报》：全年人民币平均汇率 1 美元兑 7.1217 元
PACK_USD = 94.0             # BNEF (2024)：2024 年中国电池包均价，美元/kWh
lines = []


def be(x):
    save = (x.reg_default - x["reg_a0.5"]).sum()
    hours = (24 * (x["inv_a0.5"] - x.inv_default)).sum()
    return save / hours, ((x["inv_a0.5"] - x.inv_default) / x.cap_kwh).mean()


for lab, X in (("全网格（库存 2—8 h，功率 1.1—4 倍）", P), ("数据锚定网格（库存 4/6/8 h，功率 2.5/4 倍）", P[P.cap.isin([4, 6, 8]) & P.pw.isin([2.5, 4.0])])):
    h, inv = be(X)
    per = X.groupby(["cap", "pw"]).apply(lambda x: be(x)[0], include_groups=False)
    yrs = PACK_USD * CNY_PER_USD / h / 8760
    lines.append(f"{lab}：合并盈亏平衡持有成本 {h:.4f} 元/(kWh·h)，各配置 {per.min():.4f}—{per.max():.4f}；中点计划多持有平均库存占库容 {100 * inv:.1f}%；"
                 f"按电池包 {PACK_USD:.0f} 美元/kWh × {CNY_PER_USD} = {PACK_USD * CNY_PER_USD:.0f} 元/kWh，全部价值摊入满电持有时间的盈亏平衡寿命 {yrs:.1f} 年")
txt = "\n".join(lines)
(ROOT / "results" / "breakeven_profit.txt").write_text(txt, encoding="utf-8")
print(txt)
