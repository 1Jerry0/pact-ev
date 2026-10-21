"""电池级（每日重启，6 h、2.5×、×3）调参管线相对规则的逐月差异，及同月 LightGBM 的精度、偏差与需求水平——解释前后两半符号相反的原因。"""
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "formal" / "code"))
sys.path.insert(0, str(HERE.parent))
import numpy as np
from profit_kappa import HK  # 利润口径：缺供按逐小时零售总价
import pandas as pd
import cfg
import data as D
from exp_packs_value import pick, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
H = D.mp_hourly()[S]
F = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet", columns=["series", "t", "mean"])
F = F[F.series == S].set_index("t")["mean"]
E = pd.DataFrame({"y": H, "f": F}).dropna()
E = E[E.index >= START]
mon = E.groupby(E.index.month).apply(lambda g: pd.Series(dict(y=g.y.mean(), nmae=(g.y - g.f).abs().mean() / g.y.mean(), bias=(g.f - g.y).mean() / g.y.mean(),
                                                                cv=g.y.groupby(g.index.hour).std().div(g.y.groupby(g.index.hour).mean()).mean())))
RU = pd.read_parquet(OUT / "bl_rule.parquet")
FC = pd.read_parquet(OUT / "bl_fc.parquet")
rows = {}
for pc in (8.0, 12.0, 18.0, 25.0, 50.0):
    c = "(6, 2.5, 3)"
    g = RU[(RU.cfg == c) & (RU.pc == pc)]
    Mc = g.pivot(index="day", columns="th", values="cost").sort_index()
    h = FC[(FC.cfg == c) & (FC.pc == pc)]
    Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
    lgb = [k for k in Fc.columns if not k.startswith("pi@")]
    pis = [k for k in Fc.columns if k.startswith("pi@")]
    days = Fc.index[Fc.index >= START]
    k0 = Mc[Mc.index < cfg.MP_EVAL_START].sum().idxmin()
    rule = pd.Series([Mc.loc[x, pick(Mc, x, k0)] for x in days], index=days)
    tun = pd.Series([Fc.loc[x, pick(Fc[lgb], x, "mean@0|mid")] for x in days], index=days)
    pit = pd.Series([Fc.loc[x, pick(Fc[pis], x, "pi@0|mid")] for x in days], index=days)
    hind = Mc.loc[days].min(axis=1)
    rows[f"{pc:.0f}kW 管线"] = (tun.groupby(tun.index.month).sum() / rule.groupby(rule.index.month).sum() - 1) * 100
    rows[f"{pc:.0f}kW 完全信息"] = (pit.groupby(pit.index.month).sum() / rule.groupby(rule.index.month).sum() - 1) * 100
    rows[f"{pc:.0f}kW 事后最佳阈值"] = (hind.groupby(hind.index.month).sum() / rule.groupby(rule.index.month).sum() - 1) * 100
T = pd.DataFrame(rows).join(mon)
txt = "逐月（%，相对滚动规则）与 LightGBM 精度（nMAE、相对偏差）、平均需求（kWh/h）、逐时变异系数：\n" + T.round(2).to_string()
(OUT / "ana_split_month.txt").write_text(txt, encoding="utf-8")
print(txt)
