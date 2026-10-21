"""电池级"预测 vs 滚动规则"的横截面检验：把 UrbanEV 各区的逐时充电量当作换电需求（与正文小时模型里的 UrbanEV 换电复现同一做法），
在电池级仿真里跑 60 个区（按逐时变异系数分层抽取，平均小时需求 40–800 kWh），看预测管线相对规则的收益是否随需求的规律性变化。
只有两座真实换电站时，"需求越不规律、预测越值钱"的说法只有 n = 2；这里用 60 条真实需求曲线补横截面证据（到达事件是由逐时电量确定性拆出来的，
每小时的换电次数 = 累计电量 / 25 kWh 的取整差分，小时内时刻随机；不额外加入泊松噪声）。
配置：库容 6 小时、功率 2.5 倍、罚价 ×3；充电 12 kW（约 2.25 h）与 25 kW（约 1 h）。电价用上海分时电价（同正文 UrbanEV 决策实验）。
预测：UrbanEV 的全局 LightGBM（校准后分位数）；10 月为调参历史，评估期 11 月 1 日–2 月 28 日（120 天）。用法：python exp_bl_uev.py"""
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
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import cfg
import data as D
from exp_bl import task, QL
from exp_packs_value import pick, rel_ci

OUT = HERE.parents[2] / "results" / "profit_basis"
GRID = [((6, 2.5, 3), 12.0), ((6, 2.5, 3), 25.0)]
EBAR = 25.0
NZ = 60


def ztask(z, hmean, days):
    """days: [(day, y, price, tot, f, Q)]（f、Q 在无预测的日子为 None）。"""
    import zlib
    rng = np.random.default_rng(zlib.crc32(str(z).encode()))
    if os.environ.get("UEVSMOKE"):
        days = days[50:75]
    rule, rows = [], []
    for d, y, price, tot, f, Q in days:
        n = np.diff(np.r_[0.0, np.round(np.cumsum(y) / EBAR)]).astype(int)
        steps = np.sort(np.concatenate([4 * h + rng.integers(0, 4, n[h]) for h in range(24)])) if n.sum() else np.array([], int)
        fcs = None if f is None else {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
        a, b = task(d, steps.astype(int), np.full(len(steps), EBAR), y, price, tot, hmean, fcs, EBAR, GRID)
        rule += [(z, *r[:3], r[4], r[6]) for r in a]
        rows += [(z, *r[:3], r[4], r[5]) for r in b]
    return rule, rows


def analyse():
    RU = pd.read_parquet(OUT / "bl_uev_rule.parquet")
    FC = pd.read_parquet(OUT / "bl_uev_fc.parquet")
    Z = pd.read_csv(OUT / "bl_uev_zones.csv").set_index("zone")
    Z.index = Z.index.astype(str)
    res, daily = [], {}
    for (z, pc), g in RU.groupby(["zone", "pc"], sort=False):
        Mc = pd.DataFrame(np.vstack(g.costs.to_list()), index=pd.DatetimeIndex(g.day)).sort_index()
        h = FC[(FC.zone == z) & (FC.pc == pc)]
        Fc = h.pivot(index="day", columns="cand", values="cost").sort_index()
        lgb = [k for k in Fc.columns if not k.startswith("pi@")]
        pis = [k for k in Fc.columns if k.startswith("pi@")]
        days = [d for d in Fc.index if d >= cfg.UEV_EVAL_START]
        k0 = Mc[Mc.index <= cfg.UEV_DEV_END].sum().idxmin()
        ref = np.array([Mc.loc[d, pick(Mc, d, k0)] for d in days])
        tun = np.array([Fc.loc[d, pick(Fc[lgb], d, "mean@0|mid")] for d in days])
        pit = np.array([Fc.loc[d, pick(Fc[pis], d, "pi@0|mid")] for d in days])
        full = g.set_index("day").full.loc[days].to_numpy()
        row = dict(zone=z, pc=pc, cv=Z.loc[z, "cv"], hmean=Z.loc[z, "hmean"], rule=ref.sum(), tuned=tun.sum(), pi=pit.sum(), full=full.sum(),
                   default=Fc.loc[days, "mean@0|default"].sum(), fmax=Fc.loc[days, "mean@0|max"].sum(), fmin=Fc.loc[days, "mean@0|min"].sum(), hind=Mc.loc[days].min(axis=1).sum())
        res.append(row)
        for nm, x in (("rule", ref), ("tuned", tun), ("pi", pit), ("default", Fc.loc[days, "mean@0|default"].to_numpy()), ("fmax", Fc.loc[days, "mean@0|max"].to_numpy()), ("fmin", Fc.loc[days, "mean@0|min"].to_numpy())):
            daily.setdefault((pc, nm), []).append(pd.Series(x / ref.mean(), index=days))        # 各区按自己的规则日均成本归一，避免大区主导
    T = pd.DataFrame(res)
    T.to_csv(OUT / "exp_bl_uev.csv", index=False, encoding="utf-8-sig")
    lines = []
    for pc, g in T.groupby("pc"):
        gain = 100 * (g.tuned / g.rule - 1)
        s = f"{pc:.0f} kW，{len(g)} 个区：调参管线相对规则（各区 %）中位 {gain.median():+.1f}，四分位 {gain.quantile(.25):+.1f}…{gain.quantile(.75):+.1f}，管线更便宜的区占 {100 * (gain < 0).mean():.0f}%；"
        for nm in ("tuned", "default", "fmax", "fmin", "pi"):
            a = pd.concat(daily[(pc, nm)], axis=1).mean(axis=1)
            b = pd.concat(daily[(pc, "rule")], axis=1).mean(axis=1)
            m, lo, hi = rel_ci(a.to_numpy(), b.to_numpy())
            s += f"各区等权合并 {nm} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
        r1, p1 = spearmanr(g.cv, gain)
        r2, p2 = spearmanr(np.log(g.hmean), gain)
        pig = 100 * (g.pi / g.rule - 1)
        r3, p3 = spearmanr(g.cv, pig)
        s += f"收益与逐时变异系数的秩相关 {r1:+.2f}（p={p1:.3f}），与区规模（log 平均需求）{r2:+.2f}（p={p2:.3f}）；完全信息上限中位 {pig.median():+.1f}，其与变异系数的秩相关 {r3:+.2f}（p={p3:.3f}）；"
        ter = pd.qcut(g.cv, 3, labels=["规律", "中等", "不规律"])
        s += "按变异系数三分位的收益中位：" + "，".join(f"{k} {v:+.1f}" for k, v in gain.groupby(ter, observed=True).median().items())
        s += f"；最大 vs 最小库存解（各区中位）{100 * (g.fmax / g.fmin - 1).median():+.1f}%；默认解相对规则中位 {100 * (g.default / g.rule - 1).median():+.1f}%；换下即充相对规则中位 {100 * (g.full / g.rule - 1).median():+.1f}%"
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_uev.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ana":
        analyse()
        sys.exit()
    V = D.uev_volume()
    dev = V.loc[:cfg.UEV_DEV_END]
    hm = dev.mean()
    cv = (dev.groupby(dev.index.hour).std() / dev.groupby(dev.index.hour).mean()).mean()
    ok = hm[(hm >= 40) & (hm <= 800)].index
    Z = pd.DataFrame({"hmean": hm[ok], "cv": cv[ok]}).sort_values("cv")
    sel = Z.iloc[np.unique(np.round(np.linspace(0, len(Z) - 1, NZ)).astype(int))]
    sel.rename_axis("zone").to_csv(OUT / "bl_uev_zones.csv")
    F = pd.read_parquet(cfg.OUT / "fc" / "uev_lgb_global_cal.parquet")
    F = F[F.series.astype(str).isin(sel.index)]
    PR = {mo: (D.price("A1", mo), HK(D.price("A1", mo, "total"))) for mo in (9, 10, 11, 12, 1, 2)}
    Tk = []
    for z in sel.index:
        g = F[F.series.astype(str) == z].set_index("t")
        days = []
        for d in pd.date_range("2022-09-01", cfg.UEV_FC_END, freq="D"):
            idx = pd.date_range(d, periods=24, freq="h")
            y = V[z].reindex(idx).to_numpy()
            if not np.isfinite(y).all():
                continue
            f = Q = None
            if d >= pd.Timestamp("2022-10-01"):
                f, Q = g["mean"].reindex(idx).to_numpy(), g[cfg.QC].reindex(idx).to_numpy()
                if not (np.isfinite(f).all() and np.isfinite(Q).all()):
                    f = Q = None
                    if d >= cfg.UEV_EVAL_START:
                        continue
            days.append((d, y, *PR[d.month], f, Q))
        Tk.append((z, float(sel.loc[z, "hmean"]), days))
    if len(sys.argv) > 1:
        Tk = Tk[:int(sys.argv[1])]
    print(len(Tk), "zones", flush=True)
    res = Parallel(n_jobs=7, batch_size=1, verbose=1)(delayed(ztask)(*t) for t in Tk)
    pd.DataFrame([r for a, _ in res for r in a], columns=["zone", "day", "cfg", "pc", "costs", "full"]).to_parquet(OUT / "bl_uev_rule.parquet")
    pd.DataFrame([r for _, b in res for r in b], columns=["zone", "day", "cfg", "pc", "cand", "cost"]).to_parquet(OUT / "bl_uev_fc.parquet")
    analyse()
