"""跨日连续的电池级仿真（exp_bl_carry.py 的做法）推广到：
  a7x  A7 的另外四组（4 小时库存 × 12 / 25 kW；罚价 ×10 × 12 / 25 kW）；
  a8   第二座换电站 A8（15 块电池各占一个充电位，罚价 ×3 / ×10；8 块电池 3 个充电位，×3；1 小时充满）；
  uev  60 个深圳区的需求曲线（6 小时库存、2.5 倍功率、罚价 ×3，12 / 25 kW），事件的拆法与 exp_bl_uev.py 逐例相同。
每个固定策略（66 个阈值对、24 个计划候选、换下即充）连续仿真全程，再按 D−29…D−2 的连续仿真成本逐日滚动选，连续仿真切换策略。
用法：python exp_bl_carry2.py a7x | a8 | uev [区数]；python exp_bl_carry2.py ana_uev"""
import os
import pathlib
import sys
import zlib
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
from exp_bl import load_events, ETA
from exp_bl_carry import run_policy, roll_pick, POLS, CANDS, QL
from exp_packs_value import rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
RAW = ROOT / "data" / "raw" / "mp_evdata"
KEEP = ["mean@0|min", "mean@0|default", "mean@0|mid", "mean@0|max", "full"]


def study(tag, par, pc, days, EV, FCS, PR, hmean, ebar, start, nj=1):
    """返回逐日成本表（评估期）：rule_roll、fc_roll、若干固定策略；以及各自含期初期末电量估值的总成本。"""
    run = lambda p: run_policy(par, pc, days, EV, FCS, PR, hmean, ebar, p)
    res = Parallel(n_jobs=nj, batch_size=1)(delayed(run_policy)(par, pc, days, EV, FCS, PR, hmean, ebar, p) for p in POLS[:-2]) if nj > 1 else [run(p) for p in POLS[:-2]]
    R = dict(zip(POLS[:-2], res))
    Mr = pd.DataFrame({p: R[p][:, 0] for p in R if p.startswith("rule")}, index=days)
    Mf = pd.DataFrame({p: R[p][:, 0] for p in CANDS}, index=days)
    kr, kf = roll_pick(Mr, days, "rule0"), roll_pick(Mf, days, "mean@0|mid")
    R["rule_roll"], R["fc_roll"] = run(kr), run(kf)
    K = max(int(round(par[0] * hmean / ebar)), 2)
    lo = min(PR[d][0].min() for d in days[-7:])
    val = lambda need_end: (lambda gap: gap * lo / ETA if gap > 0 else gap * lo * ETA)(need_end - K * ebar / 2)
    m = np.array([d >= start for d in days])
    i0 = np.flatnonzero(m)[0]
    tot = lambda a: a[m, 0].sum() + val(a[-1, 2]) - val(a[i0 - 1, 2])
    names = ["rule_roll", "fc_roll"] + KEEP
    daily = pd.DataFrame({p: R[p][m, 0] for p in names}, index=[d for d in days if d >= start])
    bestf = min(CANDS, key=lambda p: R[p][m, 0].sum())
    bestr = min(Mr.columns, key=lambda p: R[p][m, 0].sum())
    dem = sum(np.minimum(EV[d][1], 1.6 * ebar).sum() for d in days if d >= start)
    summ = dict(tag=tag, cfg=str(par), pc=pc, days=int(m.sum()), **{p: tot(R[p]) for p in names}, best_fixed_fc=tot(R[bestf]), best_fixed_rule=tot(R[bestr]),
                short_rule=R["rule_roll"][m, 1].sum() / max(dem, 1e-9), short_fc=R["fc_roll"][m, 1].sum() / max(dem, 1e-9),
                picks=str(pd.Series([kf[d] for d in days if d >= start]).value_counts().head(3).to_dict()))
    return summ, daily


def line(summ, daily, extra=""):
    s = f"{summ['tag']} {summ['cfg']} {summ['pc']:.2f}（{summ['days']} 天）：滚动规则 {summ['rule_roll'] / summ['days']:.1f} /日，未满足 {100 * summ['short_rule']:.2f}%；相对滚动规则 %："
    for p, lab in (("fc_roll", "调参管线"), ("mean@0|min", "最小库存解"), ("mean@0|default", "默认解"), ("mean@0|mid", "中点解"), ("mean@0|max", "最大库存解"), ("full", "换下即充")):
        m, lo, hi = rel_ci(daily[p].to_numpy(), daily["rule_roll"].to_numpy())
        s += f" {lab} {100 * (summ[p] / summ['rule_roll'] - 1):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
    m, lo, hi = rel_ci(daily["mean@0|max"].to_numpy(), daily["mean@0|min"].to_numpy())
    s += f" 最大 vs 最小 {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
    s += f" 事后最好的固定计划候选 {100 * (summ['best_fixed_fc'] / summ['rule_roll'] - 1):+.1f}，事后最好的固定阈值 {100 * (summ['best_fixed_rule'] / summ['rule_roll'] - 1):+.1f}；管线常选 {summ['picks']}{extra}"
    return s


CK = OUT / "carry_uev"


def uev_zone(z, hmean, days_raw, PRm):
    """days_raw: [(day, y, f, Q)]，与 exp_bl_uev.ztask 相同的顺序与随机数流。逐区落盘（断点续跑；多个进程可同时跑，已开工的区跳过）。"""
    CK.mkdir(exist_ok=True)
    fp, lock = CK / f"{z}.pkl", CK / f"{z}.lock"
    if fp.exists():
        return pd.read_pickle(fp)
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        return None
    rng = np.random.default_rng(zlib.crc32(str(z).encode()))
    EV, FCS, PR = {}, {}, {}
    for d, y, f, Q in days_raw:
        n = np.diff(np.r_[0.0, np.round(np.cumsum(y) / 25.0)]).astype(int)
        steps = np.sort(np.concatenate([4 * h + rng.integers(0, 4, n[h]) for h in range(24)])) if n.sum() else np.array([], int)
        if f is not None:
            EV[d] = (steps.astype(int), np.full(len(steps), 25.0))
            FCS[d] = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
            PR[d] = PRm[d.month]
    days = sorted(FCS)
    out = []
    for par, pc in (((6, 2.5, 3), 12.0), ((6, 2.5, 3), 25.0)):
        summ, daily = study(z, par, pc, days, EV, FCS, PR, hmean, 25.0, cfg.UEV_EVAL_START)
        out.append((summ, daily.assign(zone=z, pc=pc)))
    pd.to_pickle(out, fp)
    lock.unlink(missing_ok=True)
    return out


def ana_uev():
    T = pd.read_csv(OUT / "exp_bl_carry_uev.csv")
    T["zone"] = T.tag.astype(str)
    Dd = pd.read_parquet(OUT / "bl_carry_uev_daily.parquet")
    Z = pd.read_csv(OUT / "bl_uev_zones.csv")
    Z["zone"] = Z.zone.astype(str)
    T = T.merge(Z, on="zone")
    old = pd.read_csv(OUT / "exp_bl_uev.csv")
    old["zone"] = old.zone.astype(str)
    lines = [f"UrbanEV {T.zone.nunique()} 个区，跨日连续的电池级仿真；括号内为每日重启设定下的对应值（exp_bl_uev.csv）"]
    for pc, g in T.groupby("pc"):
        o = old[old.pc == pc].set_index("zone").loc[g.zone]
        gain = 100 * (g.fc_roll / g.rule_roll - 1)
        og = 100 * (o.tuned / o.rule - 1)
        s = f"{pc:.0f} kW：调参管线相对规则（各区 %）中位 {gain.median():+.1f}（{og.median():+.1f}），四分位 {gain.quantile(.25):+.1f}…{gain.quantile(.75):+.1f}，管线更便宜的区占 {100 * (gain < 0).mean():.0f}%（{100 * (og < 0).mean():.0f}%）；"
        dd = Dd[Dd.pc == pc]
        norm = {}
        for nm in ("fc_roll", "mean@0|default", "mean@0|max", "mean@0|min", "full", "rule_roll"):
            P = dd.pivot(index="day", columns="zone", values=nm)
            Rf = dd.pivot(index="day", columns="zone", values="rule_roll")
            norm[nm] = (P / Rf.mean()).mean(axis=1)
        for nm, lab in (("fc_roll", "调参管线"), ("mean@0|default", "默认解"), ("mean@0|max", "最大库存解"), ("mean@0|min", "最小库存解"), ("full", "换下即充")):
            m, lo, hi = rel_ci(norm[nm].to_numpy(), norm["rule_roll"].to_numpy())
            s += f"各区等权合并 {lab} {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]；"
        r1, p1 = spearmanr(g.cv, gain)
        r2, p2 = spearmanr(np.log(g.hmean), gain)
        big, small = g.hmean >= 200, g.hmean < 100
        s += (f"收益与逐时变异系数的秩相关 {r1:+.2f}（p={p1:.3f}），与区规模 {r2:+.2f}（p={p2:.3f}）；平均需求 ≥200 kWh/h 的 {int(big.sum())} 个区：管线更便宜的占 {100 * (gain[big] < 0).mean():.0f}%，中位 {gain[big].median():+.1f}；"
              f"<100 的 {int(small.sum())} 个区：管线更贵的占 {100 * (gain[small] > 0).mean():.0f}%，中位 {gain[small].median():+.1f}；"
              f"默认解相对规则中位 {100 * (g['mean@0|default'] / g.rule_roll - 1).median():+.1f}%；最大 vs 最小库存解中位 {100 * (g['mean@0|max'] / g['mean@0|min'] - 1).median():+.1f}%；换下即充相对规则中位 {100 * (g.full / g.rule_roll - 1).median():+.1f}%")
        lines.append(s)
    txt = "\n".join(lines)
    (OUT / "exp_bl_carry_uev.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    mode = sys.argv[1]
    NJ = int(os.environ.get("NJ", 7))
    if mode == "ana_uev":
        ana_uev()
    elif mode in ("a7x", "a8"):
        if mode == "a7x":
            S, pst = cfg.BSS, "A7"
            EV, ebar = load_events()
            grid = [((4, 2.5, 3), 12.0), ((4, 2.5, 3), 25.0), ((6, 2.5, 10), 12.0), ((6, 2.5, 10), 25.0)]
            end, start = cfg.MP_FC_END, START
        else:
            S = pst = "A8"
            ebar = 1.0
            p1 = 1.0 / ETA
            grid = [((11, 12.0, 3), p1), ((11, 12.0, 10), p1), ((6, 2.5, 3), p1)]
            d0 = pd.read_excel(RAW / "charging_session.xlsx", sheet_name="A8")
            d0.columns = ["conn", "disc", "energy"]
            d0["disc"] = pd.to_datetime(d0.disc, errors="coerce")
            a = d0[d0.disc.notna()].sort_values("disc")
            a["day"], a["step"] = a.disc.dt.normalize(), a.disc.dt.hour * 4 + a.disc.dt.minute // 15
            EV = {d: (g.step.to_numpy(), np.ones(len(g))) for d, g in a.groupby("day")}
            end, start = pd.Timestamp("2024-10-31"), START
        H = D.mp_hourly()
        hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
        FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
        FQ = FQ[FQ.series == S].set_index("t")
        FCS, PR = {}, {}
        for d in pd.date_range(cfg.MP_EVAL_START, end, freq="D"):
            idx = pd.date_range(d, periods=24, freq="h")
            f, Q = FQ["mean"].reindex(idx).to_numpy(), FQ[cfg.QC].reindex(idx).to_numpy()
            if np.isfinite(f).all() and np.isfinite(Q).all() and np.isfinite(H[S].reindex(idx).to_numpy()).all() and (d in EV or mode == "a8"):
                if d not in EV:
                    EV[d] = (np.array([], int), np.array([]))
                FCS[d] = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}}
                PR[d] = (D.price(S, d.month), HK(D.price(pst, d.month, "total")))
        days = sorted(FCS)
        lines, rows = [], []
        for par, pc in grid:
            summ, daily = study(mode, par, pc, days, EV, FCS, PR, hmean, ebar, start, nj=NJ)
            lines.append(line(summ, daily))
            rows.append(summ)
            print(lines[-1], flush=True)
        (OUT / f"exp_bl_carry_{mode}.txt").write_text("\n".join(lines), encoding="utf-8")
        pd.DataFrame(rows).to_csv(OUT / f"exp_bl_carry_{mode}.csv", index=False, encoding="utf-8-sig")
    else:
        V = D.uev_volume()
        sel = pd.read_csv(OUT / "bl_uev_zones.csv")
        sel["zone"] = sel.zone.astype(str)
        sel = sel.set_index("zone")
        F = pd.read_parquet(cfg.OUT / "fc" / "uev_lgb_global_cal.parquet")
        F = F[F.series.astype(str).isin(sel.index)]
        PRm = {mo: (D.price("A1", mo), HK(D.price("A1", mo, "total"))) for mo in (9, 10, 11, 12, 1, 2)}
        Tk = []
        for z in sel.index:
            g = F[F.series.astype(str) == z].set_index("t")
            dr = []
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
                dr.append((d, y, f, Q))
            Tk.append((z, float(sel.loc[z, "hmean"]), dr, PRm))
        if len(sys.argv) > 2 and sys.argv[2] == "rev":            # 第二个进程从后往前跑，与第一个进程会合
            Tk = Tk[::-1]
        print(len(Tk), "zones", flush=True)
        res = Parallel(n_jobs=NJ, batch_size=1, verbose=5)(delayed(uev_zone)(*t) for t in Tk)
        if all((CK / f"{t[0]}.pkl").exists() for t in Tk):
            res = [pd.read_pickle(CK / f"{t[0]}.pkl") for t in Tk]
            pd.DataFrame([s for rr in res for s, _ in rr]).to_csv(OUT / "exp_bl_carry_uev.csv", index=False, encoding="utf-8-sig")
            pd.concat([dd.rename_axis("day").reset_index() for rr in res for _, dd in rr]).to_parquet(OUT / "bl_carry_uev_daily.parquet")
            ana_uev()
        else:
            print("未全部完成：", sum((CK / f"{t[0]}.pkl").exists() for t in Tk), "/", len(Tk))
