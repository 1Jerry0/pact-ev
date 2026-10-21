"""电池级仿真的跨日连续版本：状态（每块电池的缺电量、是否在充）逐日结转，不再每天从半满重启、也不做逐日的日末估值。
检验"规则 ≈ 调参后的预测管线"是不是每日重启与日末估值造成的。
做法：
  1. 每个固定策略（66 个阈值对；24 个计划候选 = 预测水平 × 选解 × 裕量；换下即充）各自从 4 月 1 日连续仿真到年底，记逐日成本（电费 + 罚金）；
     计划候选在每天 0 点从实际状态出发、用日前预测排当天计划（exp_bl_mpc.replan，只排一次），
     并且在晚间谷段（当天最后一个谷价时段，22–24 时）与规则一样把电池充满——日前 LP 的日末估值按谷价计，对"22 点后充还是次日凌晨充"无差异，
     不加这一条时最小库存解会在午夜把站留空（附一个不加的对照，说明这一人为因素有多大）。
  2. 滚动策略：每天按各固定策略在 D−29…D−2 的连续仿真成本选一个（与重启设定下的调参办法相同），再连续仿真这条切换策略；评估 5 月 1 日起。
同时做一个数值检查：电池级 LP 的最大库存解在每一步的满电块数、库存位置是否都不低于最小库存解（格结构）。
配置：6 小时库存、2.5 倍功率、罚价 ×3，单块 8/12/18/25/50 kW。用法：python exp_bl_carry.py；python exp_bl_carry.py ana"""
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
from exp_packs import TH
from exp_bl import nsteps, load_events, lp_bl, position, NS, ETA
from exp_bl_mpc import replan
from exp_packs_value import rel_ci, START

OUT = HERE.parents[2] / "results" / "profit_basis"
S = cfg.BSS
GRID = [((6, 2.5, 3), pc) for pc in (8.0, 12.0, 18.0, 25.0, 50.0)]
QL = (0.6, 0.7, 0.8, 0.9)
CANDS = [f"mean@{mg}|{tie}" for mg in (0, 1, 2) for tie in ("min", "mid", "max", "default")] + [f"q{q}@0|{tie}" for q in QL for tie in ("min", "mid", "max")]
PIC = [f"{lv}@{mg}|{tie}" for lv in ("pih", "pi") for mg in (0, 1, 2) for tie in ("mid", "max")]      # 上界：实际逐小时需求 / 实际逐 15 分钟到达
POLS = [f"rule{i}" for i in range(len(TH))] + CANDS + ["full", "mean@0|max#nofill", "mean@0|min#nofill"]


def sim_day(need, on, ev_step, ev_e, price, pen, C, pc, target):
    """与 exp_bl.run_day_bl 相同的一天，但状态由外部传入并传出，不做日末估值。返回 (电费 + 罚金, 未满足电量)。"""
    add = ETA * pc * 0.25
    cost = unserved = 0.0
    k = 0
    for step in range(NS):
        h = step // 4
        ready = need <= 1e-9
        on &= ~ready
        want = int(round(target[step])) - int(ready.sum())
        want = max(min(want, C, int((~ready).sum())), 0)
        n_on = int(on.sum())
        if want > n_on:
            idle = np.flatnonzero(~ready & ~on)
            on[idle[np.argsort(need[idle])[:want - n_on]]] = True
        elif want < n_on:
            act = np.flatnonzero(on)
            on[act[np.argsort(-need[act])[:n_on - want]]] = False
        e = np.minimum(need[on], add)
        need[on] -= e
        cost += price[h] * e.sum() / ETA
        while k < len(ev_step) and ev_step[k] == step:
            rd = np.flatnonzero(need <= 1e-9)
            if len(rd):
                need[rd[0]] = ev_e[k]
                on[rd[0]] = False
            else:
                unserved += ev_e[k]
                cost += pen[h] * ev_e[k]
            k += 1
    return cost, unserved


def target_of(pol, need, on, lv, price, pen, K, C, n, ebar, add, cls):
    if pol.startswith("rule"):
        a, b = TH[int(pol[4:])]
        return np.where(cls == 0, K, np.where(cls == 1, a * K, b * K))
    if pol == "full":
        return np.full(NS, float(K))
    cand, _, flag = pol.partition("#")
    nm, rest = cand.split("@")
    mg, tie = rest.split("|")
    tgt = replan(0, need, on & (need > 1e-9), lv[nm], price, pen, K, C, n + int(mg), ebar, add, tie)
    if tgt is None:
        return np.full(NS, float(K))
    tgt = np.array(tgt, float)
    if flag != "nofill":
        v = NS
        while v > 0 and cls[v - 1] == 0:
            v -= 1
        tgt[v:] = K                                               # 晚间谷段充满，与规则相同
    return tgt


def run_policy(par, pc, days, EV, FCS, PR, hmean, ebar, chooser):
    """chooser：策略名（固定策略），或 {日: 策略名}（切换策略）。返回逐日 (成本, 未满足, 日末站内缺电量)。"""
    cap, pmax = par[0] * hmean, par[1] * hmean
    K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)
    add = ETA * pc * 0.25
    need, on = np.r_[np.zeros(K // 2), np.full(K - K // 2, ebar)], np.zeros(K, bool)
    out = []
    for d in days:
        price, tot = PR[d]
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        cls = np.repeat(np.searchsorted(np.unique(np.round(price, 6)), np.round(price, 6)), 4)
        es, ee = EV[d][0], np.minimum(EV[d][1], 1.6 * ebar)
        lv = {nm: np.repeat(np.maximum(f, 0) / ebar / 4, 4) for nm, f in FCS[d].items()}
        lv["pi"] = np.bincount(es, minlength=NS).astype(float)
        pol = chooser if isinstance(chooser, str) else chooser[d]
        tgt = target_of(pol, need, on, lv, price, pen, K, C, n, ebar, add, cls)
        c, u = sim_day(need, on, es, ee, price, pen, C, pc, tgt)
        out.append((c, u, float(need.sum())))
    return np.array(out)


def lattice(par, pc, days, FCS, PR, hmean, ebar):
    cap, pmax = par[0] * hmean, par[1] * hmean
    K, C, n = max(int(round(cap / ebar)), 2), max(int(pmax // pc), 1), nsteps(pc, ebar)
    out = []
    for d in days:
        price, tot = PR[d]
        pen = par[2] * tot * (1 + 1e-5 * (23 - np.arange(24)))
        dd = np.repeat(np.maximum(FCS[d]["mean"], 0) / ebar / 4, 4)
        x0, f0 = lp_bl(dd, price, pen, K, C, n, ebar)
        xl, _ = lp_bl(dd, price, pen, K, C, n, ebar, 1.0, f0)
        xh, _ = lp_bl(dd, price, pen, K, C, n, ebar, -1.0, f0)
        out.append((float((xh[NS:2 * NS + 1] - xl[NS:2 * NS + 1]).min()), float((position(xh, n) - position(xl, n)).min()), float((xh[NS:2 * NS + 1] - xl[NS:2 * NS + 1]).mean())))
    return out


def roll_pick(M, days, fb):
    ks = {}
    for d in days:
        past = M.loc[d - pd.Timedelta(days=29): d - pd.Timedelta(days=2)]
        ks[d] = past.sum().idxmin() if len(past) >= 7 else fb
    return ks


if __name__ == "__main__":
    EV, ebar = load_events()
    H = D.mp_hourly()
    hmean = float(H[S].loc[:cfg.MP_DEV_END].mean())
    FQ = pd.read_parquet(cfg.OUT / "fc" / "mp_lgb_local.parquet")
    FQ = FQ[FQ.series == S].set_index("t")
    FCS, PR = {}, {}
    for d in pd.date_range(cfg.MP_EVAL_START, cfg.MP_FC_END, freq="D"):
        idx = pd.date_range(d, periods=24, freq="h")
        f, Q = FQ["mean"].reindex(idx).to_numpy(), FQ[cfg.QC].reindex(idx).to_numpy()
        if d in EV and np.isfinite(f).all() and np.isfinite(Q).all() and np.isfinite(H[S].reindex(idx).to_numpy()).all():
            FCS[d] = {"mean": f, **{f"q{q}": np.array([np.interp(q, cfg.QS, row) for row in Q]) for q in QL}, "pih": H[S].reindex(idx).to_numpy()}
            PR[d] = (D.price(S, d.month), HK(D.price("A7", d.month, "total")))
    days = sorted(FCS)
    ev = [d for d in days if d >= START]
    NJ = int(os.environ.get("NJ", 7))
    POLS = POLS + PIC
    res = Parallel(n_jobs=NJ, batch_size=1, verbose=1)(delayed(run_policy)(par, pc, days, EV, FCS, PR, hmean, ebar, p) for par, pc in GRID for p in POLS)
    lat = Parallel(n_jobs=NJ, batch_size=1)(delayed(lattice)(par, pc, ev, FCS, PR, hmean, ebar) for par, pc in GRID)
    L = np.array([r for rr in lat for r in rr])
    base = pd.read_csv(OUT / "exp_bl.csv")
    lo = min(PR[d][0].min() for d in days[-7:])
    val = lambda need_end, K: (lambda gap: gap * lo / ETA if gap > 0 else gap * lo * ETA)(need_end - K * ebar / 2)
    lines = ["电池级仿真，状态跨日结转（期末电量差按谷价估值）；相对滚动规则的总成本差 %，括号内为每日重启设定下的对应值（exp_bl.csv）"]
    rows, i, daily_rows = [], 0, []
    for par, pc in GRID:
        K = max(int(round(par[0] * hmean / ebar)), 2)
        R = {p: res[i + j] for j, p in enumerate(POLS)}
        i += len(POLS)
        Mr = pd.DataFrame({p: R[p][:, 0] for p in POLS if p.startswith("rule")}, index=days)
        Mf = pd.DataFrame({p: R[p][:, 0] for p in CANDS}, index=days)
        kr = roll_pick(Mr, days, Mr.loc[:pd.Timestamp("2024-04-28")].sum().idxmin())
        kf = roll_pick(Mf, days, "mean@0|mid")
        kph = roll_pick(pd.DataFrame({p: R[p][:, 0] for p in PIC if p.startswith("pih@")}, index=days), days, "pih@0|mid")
        kpi = roll_pick(pd.DataFrame({p: R[p][:, 0] for p in PIC if p.startswith("pi@")}, index=days), days, "pi@0|mid")
        sw = Parallel(n_jobs=4)(delayed(run_policy)(par, pc, days, EV, FCS, PR, hmean, ebar, k) for k in (kr, kf, kph, kpi))
        m = np.array([d >= START for d in days])
        tot = lambda a: a[m, 0].sum() + val(a[-1, 2], K) - val(a[np.flatnonzero(m)[0] - 1, 2], K)       # 评估期内的成本 + 期初期末站内电量差的估值
        ref = sw[0]
        daily_rows.append(pd.DataFrame({"day": days, "cfg": str(par), "pc": pc, "rule_roll": ref[:, 0], "fc_roll": sw[1][:, 0], "pih_roll": sw[2][:, 0], "pi_roll": sw[3][:, 0],
                                        **{k: R[k][:, 0] for k in ("mean@0|min", "mean@0|default", "mean@0|mid", "mean@0|max", "full")}}))
        b = base[(base.cfg == str(par)) & (base.pc == pc)].iloc[0]
        s = f"{par} {pc:.0f} kW（{int(m.sum())} 天）：滚动规则 {tot(ref) / m.sum():.0f} 元/日（重启 {b.cost_rule:.0f}），未满足 {100 * ref[m, 1].sum() / sum(np.minimum(EV[d][1], 1.6 * ebar).sum() for d in ev):.2f}%；"
        cmp = [("调参管线（滚动选）", sw[1], b.fc_roll), ("实际逐时需求，调参", sw[2], np.nan), ("完全信息，调参", sw[3], b.pi_roll), ("最小库存解", R["mean@0|min"], b.fc_min), ("默认解", R["mean@0|default"], b.fc_default), ("中点解", R["mean@0|mid"], b.fc_mid),
               ("最大库存解", R["mean@0|max"], b.fc_max), ("换下即充", R["full"], b.full), ("最大库存解，晚谷不充满", R["mean@0|max#nofill"], np.nan), ("最小库存解，晚谷不充满", R["mean@0|min#nofill"], np.nan)]
        for nm, a, bv in cmp:
            mm, l_, h_ = rel_ci(a[m, 0], ref[m, 0])
            s += f" {nm} {100 * (tot(a) / tot(ref) - 1):+.1f} [{100 * l_:+.1f}, {100 * h_:+.1f}]（{bv:+.1f}）；"
            rows.append(dict(cfg=str(par), pc=pc, pol=nm, rel=100 * (tot(a) / tot(ref) - 1), lo=100 * l_, hi=100 * h_, restart=bv, rule=tot(ref) / m.sum()))
        mm, l_, h_ = rel_ci(R["mean@0|max"][m, 0], R["mean@0|min"][m, 0])
        s += f" 最大 vs 最小 {100 * mm:+.1f} [{100 * l_:+.1f}, {100 * h_:+.1f}]；"
        mm, l_, h_ = rel_ci(R["mean@0|max"][m, 0], R["mean@0|default"][m, 0])
        s += f" 最大 vs 默认 {100 * mm:+.1f} [{100 * l_:+.1f}, {100 * h_:+.1f}]；"
        bestf = min(CANDS, key=lambda p: R[p][m, 0].sum())
        bestr = min(Mr.columns, key=lambda p: R[p][m, 0].sum())
        s += f" 事后最好的固定计划候选 {bestf} {100 * (tot(R[bestf]) / tot(ref) - 1):+.1f}，事后最好的固定阈值 {100 * (tot(R[bestr]) / tot(ref) - 1):+.1f}；管线常选 {pd.Series([kf[d] for d in ev]).value_counts().head(3).to_dict()}"
        lines.append(s)
    lines.append(f"格结构检查（{len(L)} 个日 × 配置，重启设定下均值预测的日前 LP）：最大库存解逐步满电块数 ≥ 最小库存解的占比 {100 * (L[:, 0] > -1e-6).mean():.1f}%（最小差 {L[:, 0].min():.2e}）；"
                 f"库存位置逐步 ≥ 的占比 {100 * (L[:, 1] > -1e-6).mean():.1f}%（最小差 {L[:, 1].min():.2e}）；平均多出的满电块数 {L[:, 2].mean():.1f}")
    txt = "\n".join(lines)
    (OUT / "exp_bl_carry.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "exp_bl_carry.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_rows).to_parquet(OUT / "bl_carry_daily.parquet")
    print(txt)
