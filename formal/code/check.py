"""运行后的硬性校验：遗憾非负、正对照严格为 0、负对照对交换不变、预测无缺失。任何一项失败都在日志中标记 FAIL。"""
import numpy as np
import pandas as pd
import cfg
from util import log


def main():
    ok = True
    for f in sorted((cfg.OUT / "fc").glob("*.parquet")):
        F = pd.read_parquet(f, columns=["t", "y", "mean"] + cfg.QC)
        F = F[(F.t >= (cfg.MP_EVAL_START if f.name.startswith("mp_") else cfg.UEV_EVAL_START)) & F.y.notna()]
        bad = int(F[["mean"] + cfg.QC].isna().any(axis=1).sum())
        if bad:
            ok = False
            log(f"FAIL 预测缺失 {f.name}: {bad} 行", "check")
    for name in ("mp_sched", "uev_sched"):
        fp = cfg.OUT / ("dec" if name.startswith("mp") else "uev") / f"{name}.parquet"
        if fp.exists():
            S = pd.read_parquet(fp)
            tol = 1e-6 * S.opt.abs().clip(lower=1)
            neg = int((S.regret < -tol).sum())
            log(f"{'FAIL' if neg else 'OK'} {name}: {len(S):,} 行，负遗憾 {neg} 行，最小遗憾 {S.regret.min():.3e}", "check")
            ok &= neg == 0
    fp = cfg.OUT / "mech" / "ctrl.parquet"
    if fp.exists():
        C = pd.read_parquet(fp)
        pc = float(np.abs(C[["pc_current", "pc_open"]]).to_numpy().max())
        nc = float(C.neg_ctrl.max())
        log(f"{'FAIL' if pc > 1e-3 else 'OK'} 正对照最大偏差 {pc:.2e} 元；{'FAIL' if nc > 1e-6 else 'OK'} 负对照最大偏差 {nc:.2e}", "check")
        ok &= pc <= 1e-3 and nc <= 1e-6
        M = pd.read_parquet(cfg.OUT / "mech" / "mech.parquet")
        neg = int((M[["reg_current", "reg_open"]] < -1e-5).to_numpy().sum())
        log(f"{'FAIL' if neg else 'OK'} E4 负遗憾 {neg} 个", "check")
        ok &= neg == 0
    log("全部校验通过" if ok else "存在 FAIL，请查看日志", "check")
    return ok


if __name__ == "__main__":
    main()
