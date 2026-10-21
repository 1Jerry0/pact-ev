"""通用工具：日志、容器内可用 CPU 核数、并行执行、可续跑的分块保存。"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import math
import time
import pathlib
import pandas as pd
from joblib import Parallel, delayed
import cfg


def log(msg, name="run"):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(cfg.LOGS / f"{name}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_cpus():
    """容器实际分到的核数（cgroup 配额），避免在 AutoDL 等容器里按宿主机核数超额开进程。"""
    env = os.environ.get("N_JOBS")
    if env:
        return max(1, int(env))
    n = os.cpu_count() or 1
    try:
        q = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q[0] != "max":
            n = min(n, math.floor(int(q[0]) / int(q[1])))
    except Exception:
        try:
            quota = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = min(n, math.floor(quota / period))
        except Exception:
            pass
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, n)


def pmap(fn, tasks, n_jobs=None, desc="", name="run", batch_size="auto"):
    """并行执行 fn(*task)，按完成顺序返回结果；每完成约 5% 记一次日志。"""
    n_jobs = n_jobs or n_cpus()
    tasks = list(tasks)
    out, t0 = [], time.time()
    step = max(1, len(tasks) // 20)
    gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered", batch_size=batch_size)(delayed(fn)(*t) for t in tasks)
    for i, r in enumerate(gen, 1):
        out.append(r)
        if i % step == 0 or i == len(tasks):
            el = time.time() - t0
            log(f"{desc} {i}/{len(tasks)}，已用 {el / 60:.1f} 分钟，预计剩余 {el / i * (len(tasks) - i) / 60:.1f} 分钟", name)
    return out


def done(path):
    return pathlib.Path(path).exists()


def save(df, path):
    path = pathlib.Path(path)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_glob(pattern):
    files = sorted(cfg.OUT.glob(pattern))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
