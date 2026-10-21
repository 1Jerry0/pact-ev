"""冒烟检查：各模型 3 月的 CRPS（相对 naive），与预实验量级对照；检查行数与缺失。"""
import sys
import numpy as np
import pandas as pd
import cfg
import sim

ds = sys.argv[1] if len(sys.argv) > 1 else "mp"
rows = []
for f in sorted((cfg.OUT / "fc").glob(f"{ds}_*.parquet")):
    F = pd.read_parquet(f)
    F = F[(F.t >= "2024-03-01") & (F.t < "2024-04-01")] if ds == "mp" else F[F.t >= "2022-10-15"]
    F = F.dropna(subset=["y"])
    for s, g in F.groupby("series"):
        rows.append(dict(file=f.stem, series=s, n=len(g), nan_mean=int(g["mean"].isna().sum()),
                         crps=float(sim.crps_q(g.y.to_numpy(), g[cfg.QC].to_numpy()).mean()),
                         mae=float(np.abs(g.y - g["mean"]).mean())))
R = pd.DataFrame(rows)
base = R[R.file == f"{ds}_naive"].set_index("series").crps
R["rel_crps"] = R.crps / R.series.map(base)
print(R.pivot_table(index="file", columns="series", values="rel_crps").round(3).to_string())
print("行数与缺失：", R.groupby("file")[["n", "nan_mean"]].sum().to_dict("index"))
