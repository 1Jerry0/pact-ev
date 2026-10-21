"""Read-only: what does the raw A7 session table imply about the real station's battery slots and charging power?"""
import sys, pathlib
import numpy as np
import pandas as pd

raw = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "mp_evdata"
meta = pd.read_excel(raw / "charging_session.xlsx", sheet_name="站点确定")
print(meta.to_string())
for s in ("A7", "A8"):
    d = pd.read_excel(raw / "charging_session.xlsx", sheet_name=s)
    d.columns = ["conn", "disc", "energy"]
    d["conn"], d["disc"] = pd.to_datetime(d.conn, errors="coerce"), pd.to_datetime(d.disc, errors="coerce")
    d["energy"] = pd.to_numeric(d.energy, errors="coerce")
    d = d.dropna()
    d["dur_h"] = (d.disc - d.conn).dt.total_seconds() / 3600
    d = d[(d.dur_h > 0) & (d.energy > 0)]
    d["kw"] = d.energy / d.dur_h
    print(f"\n== {s}: sessions {len(d)}, energy/session median {d.energy.median():.1f} kWh (p10 {d.energy.quantile(.1):.1f}, p90 {d.energy.quantile(.9):.1f})")
    print("duration h quantiles", d.dur_h.quantile([.05, .25, .5, .75, .95]).round(2).tolist(), "| avg kW quantiles", d.kw.quantile([.05, .25, .5, .75, .95]).round(1).tolist())
    ev = pd.concat([pd.Series(1, index=d.conn), pd.Series(-1, index=d.disc)]).sort_index().cumsum()
    print("concurrent connected batteries: median", int(ev.median()), "p90", int(ev.quantile(.9)), "p99", int(ev.quantile(.99)), "max", int(ev.max()))
    pw = pd.concat([pd.Series(d.kw.to_numpy(), index=d.conn), pd.Series(-d.kw.to_numpy(), index=d.disc)]).sort_index().cumsum()
    print("concurrent charging power kW: median", round(pw.median(), 1), "p90", round(pw.quantile(.9), 1), "p99", round(pw.quantile(.99), 1), "max", round(pw.max(), 1))
    days = (d.disc.max().normalize() - d.disc.min().normalize()).days + 1
    hmean = d.energy.sum() / (days * 24)
    print(f"mean hourly energy {hmean:.1f} kWh/h over {days} days; p99 concurrent power / mean = {pw.quantile(.99) / hmean:.2f}; max/mean = {pw.max() / hmean:.2f}")
    # hourly charging energy actually drawn (spread uniformly over each session) by hour of day vs swap-out energy by hour of day
    by_disc = d.groupby(d.disc.dt.hour).energy.sum() / d.energy.sum()
    by_conn = d.groupby(d.conn.dt.hour).energy.sum() / d.energy.sum()
    print("share of energy by hour of disconnection:", by_disc.round(3).tolist())
    print("share of energy by hour of connection   :", by_conn.round(3).tolist())
