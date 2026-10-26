"""Run the stage-2 analyses (everything after the formal forecasting run) in the order that produced the paper's results.

  python run_pipeline.py --list                      # show all steps with their numbers
  python run_pipeline.py                             # run every group in order
  python run_pipeline.py --group hourly battery      # run selected groups
  python run_pipeline.py --from 40                   # resume from step 40
  python run_pipeline.py --jobs 16                   # worker processes for scripts that parallelise (env NJ)

Each step's console output goes to results/logs/<step>.txt. Stops at the first failing step unless --keep-going.
Prerequisite: formal/out/ produced by formal/run_all.sh (stage 1). See README.md and PAPER_MAP.md."""
import argparse
import datetime
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DIRS = {"s": ROOT / "scripts", "p": ROOT / "scripts" / "profit_basis"}
LOG = ROOT / "results" / "logs"

STEPS = [
    ('prespecified', 's', 'exp_bss.py', ['anchor'], {}),
    ('prespecified', 's', 'exp_bl.py', [], {}),
    ('prespecified', 's', 'exp_bl2.py', ['A'], {}),
    ('prespecified', 's', 'exp_bl2.py', ['B'], {}),
    ('prespecified', 's', 'exp_bl_models.py', ['extra'], {}),
    ('prespecified', 's', 'exp_bl_models.py', ['models'], {}),
    ('prespecified', 's', 'exp_bl_tariffs.py', [], {}),
    ('prespecified', 's', 'exp_bl_tariffs2.py', [], {}),
    ('prespecified', 's', 'exp_bl_profile_x10.py', [], {}),
    ('prespecified', 's', 'penalty_reweight.py', [], {}),
    ('hourly', 's', 'exp_cost_basis.py', [], {}),
    ('hourly', 's', 'exp_cost_basis_st.py', [], {}),
    ('hourly', 's', 'exp_cost_basis_t4.py', ['A7'], {}),
    ('hourly', 's', 'exp_cost_basis_t4.py', ['A8'], {}),
    ('hourly', 's', 'exp_cost_basis_t4.py', ['A7tight'], {}),
    ('hourly', 's', 'ana_cost_basis.py', [], {}),
    ('hourly', 's', 'ana_cost_basis_st.py', [], {}),
    ('hourly', 's', 'ana_cost_basis_t4.py', [], {}),
    ('hourly', 's', 'exp_safety_stock.py', [], {}),
    ('hourly', 's', 'ana_safety_stock.py', [], {}),
    ('hourly', 's', 'rev_dswe_profit.py', [], {}),
    ('hourly', 's', 'rev_dswe_profit.py', [], {'GRID': 'tight'}),
    ('hourly', 's', 'rev_dswe_st_profit.py', [], {}),
    ('hourly', 's', 'exp_h1_profit.py', ['run'], {}),
    ('hourly', 's', 'exp_h1_profit.py', ['ana'], {}),
    ('hourly', 's', 'exp_h2_profit.py', ['run'], {}),
    ('hourly', 's', 'exp_h2_profit.py', ['ana'], {}),
    ('hourly', 's', 'ana_h3_profit.py', [], {}),
    ('hourly', 's', 'ana_h_subsample_profit.py', [], {}),
    ('hourly', 's', 'ana_hourly_decomp_profit.py', [], {}),
    ('hourly', 's', 'exp_storage_profit.py', ['tariffs'], {}),
    ('hourly', 's', 'exp_storage_profit.py', ['q15'], {}),
    ('hourly', 's', 'exp_storage_profit.py', ['ana'], {}),
    ('hourly', 's', 'pilot_naive_weights.py', [], {}),
    ('hourly', 's', 'ana_naive_weights.py', [], {}),
    ('hourly', 's', 'rev_dswe_profit_delta.py', [], {'REL': '0.05'}),
    ('hourly', 's', 'rev_dswe_profit_delta.py', [], {'REL': '0.2'}),
    ('hourly', 's', 'exp_eta_check.py', ['0.93'], {}),
    ('hourly', 's', 'exp_eta_check.py', ['0.95'], {}),
    ('hourly', 's', 'rev_table3_profit.py', [], {}),
    ('hourly', 'p', 'exp_conn.py', [], {}),
    ('hourly', 'p', 'exp_prices_order.py', [], {}),
    ('hourly', 'p', 'exp_eps.py', [], {}),
    ('hourly', 'p', 'exp_leadtime.py', [], {}),
    ('hourly', 'p', 'rev_a8.py', [], {}),
    ('hourly', 'p', 'rev_uev.py', [], {}),
    ('hourly', 'p', 'rev_uev_rule.py', [], {}),
    ('hourly', 'p', 'exp_dfl.py', [], {}),
    ('hourly', 'p', 'exp_dfl.py', ['ana'], {}),
    ('hourly', 'p', 'exp_dfl_learn.py', [], {}),
    ('hourly', 'p', 'exp_dfl_grad.py', [], {}),
    ('hourly', 'p', 'exp_dfl_grad.py', ['ana'], {}),
    ('hourly', 'p', 'exp_solvers.py', [], {}),
    ('hourly', 'p', 'exp_tariffs.py', [], {}),
    ('hourly', 'p', 'buffer_alpha.py', ['A7', 'A8', 'UEV'], {}),
    ('hourly', 'p', 'exp_map.py', [], {}),
    ('hourly', 'p', 'ana_map.py', [], {}),
    ('hourly', 'p', 'exp_prop3.py', [], {}),
    ('hourly', 'p', 'fc_intraday.py', [], {}),
    ('hourly', 'p', 'exp_packs_value.py', [], {}),
    ('hourly', 'p', 'exp_packs_e4.py', [], {}),
    ('hourly', 'p', 'exp_packs_scaled.py', [], {}),
    ('hourly', 's', 'desc_stats.py', [], {}),
    ('hourly', 's', 'check_facts.py', [], {}),
    ('hourly', 's', 'review_a7_capacity.py', [], {}),
    ('battery', 'p', 'exp_bl.py', [], {}),
    ('battery', 'p', 'exp_bl2.py', ['A'], {}),
    ('battery', 'p', 'exp_bl2.py', ['B'], {}),
    ('battery', 'p', 'exp_bl_models.py', ['extra'], {}),
    ('battery', 'p', 'exp_bl_models.py', ['models'], {}),
    ('battery', 'p', 'exp_bl_tariffs.py', [], {}),
    ('battery', 'p', 'exp_bl_tariffs2.py', [], {}),
    ('battery', 'p', 'exp_bl_more.py', ['conn'], {}),
    ('battery', 'p', 'exp_bl_more.py', ['a8'], {}),
    ('battery', 'p', 'exp_bl_a8_profile.py', [], {}),
    ('battery', 'p', 'exp_bl_int.py', [], {}),
    ('battery', 'p', 'exp_bl_rule_free.py', [], {}),
    ('battery', 'p', 'exp_bl_prices.py', [], {}),
    ('battery', 'p', 'exp_bl_prices.py', ['ana'], {}),
    ('battery', 'p', 'ana_bl_prices2.py', [], {}),
    ('battery', 'p', 'exp_bl_mpc.py', [], {}),
    ('battery', 'p', 'exp_bl_mpc.py', ['ana'], {}),
    ('battery', 'p', 'exp_bl_carry.py', [], {}),
    ('battery', 'p', 'exp_bl_carry.py', ['ana'], {}),
    ('battery', 'p', 'exp_bl_carry2.py', ['a7x'], {}),
    ('battery', 'p', 'exp_bl_carry2.py', ['a8'], {}),
    ('battery', 'p', 'exp_bl_carry_models.py', [], {}),
    ('battery', 'p', 'exp_bl_uev.py', [], {}),
    ('battery', 'p', 'exp_bl_uev_profile.py', [], {}),
    ('battery', 'p', 'exp_bl_uev_profile.py', ['ana'], {}),
    ('battery', 'p', 'exp_bl_carry2.py', ['uev'], {}),
    ('battery', 'p', 'exp_bl_carry2.py', ['ana_uev'], {}),
    ('battery', 'p', 'ana_bl_window.py', [], {}),
    ('battery', 'p', 'ana_split_month.py', [], {}),
    ('battery', 'p', 'ana_uev_acc.py', [], {}),
    ('battery', 'p', 'exp_bl_valley.py', [], {}),
    ('battery', 's', 'exp_bl_profit.py', ['main'], {}),
    ('battery', 's', 'exp_bl_profit.py', ['models'], {}),
    ('battery', 's', 'exp_bl_profit.py', ['tariffs'], {}),
    ('battery', 's', 'ana_bl_profit.py', [], {}),
    ('battery', 's', 'exp_bl_rule_free_profit.py', [], {}),
    ('report', 's', 'ana_maxmin_bases.py', [], {}),
    ('report', 's', 'ana_annual_value.py', [], {}),
    ('report', 's', 'ana_breakeven_profit.py', [], {}),
    ('report', 's', 'ana_halves_profit.py', [], {}),
    ('report', 's', 'make_fig7b_csv.py', [], {}),
    ('report', 's', 'rev_table5_profit.py', [], {}),
    ('report', 's', 'rev_appendix_profit.py', [], {}),
    ('report', 's', 'toy_example_v8.py', [], {}),
    ('report', 's', 'make_fig_csvs.py', [], {}),
    ('report', 's', 'fig_v8.py', [], {}),
    ('report', 's', 'fig_ae.py', [], {}),
]


def tag(i, step):
    _, d, script, args, env = step
    t = f"{i:03d}_{'pb_' if d == 'p' else ''}{script[:-3]}" + "".join(f"_{a}" for a in args)
    return t + "".join(f"_{k}{v}" for k, v in env.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="*", choices=["prespecified", "hourly", "battery", "report"])
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args()
    for p in (ROOT / "results" / "profit_basis" / "tables_extra", ROOT / "results" / "figures", LOG):
        p.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "formal" / "out" / "fc").exists() and not a.list:
        sys.exit("formal/out/fc not found: run stage 1 (formal/run_all.sh) first, see README.md")
    todo = [(i, s) for i, s in enumerate(STEPS, 1) if i >= a.start and (not a.group or s[0] in a.group)]
    if a.list:
        for i, s in todo:
            print(f"{i:3d}  {s[0]:<12} {'scripts/profit_basis' if s[1] == 'p' else 'scripts':<20} {s[2]} {' '.join(s[3])} "
                  + " ".join(f"{k}={v}" for k, v in s[4].items()))
        return
    for i, s in todo:
        group, d, script, args, env = s
        e = {**os.environ, "PYTHONIOENCODING": "utf-8", **env}
        if a.jobs:
            e["NJ"] = str(a.jobs)
        t0 = datetime.datetime.now()
        with open(LOG / f"{tag(i, s)}.txt", "w", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, script, *args], cwd=DIRS[d], stdout=fh, stderr=subprocess.STDOUT, env=e)
        mins = (datetime.datetime.now() - t0).total_seconds() / 60
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M} step {i:3d} {tag(i, s)} exit={r.returncode} {mins:.1f} min"
        print(line, flush=True)
        with open(LOG / "_pipeline.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if r.returncode != 0 and not a.keep_going:
            sys.exit(f"step {i} failed; see {LOG / (tag(i, s) + '.txt')}")


if __name__ == "__main__":
    main()
