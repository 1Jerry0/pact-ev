# Where each result comes from

Paths are relative to the repository root. `s/` = `scripts/`, `p/` = `scripts/profit_basis/`, `R/` = `results/`, `PB/` = `results/profit_basis/`, `F/` = `formal/out/tables/`. Text outputs are in Chinese; numbers can be matched directly. Console-only results are in `results/logs/`.

## Data and setting (Section 3)

| Result | Script | Output |
|---|---|---|
| Table 1, descriptive statistics | `s/desc_stats.py` | `R/descriptive_stats.csv` |
| Table S2, within-day features | `s/desc_stats.py` | `R/within_day_features.csv` |
| Section 3.3, A7 batteries charging simultaneously (p99, maximum), concurrent power, share of energy started in the valley; Section 4.1 forecast biases | `s/check_facts.py` | log (power is printed relative to the annual mean of 334.9 kWh/h; the paper rescales to the development-period mean of 394 kWh/h) |
| Section 2.6 and 3.3, session durations and session power | `s/review_a7_capacity.py` | log |
| Table S1, CRPS; Fig. 3(a) declaration costs | `formal/code/analysis.py` | `F/e1_accuracy.csv`, `F/e2_declaration.csv` |

## Accuracy and operating cost (Section 4.1)

| Result | Script | Output |
|---|---|---|
| Fig. 3 | `s/fig_v8.py accuracy` | `R/figures/fig2_accuracy_value.png` (manuscript Fig. 3; inputs: `F/`, `R/cb_storage_sched.parquet` from `s/exp_cost_basis_st.py`, `R/cb_hourly_plans.parquet` from `s/exp_cost_basis.py`, `R/h1_profit.parquet` from `s/exp_h1_profit.py`) |
| Table 2 | `s/rev_table3_profit.py` | `R/table3_profit.md` |
| Error scaling 0.8 / 0.5 / 0.2 | `s/exp_cost_basis_t4.py`, `s/ana_cost_basis_t4.py` | `R/cb_t4_summary.txt` |
| Battery level: Kendall's τ of MAE and cost, TimesFM-3.0 vs LightGBM | `s/exp_bl_profit.py models`, `s/ana_bl_profit.py` | `R/blp_summary.txt` |
| UrbanEV agreement rate (H1) | `s/exp_h1_profit.py` | `R/h1_profit.txt` |

## Error prices (Section 4.2)

| Result | Script | Output |
|---|---|---|
| Fig. 1, Fig. 4(a); prices 0.29 / 0.04 CNY/kWh, 55% in six hours | `s/rev_dswe_profit.py` | `R/rev_dswe_profit.txt`, `R/rev_prices_profit.pkl` |
| Storage station prices (A1) | `s/rev_dswe_st_profit.py` | `R/rev_dswe_st_profit.txt` |
| Fig. 4(b), battery-level prices | `p/exp_bl_prices.py`, `p/ana_bl_prices2.py` | `PB/exp_bl_prices.txt`, `PB/bl_prices.pkl` |
| Fig. 4(c), permuted errors (hourly; H2) | `s/exp_h2_profit.py` | `R/h2_profit.txt` |
| Fig. 4(c), permuted errors (battery level) | `p/exp_packs_e4.py`, then `s/make_fig_csvs.py` | `PB/exp_packs_e4.txt`, `PB/e4_dispersion.csv` |
| Fig. 5, within-day R² with DSWE (H3) | `s/rev_dswe_profit.py`, `s/rev_dswe_st_profit.py`, `s/ana_h3_profit.py` | `R/rev_dswe_profit.txt`, `R/rev_dswe_st_profit.txt`, `R/h3_profit.txt` |
| Fig. 5(a), fixed cost weights | `s/pilot_naive_weights.py`, `s/ana_naive_weights.py` | `R/naive_weights_profit.txt` |

## Equal-cost plans (Section 4.3)

| Result | Script | Output |
|---|---|---|
| Width of the optimal face; deductibles; Proposition 3 check | `p/exp_prop3.py`, `p/exp_leadtime.py` | `PB/exp_prop3.txt`, `PB/exp_leadtime.txt` |
| Fig. 6(b), Table S3, solver positions | `p/exp_solvers.py` | `PB/exp_solvers.txt`, `PB/exp_solvers.parquet` |
| Table 3 (A7 data-anchored and tight grids, A8; midpoint vs default, interior point, random choice, peak-minimising) | `s/exp_cost_basis_t4.py`, `s/ana_cost_basis_t4.py` | `R/cb_t4_summary.txt` |
| Service level, Fig. 6(d) | `s/exp_cost_basis.py`, `s/ana_cost_basis.py` | `R/cb_hourly_summary.txt`, `R/cb_tradeoff_points.parquet` |
| Table 3 (UrbanEV) | `p/rev_uev.py` | `PB/rev_uev.txt` |
| Other tariffs | `p/exp_tariffs.py` | `PB/exp_tariffs.txt` |
| Buffer share chosen in advance | `p/buffer_alpha.py` | `PB/tables_extra/buffer_alpha_A7.md` |
| Table 4, Fig. 6(c), Table S4 | `p/exp_bl.py`, `p/exp_bl2.py`, `s/rev_table5_profit.py`, `s/rev_appendix_profit.py` | `PB/exp_bl.csv`, `PB/exp_bl2_a.csv`, `R/table5_profit.md`, `R/appendix_profit.md` |
| Plans per model on the face (battery level) | `s/exp_bl_profit.py models`, `s/ana_bl_profit.py` | `R/blp_summary.txt` |
| Holding cost of the buffer | `s/ana_breakeven_profit.py` (uses `PB/bss_map_plans.parquet` from `p/exp_map.py`) | `R/breakeven_profit.txt` |
| Fig. 6(a) and all panels | `s/fig_v8.py plan` | `R/figures/fig5_plan.png` |

## Value of planning versus forecasting (Section 4.4)

| Result | Script | Output |
|---|---|---|
| Hourly decomposition (32% / 33% / 63%) | `s/ana_hourly_decomp_profit.py` | `R/hourly_decomp_profit.txt` |
| Fig. 7(a), Table S5 | `s/exp_bl_profit.py main` and `tariffs`, `s/ana_bl_profit.py`, `s/exp_bl_rule_free_profit.py` | `R/blp_summary.txt`, `R/blp_rule_free.txt` |
| Fig. 7(b): Shenzhen zones; A8 | `p/exp_bl_uev_profile.py`, `s/make_fig7b_csv.py`; `p/exp_bl_a8_profile.py`, `s/make_fig_csvs.py` | `PB/uev_profile_fc.csv`; `PB/exp_bl_a8_profile_fc.csv` |
| Storage stations, profile vs LightGBM | `s/exp_storage_profit.py tariffs` and `ana` | `R/storage_profit.txt` |
| Arrival noise and intraday updates | `p/fc_intraday.py`; `p/exp_bl_mpc.py` | `PB/fc_intraday.txt`; `PB/exp_bl_mpc.txt` |
| Fig. 8 | `p/exp_bl.py`, `p/exp_bl2.py`, `p/exp_bl_carry.py`, `p/ana_split_month.py`, `p/exp_bl_uev.py`, `p/ana_uev_acc.py`; `s/fig_v8.py when` | `PB/exp_bl.csv`, `PB/exp_bl_carry.csv`, `PB/ana_split_month.txt`, `PB/exp_bl_uev.csv`, `PB/ana_uev_acc.txt`; `R/figures/fig7_when.png` |
| Table 5, purchases by tariff period | `p/exp_bl_valley.py` | `PB/exp_bl_valley.txt` |

## Decision-focused forecasts (Section 4.5)

| Result | Script | Output |
|---|---|---|
| Table 6 | `p/exp_dfl.py`, `p/exp_dfl_learn.py`, `p/exp_dfl_grad.py` | `PB/exp_dfl_x3.txt`, `PB/exp_dfl_learn_x3.txt`, `PB/exp_dfl_grad_x3.txt` (the suffix `x3` is a legacy label) |
| Stochastic programming (52% vs 0.6%) | `p/exp_leadtime.py` | `PB/exp_leadtime.txt` (`reg_saa` at L = 0) |

## Robustness (Section 4.6) and implications (Section 4.7)

| Result | Script | Output |
|---|---|---|
| Table 7, pre-specified basis | `formal/code/analysis.py` | `F/summary.md` |
| Table 7, profit basis | `s/exp_h1_profit.py`, `s/exp_h2_profit.py`, `s/ana_h3_profit.py`, `s/ana_cost_basis_t4.py`, `s/ana_cost_basis.py` | `R/h1_profit.txt`, `R/h2_profit.txt`, `R/h3_profit.txt`, `R/cb_t4_summary.txt`, `R/cb_hourly_summary.txt` |
| Hypotheses on sub-samples (Note S3) | `s/ana_h_subsample_profit.py` | `R/h_subsample_profit.txt` |
| Table 8, ×3 and ×10 columns | group `prespecified`: `s/exp_bss.py anchor`, `s/exp_bl*.py`, `s/exp_bl_profile_x10.py`, `s/penalty_reweight.py`; `s/ana_maxmin_bases.py` | `R/penalty_reweight.txt`, `R/exp_bl_profile_x10.txt`, `R/maxmin_bases.txt` |
| Table S6, carry-over | `p/exp_bl_carry.py`, `p/exp_bl_carry2.py`, `p/exp_bl_carry_models.py`, `s/rev_appendix_profit.py` | `PB/exp_bl_carry*.txt`, `R/appendix_profit.md` |
| Table S7, sub-periods | `s/ana_halves_profit.py`, `s/rev_appendix_profit.py` | `R/halves_profit.txt`, `R/appendix_profit.md` |
| Table 9: swaps at connection | `p/exp_conn.py`; `p/exp_bl_more.py conn` | `PB/bss_conn.parquet`; `PB/exp_bl_conn.txt` |
| Table 9: tuning window; stock 8 h and power 4× | `p/ana_bl_window.py`; `p/exp_bl_models.py extra` | `PB/ana_bl_window.txt`; log of `exp_bl_models extra` |
| Table 9: rule scaled by demand ratio | `p/exp_packs_scaled.py` | `PB/exp_packs_scaled.txt` |
| Table 9: whole-battery plans | `p/exp_bl_int.py` | `PB/exp_bl_int.txt` |
| Table 9: storage settled at 15 min | `s/exp_storage_profit.py q15` and `ana` | `R/storage_profit.txt` |
| Table 9: finite-difference step | `s/rev_dswe_profit_delta.py` (REL=0.05, 0.2) | `R/rev_dswe_profit_rel05.txt`, `R/rev_dswe_profit_rel20.txt` |
| Table 9: charging efficiency 0.93 | `s/exp_eta_check.py 0.93` and `0.95` | `R/eta_check_093.txt`, `R/eta_check_095.txt` |
| Annual savings (Section 4.7) | `s/ana_annual_value.py` | `R/annual_value.txt` |
| Appendix A numerical checks | `p/exp_prop3.py`, `p/exp_map.py`, `p/exp_leadtime.py`, `p/exp_bl.py` | `PB/exp_prop3.txt`, `PB/bss_map_plans.parquet`, `PB/exp_leadtime.txt`, `PB/exp_bl.txt` |
| Worked example (four-hour station) | `s/toy_example_v8.py` | log |
| All figures | `s/fig_v8.py` | `R/figures/`: manuscript Fig. 1 = `fig1_example.png`, Fig. 2 = `fig2_framework.png` (schematic, `python fig_v8.py framework`; the manuscript version carries the section numbers of the journal version), Fig. 3 = `fig2_accuracy_value.png`, Fig. 4 = `fig3_prices.png`, Fig. 5 = `fig4_dswe.png`, Fig. 6 = `fig5_plan.png`, Fig. 7 = `fig6_value.png`, Fig. 8 = `fig7_when.png` |
