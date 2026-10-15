# Verification Notes: FE/LB/AR Level-Group 3x3 Charts and Cycle-3 Stacked Charts

Working directory: `C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis`
Outputs directory: `plot_outputs\`
Python: `C:\Users\rbayoglu\AppData\Local\anaconda3\envs\osim\python.exe`

## 1. Scope of the Request

1. FE fallback: when FE cycle 3 is not available, use the last good cycle (highest complete cycle <= 3).
2. Two level-group 3x3 charts (2 rows of levels x 3 columns of FE, LB, AR):
   - Group A: T2-T3 and T4-T5 in one chart.
   - Group B: T6-T7 and T8-T9 in another chart.
3. Source manifest (earlier phase): T8-T9, T6-T7, T4-T5, T2-T3 added to the level manifest. The requested `.xls` file does not exist; the generator writes CSV and MD (`plot_outputs\all_levels_interventions_source_files.{csv,md}`).

## 2. Files Changed

| File | Change |
|---|---|
| `plot_all_levels_fe_cycle3_centered_mean_stacked.py` | Added last-good FE cycle fallback. Refactored into `plot_fe_cycle3(levels, out_stem, title)`. Legend appends `(cycle N)` when N != 3. CSV `cycle` column records the cycle used. |
| `plot_all_levels_lb_ar_cycle_centered_mean_stacked.py` | `plot_axis(axis, cycle_number, levels=LEVELS, out_stem=None, title=None)`. Defaults preserve the original outputs. Figure height scales with level count. |
| `plot_all_levels_fe_lb_ar_3x3_centered_mean.py` | Constants `LEVEL_GROUPS = (("T2-T3", "T4-T5"), ("T6-T7", "T8-T9"))` and `DEFAULT_STEMS`. Functions `load_axis_data(axis, config, stem)`, `plot_3x3(levels, stems, out_stem, title)`, `generate_level_group_inputs(levels)`. `main()` writes the default chart plus the two group charts. |
| `create_level_intervention_file_table.py` | `LEVELS` extended to 8 levels (T2-T3, T4-T5, T6-T7, T8-T9 added). |

No shared loader was modified in this phase except the earlier skip rule in `plot_levels_all_interventions_lb_ar_cycles_centered_mean_stacked.py` and the FE loader (`load_cases`), which skip intervention folders without `Motor\Book*.xlsx`.

## 3. Fallback Logic (FE)

```python
usable_cycles = [cycle for cycle in case["available_cycles"] if cycle <= 3]
cycle_number = max(usable_cycles)
cycle_start, cycle_end = case["boundaries"][cycle_number - 1:cycle_number + 1]
```

A cycle is "available" only when Optotrak coverage is complete (`available_cycles`).

Result: only `T2-T3 / 2. Partial facetectomy 2` falls back, to cycle 2 (its complete FE cycles are 1 and 2). All other interventions use cycle 3.

## 4. Curve Construction

### 4.1 FE and AR/LB full-coverage curves

Source: `plot_intact_l4l5_fe_loading_unloading_mean_by_cycle.py`

- `loading_window_start`: moves the cycle start back to the last zero-moment crossing before the cycle start.
- `loading_ramp_curve(angle, moment)`:
  1. Finds the moment minimum (flexion peak) and maximum (extension peak) within the cycle.
  2. For each peak, `loading_ramp_from_zero` keeps only the leg from the last zero-moment crossing before that peak to the peak. Unloading after each peak is discarded.
  3. Each leg is resampled to a 500-point moment grid (`MEAN_POINTS = 500`), smoothed with an 11-point moving average, and re-zeroed at 0 Nm.
  4. Flexion and extension legs are concatenated end to end. They are not averaged with each other or with any unloading data.

Note: `centered_mean_curve` in the same module averages both branches (loading and unloading). It is used by the L4-L5 intact plot's `main()`, not by the stacked FE, LB, or AR charts in this phase.

### 4.2 AR/LB partial-coverage curves

When coverage is incomplete (`sample_coverage < 0.999` or `time_coverage < 0.999`), the script plots the raw observed trace (`partial_observed_trace`, dashed line, legend shows `partial N%`) instead of the loading ramp.

## 5. Verification Facts Already Checked

- Group CSV cycle check (FE, `T2-T3_T4-T5_FE_cycle3_centered_mean_stacked.csv` and `T6-T7_T8-T9_FE_cycle3_centered_mean_stacked.csv`): `Partial facetectomy 2` in T2-T3 has cycle 2. All other rows have cycle 3.
- AR, `T6-T7_T8-T9_AR_cycle3_centered_mean_stacked.csv`: all 15 curves are `centered_loading_ramp` with `opto_sample_coverage_fraction = 1.0`. `opto_time_coverage_fraction` was not checked in this pass. Rows per curve: 999.
  - T6-T7 (8): Intact, Partial facetectomy 1, Partial facetectomy 2, Full uni facetectomy, Full bilat facetectomy, Resected Ligamentum Flavum, Laminectomy, PLL Release.
  - T8-T9 (7): Intact, Partial facetectomy 1, Partial facetectomy 2, Full uni facetectomy, Full bilat facetectomy, Resected Ligamentum Flavum, Laminectomy. PLL Release is excluded (NOT PERFORMED, no motor or Optotrak data).
- Group 3x3 charts render with the expected row and column labels (visual check done).
- The 3x3 script and the FE script have no lint errors. Pylance reports unresolved-import diagnostics in the LB/AR script (`plot_intact_l4l5_fe_loading_unloading_mean_by_cycle`, `plot_l4l5_all_interventions_lb_ar_cycle3_centered_mean`, `plot_l4l5_intact_ar_all_cycles_centered_mean`, `plot_levels_all_interventions_lb_ar_cycles_centered_mean_stacked`). Scripts run successfully under the osim interpreter, so these look like editor environment resolution, not runtime failures.

## 6. Outputs

Group charts (2 rows x 3 columns):

- `plot_outputs\all_levels_FE_LB_AR_centered_mean_3x3_T2-T3_T4-T5.png`
- `plot_outputs\all_levels_FE_LB_AR_centered_mean_3x3_T6-T7_T8-T9.png`

Intermediate per-axis stacked outputs for each group (inputs to the 3x3 charts), each with `.png`, `.csv`, and `_angle_ranges.csv`:

- `plot_outputs\T2-T3_T4-T5_FE_cycle3_centered_mean_stacked`
- `plot_outputs\T2-T3_T4-T5_LB_cycle3_centered_mean_stacked`
- `plot_outputs\T2-T3_T4-T5_AR_cycle3_centered_mean_stacked`
- `plot_outputs\T6-T7_T8-T9_FE_cycle3_centered_mean_stacked`
- `plot_outputs\T6-T7_T8-T9_LB_cycle3_centered_mean_stacked`
- `plot_outputs\T6-T7_T8-T9_AR_cycle3_centered_mean_stacked`

Default charts (levels T12-L1, L2-L3, L4-L5) regenerate with the same content as before (all cycle 3).

## 7. Known Issues and Checks for the Verifier

1. Transient `OSError: [Errno 22] Invalid argument` on first PNG save of the 3x3 script. A rerun succeeded with no code change. If it recurs, check whether the target PNG is open in another program.
2. Shared loader skip rule: folders without `Motor\Book*.xlsx` are skipped by data availability, not by intervention name. Verify the T8-T9 PLL Release exclusion comes from this rule.
3. Sign convention: FE x-axis is labeled ΔRy [deg]. Mx labels state extension is positive and flexion is negative. Verify against the source Optotrak sign convention.
4. The `.xls` named in the original request does not exist. The manifest is CSV and MD.
5. Interpretive check: FE and AR/LB full-coverage curves use only loading legs (zero to peak). Confirm this matches the intended "loading only, no averaging with unloading" method.
6. Pylance import diagnostics in the LB/AR script are environment-related; confirm the interpreter selection in VS Code if they matter.
