# 6DOF flexion-loading stiffness relative to intact

Primary: cycle 3. Repeatability check: cycle 2, using the same selected band (no refitting the band).
Each level is normalized to its own Intact condition. Surgical order follows the numbered source folders, not a universal intervention-name order.

## Method

- Use synchronized Motor Mx and sign-corrected Optotrak segment Ry, not motor angle as the response.
- Extract chronological negative-Mx loading from the last zero crossing before the flexion peak; exclude unloading and extension.
- Mirror flexion signs to positive magnitudes and reference angle to loading zero. Fit M = intercept + K * angle with a free intercept.
- Select the widest common cycle-3 band between 40% and 95% of the smallest recorded flexion peak at that level, searching 23 evenly spaced boundaries.
- Require at least 6 raw paired samples and R2 >= 0.95 in every intervention, with a band spanning at least 20% of the common peak. Ties favor the higher-moment band.
- All conditions at a level use identical bounds. No intervention-specific tuning, fixed 1-4 Nm anchors, extrapolation, or fallback cycles.
- Retained [%] = 100 * K / K_intact; cumulative change [%] = retained - 100.
- Stage change [%] = 100 * (K / K_previous - 1); stage change [percentage points] = retained - retained_previous.
- Invalid fits and gaps in the sequence remain missing; never jump across a rejected stage to compute an incremental change.

## Selected windows

| level | window_low_nm | window_high_nm | worst_primary_r2 | minimum_primary_fit_samples | window_reason | best_rejected_worst_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| T12-L1 | 3.417 | 7.637 | 0.952 | 12 | ok | 0.943 |
| L2-L3 | 3.214 | 7.633 | 0.951 | 13 | ok | -- |
| L4-L5 | -- | -- | -- | 0 | no_shared_linear_window | 0.936 |

## T12-L1

| intervention_order | intervention | k_nm_per_deg | r2 | n_fit_samples | stiffness_retained_pct | cumulative_change_pct | stage_change_pct | stage_change_percentage_points | cycle2_to3_stiffness_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Intact | 2.870 | 0.962 | 12 | 100.000 | 0.000 | -- | -- | 2.889 |
| 2 | Partial facetectomy 1 | 3.103 | 0.952 | 13 | 108.124 | 8.124 | 8.124 | 8.124 | -- |
| 3 | Partial facetectomy 2 | 3.100 | 0.975 | 13 | 108.024 | 8.024 | -0.092 | -0.099 | -- |
| 4 | Full uni facetectomy | 3.463 | 0.964 | 13 | 120.663 | 20.663 | 11.700 | 12.638 | 1.378 |
| 5 | Full bilat facetectomy | 3.119 | 0.970 | 13 | 108.690 | 8.690 | -9.922 | -11.973 | 1.421 |
| 6 | Resected Ligamentum Flavum | 2.957 | 0.986 | 14 | 103.036 | 3.036 | -5.202 | -5.654 | -0.094 |
| 7 | Laminectomy | 2.987 | 0.952 | 14 | 104.084 | 4.084 | 1.017 | 1.048 | -2.096 |
| 8 | PLL Release | 2.967 | 0.976 | 13 | 103.376 | 3.376 | -0.681 | -0.708 | 4.071 |

## L2-L3

| intervention_order | intervention | k_nm_per_deg | r2 | n_fit_samples | stiffness_retained_pct | cumulative_change_pct | stage_change_pct | stage_change_percentage_points | cycle2_to3_stiffness_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Intact | 2.206 | 0.962 | 15 | 100.000 | 0.000 | -- | -- | -3.089 |
| 2 | Partial facetectomy 1 | 2.123 | 0.967 | 18 | 96.214 | -3.786 | -3.786 | -3.786 | 3.843 |
| 3 | Partial facetectomy 2 | 2.004 | 0.964 | 20 | 90.834 | -9.166 | -5.592 | -5.380 | 2.798 |
| 4 | Full uni facetectomy | 2.145 | 0.951 | 16 | 97.219 | -2.781 | 7.030 | 6.385 | 0.070 |
| 5 | Full bilat facetectomy | 2.557 | 0.973 | 15 | 115.928 | 15.928 | 19.244 | 18.709 | 0.349 |
| 6 | Ligamentum Flavum Release | 2.927 | 0.973 | 13 | 132.679 | 32.679 | 14.449 | 16.750 | 2.335 |
| 7 | Laminectomy | 2.166 | 0.983 | 15 | 98.174 | -1.826 | -26.006 | -34.505 | -2.488 |
| 8 | PLL Release | 2.196 | 0.967 | 19 | 99.559 | -0.441 | 1.412 | 1.386 | -1.793 |

## L4-L5

| intervention_order | intervention | k_nm_per_deg | r2 | n_fit_samples | stiffness_retained_pct | cumulative_change_pct | stage_change_pct | stage_change_percentage_points | cycle2_to3_stiffness_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Intact | -- | -- | 0 | -- | -- | -- | -- | -- |
| 2 | Partial facetectomy 1 | -- | -- | 0 | -- | -- | -- | -- | -- |
| 3 | Partial facetectomy 2 | -- | -- | 0 | -- | -- | -- | -- | -- |
| 4 | Full uni facetectomy | -- | -- | 0 | -- | -- | -- | -- | -- |
| 5 | Full bilat facetectomy | -- | -- | 0 | -- | -- | -- | -- | -- |
| 6 | PLL Release | -- | -- | 0 | -- | -- | -- | -- | -- |
| 7 | Laminectomy | -- | -- | 0 | -- | -- | -- | -- | -- |

## Quality flags

| level | cycle | intervention | fit_reason | r2 |
| --- | --- | --- | --- | --- |
| T12-L1 | 2 | Partial facetectomy 1 | below_linear_r2_threshold | 0.932 |
| T12-L1 | 2 | Partial facetectomy 2 | below_linear_r2_threshold | 0.949 |
| L4-L5 | 3 | Intact | no_shared_linear_window | -- |
| L4-L5 | 3 | Partial facetectomy 1 | no_shared_linear_window | -- |
| L4-L5 | 3 | Partial facetectomy 2 | no_shared_linear_window | -- |
| L4-L5 | 3 | Full uni facetectomy | no_shared_linear_window | -- |
| L4-L5 | 3 | Full bilat facetectomy | no_shared_linear_window | -- |
| L4-L5 | 3 | PLL Release | no_shared_linear_window | -- |
| L4-L5 | 3 | Laminectomy | no_shared_linear_window | -- |
| L4-L5 | 2 | Intact | no_shared_linear_window | -- |
| L4-L5 | 2 | Partial facetectomy 1 | no_shared_linear_window | -- |
| L4-L5 | 2 | Partial facetectomy 2 | no_shared_linear_window | -- |
| L4-L5 | 2 | Full uni facetectomy | no_shared_linear_window | -- |
| L4-L5 | 2 | Full bilat facetectomy | no_shared_linear_window | -- |
| L4-L5 | 2 | PLL Release | no_shared_linear_window | -- |
| L4-L5 | 2 | Laminectomy | no_shared_linear_window | -- |

## Interpretation

These are high-load flexion-loading regression slopes, not whole-curve stiffness, neutral-zone stiffness, or loading/unloading mean stiffness.
Changes may be positive or negative; do not force monotonic reductions. Compare small stage changes with the cycle-2-to-3 differences before interpreting them.
Sequential stages measure cumulative effects and additional effects conditional on prior surgery. They do not isolate each structure's contribution to intact stiffness.
Cycles are technical repeats, not independent specimens. No population confidence intervals or significance claims are inferred from them.
The analysis retains the existing FE synchronization and cycle definitions; fit quality is not an independent proof of alignment accuracy.
The spreader force-to-pivot-moment conversion is not imposed on the 6DOF loading range.

## Outputs

- [6dof_flexion_stiffness_summary.csv](6dof_flexion_stiffness_summary.csv): primary results and cycle-2 comparisons.
- [6dof_flexion_stiffness_cycles.csv](6dof_flexion_stiffness_cycles.csv): fits, source files, synchronization, extraction, and rejection reasons for both cycles.
- [6dof_flexion_stiffness_windows.csv](6dof_flexion_stiffness_windows.csv): approved bounds and best rejected-window diagnostics.
- [6dof_flexion_stiffness_loading_samples.csv](6dof_flexion_stiffness_loading_samples.csv): paired raw loading samples and selected-band membership.
- [6dof_flexion_stiffness_comparison.png](6dof_flexion_stiffness_comparison.png): retained stiffness and conditional stage changes.
- [6dof_flexion_stiffness_loading_fits.png](6dof_flexion_stiffness_loading_fits.png): raw loading branches and accepted regressions.
