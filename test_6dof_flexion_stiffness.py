"""Focused regression tests for sequential 6DOF flexion stiffness."""

import unittest

import numpy as np
import pandas as pd

import analyze_6dof_flexion_stiffness as analysis


def linear_branch(slope: float = 2.0, peak: float = 8.0, offset: float = 0.0) -> dict:
    moment = np.linspace(0.0, peak, 81)
    return {"moment_nm": moment, "angle_deg": moment / slope + offset}


def synthetic_case(name: str = "Intact", slope: float = 2.0) -> dict:
    prefix = np.array([0.0, 4.0, 8.0, 4.0, 0.0])
    loading = np.linspace(-0.5, -8.0, 16)
    unloading = np.linspace(-7.5, 0.0, 16)
    moment = np.concatenate((prefix, loading, unloading))
    angle = np.concatenate((
        12.0 + prefix / slope,
        12.0 + loading / slope,
        12.0 - 8.0 / slope + (unloading + 8.0) / (2.0 * slope),
    ))
    period = len(moment)
    time_s = np.arange(period * 3, dtype=float)
    return {
        "name": name,
        "motor_time": time_s,
        "moment": np.tile(moment, 3),
        "aligned_angle": np.tile(angle, 3),
        "aligned_valid": np.ones(len(time_s), dtype=bool),
        "opto_time": np.array([0.0, float(period * 3)]),
        "boundaries": [float(period * cycle) for cycle in range(4)],
        "available_cycles": [1, 2, 3],
        "lag": 0.0,
        "correlation": 1.0,
        "motor_file": "synthetic.xlsx",
        "opto_file": "synthetic.xls",
    }


def normalization_rows(stiffness=(2.0, 1.6, 1.2), valid=(True, True, True), cycle=3) -> pd.DataFrame:
    return pd.DataFrame({
        "level": ["synthetic"] * 3,
        "cycle": [cycle] * 3,
        "intervention_order": [1, 2, 3],
        "intervention": ["Intact", "A", "B"],
        "k_nm_per_deg": stiffness,
        "fit_valid": valid,
        "fit_reason": ["ok" if value else "rejected" for value in valid],
        "r2": [1.0] * 3,
    })


class FlexionStiffnessTests(unittest.TestCase):
    def test_extracts_loading_not_unloading_or_extension(self):
        case = synthetic_case()
        branch = analysis.extract_flexion_loading(case, 3)
        np.testing.assert_allclose(branch["angle_deg"], branch["moment_nm"] / 2.0)
        self.assertEqual(branch["n_loading_samples"], 17)
        self.assertEqual(branch["angle_at_zero_deg"], 12.0)
        fit = analysis.fit_stiffness(branch, 3.0, 7.5)
        self.assertTrue(fit["fit_valid"])
        self.assertAlmostEqual(fit["k_nm_per_deg"], 2.0)

    def test_missing_cycle_does_not_fall_back(self):
        case = synthetic_case()
        case["available_cycles"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "cycle_not_fully_synchronized"):
            analysis.extract_flexion_loading(case, 3)

    def test_missing_motor_pair_is_rejected(self):
        case = synthetic_case()
        case["aligned_valid"][-10] = False
        with self.assertRaisesRegex(ValueError, "incomplete_motor_sample_coverage"):
            analysis.extract_flexion_loading(case, 3)

    def test_missing_time_coverage_is_rejected(self):
        case = synthetic_case()
        case["opto_time"][-1] -= 1.0
        with self.assertRaisesRegex(ValueError, "incomplete_optotrak_time_coverage"):
            analysis.extract_flexion_loading(case, 3)

    def test_free_intercept_regression_is_offset_invariant(self):
        fit = analysis.fit_stiffness(linear_branch(offset=3.0), 3.0, 7.5)
        self.assertTrue(fit["fit_valid"])
        self.assertAlmostEqual(fit["k_nm_per_deg"], 2.0)
        self.assertAlmostEqual(fit["intercept_nm"], -6.0)

    def test_nonpositive_slope_is_not_made_positive(self):
        fit = analysis.fit_stiffness(linear_branch(slope=-2.0), 3.0, 7.5)
        self.assertFalse(fit["fit_valid"])
        self.assertEqual(fit["fit_reason"], "nonpositive_loading_stiffness")
        self.assertLess(fit["k_nm_per_deg"], 0.0)

    def test_minimum_count_uses_raw_samples(self):
        branch = {"moment_nm": np.array([3.0, 5.0, 7.0]), "angle_deg": np.array([1.5, 2.5, 3.5])}
        fit = analysis.fit_stiffness(branch, 3.0, 7.0)
        self.assertEqual(fit["n_fit_samples"], 3)
        self.assertFalse(fit["fit_valid"])

    def test_window_cannot_exceed_loading_support(self):
        fit = analysis.fit_stiffness(linear_branch(peak=6.0), 3.0, 7.5)
        self.assertFalse(fit["fit_valid"])
        self.assertEqual(fit["fit_reason"], "fit_window_outside_loading_support")

    def test_shared_window_uses_common_peak_and_is_order_invariant(self):
        branches = [linear_branch(peak=8.0), linear_branch(slope=1.5, peak=6.0)]
        window = analysis.select_common_window(branches)
        reversed_window = analysis.select_common_window(list(reversed(branches)))
        self.assertTrue(window["window_valid"])
        self.assertAlmostEqual(window["window_low_nm"], 2.4)
        self.assertAlmostEqual(window["window_high_nm"], 5.7)
        self.assertEqual(window["window_low_nm"], reversed_window["window_low_nm"])
        self.assertEqual(window["window_high_nm"], reversed_window["window_high_nm"])

    def test_no_shared_linear_window_remains_rejected(self):
        moment = np.linspace(0.0, 8.0, 801)
        nonlinear = {"moment_nm": moment, "angle_deg": np.sin(30.0 * moment)}
        window = analysis.select_common_window([linear_branch(), nonlinear])
        self.assertFalse(window["window_valid"])
        self.assertEqual(window["window_reason"], "no_shared_linear_window")
        self.assertTrue(np.isnan(window["window_low_nm"]))

    def test_cumulative_and_incremental_denominators(self):
        data = analysis.normalize_stiffness(normalization_rows())
        np.testing.assert_allclose(data["stiffness_retained_pct"], [100, 80, 60])
        np.testing.assert_allclose(data["cumulative_change_pct"], [0, -20, -40])
        np.testing.assert_allclose(data["stage_change_pct"].iloc[1:], [-20, -25])
        np.testing.assert_allclose(data["stage_change_percentage_points"].iloc[1:], [-20, -20])
        self.assertTrue(np.isnan(data["stage_change_pct"].iloc[0]))

    def test_rejected_preceding_stage_is_not_skipped(self):
        data = analysis.normalize_stiffness(normalization_rows(valid=(True, False, True)))
        self.assertEqual(data["previous_intervention"].iloc[2], "A")
        self.assertTrue(np.isnan(data["stage_change_pct"].iloc[2]))
        self.assertTrue(np.isnan(data["stage_change_percentage_points"].iloc[2]))
        self.assertAlmostEqual(data["stiffness_retained_pct"].iloc[2], 60.0)

    def test_invalid_intact_prevents_normalization(self):
        data = analysis.normalize_stiffness(normalization_rows(valid=(False, True, True)))
        self.assertTrue(data["stiffness_retained_pct"].isna().all())
        self.assertAlmostEqual(data["stage_change_pct"].iloc[2], -25.0)

    def test_increases_are_not_clipped(self):
        data = analysis.normalize_stiffness(normalization_rows(stiffness=(2.0, 2.2, 1.8)))
        self.assertAlmostEqual(data["stiffness_retained_pct"].iloc[1], 110.0)
        self.assertAlmostEqual(data["stage_change_pct"].iloc[1], 10.0)

    def test_repeatability_normalizes_each_cycle_to_its_own_intact(self):
        data = analysis.normalize_stiffness(pd.concat([
            normalization_rows(stiffness=(2.0, 1.6, 1.2), cycle=3),
            normalization_rows(stiffness=(4.0, 2.0, 1.5), cycle=2),
        ], ignore_index=True))
        summary = analysis.primary_summary(data)
        self.assertAlmostEqual(summary["cycle2_stiffness_retained_pct"].iloc[1], 50.0)
        self.assertAlmostEqual(summary["cycle2_to3_retained_difference_pp"].iloc[1], 30.0)
        self.assertAlmostEqual(summary["cycle2_to3_stiffness_change_pct"].iloc[1], -20.0)

    def test_level_analysis_keeps_actual_stage_order(self):
        cases = [synthetic_case("Intact", 2.0), synthetic_case("PLL Release", 1.8), synthetic_case("Laminectomy", 1.4)]
        fits, samples, window = analysis.analyze_level("synthetic", cases)
        summary = analysis.primary_summary(fits)
        self.assertTrue(window["window_valid"])
        self.assertEqual(summary["intervention"].tolist(), ["Intact", "PLL Release", "Laminectomy"])
        self.assertEqual(len(samples), 17 * 3 * 2)
        self.assertAlmostEqual(summary["stage_change_pct"].iloc[2], 100.0 * (1.4 / 1.8 - 1.0))

    def test_missing_primary_branch_disqualifies_shared_comparison(self):
        cases = [synthetic_case(), synthetic_case("A", 1.5)]
        cases[1]["available_cycles"] = [1, 2]
        fits, _, window = analysis.analyze_level("synthetic", cases)
        self.assertFalse(window["window_valid"])
        self.assertEqual(window["window_reason"], "incomplete_primary_cycle")
        self.assertEqual(len(fits), 4)
        self.assertTrue(fits["stiffness_retained_pct"].isna().all())


if __name__ == "__main__":
    unittest.main()