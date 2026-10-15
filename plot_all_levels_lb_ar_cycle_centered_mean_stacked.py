#%%
"""Plot cycle-consistent LB and AR centered means for three lumbar levels."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import loading_ramp_curve, loading_window_start
from plot_l4l5_all_interventions_lb_ar_cycle3_centered_mean import (
    AXES as L4_AXES,
    FILE_OVERRIDES as L4_FILE_OVERRIDES,
    LEVEL_DIR as L4_LEVEL_DIR,
    intervention_folders as l4_intervention_folders,
    load_motor as l4_load_motor,
    load_optotrak as l4_load_optotrak,
    numeric_opto_number as l4_numeric_opto_number,
)
from plot_l4l5_intact_ar_all_cycles_centered_mean import estimate_cycle_consistent_lag
from plot_levels_all_interventions_lb_ar_cycles_centered_mean_stacked import (
    AXIS_CONFIG,
    LEVEL_ROOT,
    adjusted_cycle_boundaries,
    intervention_folders,
    load_motor,
    load_optotrak,
)


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
LEVELS = ("T12-L1", "L2-L3", "L4-L5")
AXIS_CYCLES = {"LB": 3, "AR": 3}
INTERVENTION_COLORS = {
    "Intact": "tab:blue",
    "Partial facetectomy 1": "tab:orange",
    "Partial facetectomy 2": "tab:green",
    "Full uni facetectomy": "tab:red",
    "Full bilat facetectomy": "tab:purple",
    "PLL Release": "tab:brown",
    "Laminectomy": "tab:pink",
    "Ligamentum Flavum Release": "tab:cyan",
    "Resected Ligamentum Flavum": "tab:cyan",
}


def load_level_cases(level: str, axis: str, config: dict) -> list[dict]:
    if level == "L4-L5":
        level_dir = L4_LEVEL_DIR
        folders = l4_intervention_folders()
    else:
        level_dir = LEVEL_ROOT / level
        folders = intervention_folders(level_dir)

    cases = []
    for intervention_order, folder in enumerate(folders, start=1):
        name = re.sub(r"^\d+\.\s*", "", folder.name)
        if level == "L4-L5":
            selected_config = dict(config)
            selected_config.update(L4_FILE_OVERRIDES.get(axis, {}).get(folder.name, {}))
            motor_time, moment, motor_file = l4_load_motor(folder, selected_config)
            opto_time, angle, opto_file = l4_load_optotrak(folder, selected_config)
        else:
            motor_time, moment, motor_file = load_motor(folder, config["moment_col"])
            opto_time, angle, opto_file = load_optotrak(folder, config["angle_col"])

        boundaries = adjusted_cycle_boundaries(motor_time, moment)
        lag, trough_lags = estimate_cycle_consistent_lag(motor_time, moment, opto_time, angle)
        requested_opto_time = motor_time - lag
        valid = (requested_opto_time >= opto_time[0]) & (requested_opto_time <= opto_time[-1])
        aligned_angle = np.full(motor_time.shape, np.nan)
        aligned_angle[valid] = np.interp(requested_opto_time[valid], opto_time, angle)
        cases.append({
            "name": name,
            "order": intervention_order,
            "motor_time": motor_time,
            "moment": moment,
            "opto_time": opto_time,
            "aligned_angle": aligned_angle,
            "valid": valid,
            "boundaries": boundaries,
            "lag": lag,
            "trough_lags": trough_lags,
            "motor_file": motor_file,
            "opto_file": opto_file,
        })
    return cases


def plot_axis(
    axis: str,
    cycle_number: int,
    levels: tuple[str, ...] = LEVELS,
    out_stem: str | None = None,
    title: str | None = None,
) -> None:
    if axis == "LB":
        base_config = AXIS_CONFIG[axis]
        output_suffix = f"LB_cycle{cycle_number}"
        default_title = f"Lateral Bending Cycle {cycle_number}: All Levels"
    else:
        base_config = AXIS_CONFIG[axis]
        output_suffix = f"AR_cycle{cycle_number}"
        default_title = f"Axial Rotation Cycle {cycle_number}: All Levels"
    title = title or default_title
    out_stem = out_stem or f"all_levels_{output_suffix}_centered_mean_stacked"
    out_png = OUT_DIR / f"{out_stem}.png"
    out_csv = OUT_DIR / f"{out_stem}.csv"
    out_range_csv = OUT_DIR / f"{out_stem}_angle_ranges.csv"

    fig, axes = plt.subplots(
        len(levels),
        1,
        figsize=(9.5, 16.0 / 3 * len(levels)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    curve_rows = []
    range_rows = []

    for level, ax in zip(levels, axes):
        config = L4_AXES[axis] if level == "L4-L5" else base_config
        cases = load_level_cases(level, axis, config)
        for case in cases:
            if len(case["boundaries"]) <= cycle_number:
                raise RuntimeError(f"{level} {case['name']}: cannot isolate cycle {cycle_number}.")
            cycle_start, cycle_end = case["boundaries"][cycle_number - 1:cycle_number + 1]
            cycle_start = loading_window_start(case["motor_time"], case["moment"], cycle_start)
            cycle_mask = (
                (case["motor_time"] >= cycle_start)
                & (case["motor_time"] < cycle_end)
            )
            keep = case["valid"] & cycle_mask
            cycle_angle = case["aligned_angle"][keep]
            cycle_moment = case["moment"][keep]
            total_rows = int(cycle_mask.sum())
            if len(cycle_angle) < 20:
                raise RuntimeError(f"{level} {case['name']} {axis} cycle {cycle_number}: too few paired samples.")

            sample_coverage = len(cycle_angle) / total_rows
            coverage_start = max(cycle_start, case["opto_time"][0] + case["lag"])
            coverage_end = min(cycle_end, case["opto_time"][-1] + case["lag"])
            time_coverage = max(0.0, coverage_end - coverage_start) / (cycle_end - cycle_start)
            has_full_coverage = sample_coverage >= 0.999 and time_coverage >= 0.999
            if has_full_coverage:
                curve_type = "centered_loading_ramp"
                curve_angle, curve_moment = loading_ramp_curve(cycle_angle, cycle_moment)
                line_style = "-"
                line_width = 2.0
                line_label = f"{case['order']}. {case['name']}"
            else:
                curve_type = "partial_observed_trace"
                curve_angle = cycle_angle
                curve_moment = cycle_moment
                line_style = "--"
                line_width = 1.6
                line_label = f"{case['order']}. {case['name']} (partial {sample_coverage:.0%})"

            ax.plot(
                curve_angle,
                curve_moment,
                color=INTERVENTION_COLORS.get(case["name"], "0.25"),
                lw=line_width,
                ls=line_style,
                label=line_label,
            )
            for phase_index, (angle_value, moment_value) in enumerate(zip(curve_angle, curve_moment)):
                curve_rows.append({
                    "level": level,
                    "axis": axis,
                    "cycle": cycle_number,
                    "intervention_order": case["order"],
                    "intervention": case["name"],
                    "curve_type": curve_type,
                    "phase_index": phase_index,
                    "angle_deg": angle_value,
                    "moment_nm": moment_value,
                    "motor_angle_lag_s": case["lag"],
                    "trough_lags_s": ",".join(f"{value:.4f}" for value in case["trough_lags"]),
                    "opto_sample_coverage_fraction": sample_coverage,
                    "opto_time_coverage_fraction": time_coverage,
                    "motor_file": case["motor_file"],
                    "optotrak_file": case["opto_file"],
                    "n_cycle_rows": len(cycle_angle),
                })
            range_rows.append({
                "level": level,
                "axis": axis,
                "cycle": cycle_number,
                "intervention_order": case["order"],
                "intervention": case["name"],
                "curve_type": curve_type,
                "n_motor_rows": total_rows,
                "n_cycle_rows": len(cycle_angle),
                "opto_sample_coverage_fraction": sample_coverage,
                "opto_time_coverage_fraction": time_coverage,
                "raw_angle_min_deg": float(np.min(cycle_angle)),
                "raw_angle_max_deg": float(np.max(cycle_angle)),
                "display_angle_min_deg": float(np.min(curve_angle)),
                "display_angle_max_deg": float(np.max(curve_angle)),
                "moment_min_nm": float(np.min(cycle_moment)),
                "moment_max_nm": float(np.max(cycle_moment)),
            })

        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.set_title(f"{level} - Cycle {cycle_number}", loc="left", fontsize=10)
        ax.set_ylabel(base_config["moment_label"])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(alpha=0.25)
        ax.text(
            8.0,
            1.0,
            "Left",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        ax.text(
            -2.0,
            -9.0,
            "Right",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        ax.legend(loc="upper left", fontsize=7, ncol=2, title="Intervention order")

    axes[-1].set_xlabel(base_config["angle_label"])
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(out_csv, index=False)
    pd.DataFrame(range_rows).to_csv(out_range_csv, index=False)
    print(f"saved {out_png}")
    print(f"saved {out_csv}")
    print(f"saved {out_range_csv}")
    print(pd.DataFrame(range_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    for axis, cycle_number in AXIS_CYCLES.items():
        plot_axis(axis, cycle_number)


if __name__ == "__main__":
    main()
