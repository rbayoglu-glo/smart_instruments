#%%
"""Plot cycle-consistent AR data for L2-L3 PLL Release."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import centered_mean_curve
from plot_l4l5_all_interventions_lb_ar_cycle3_centered_mean import fe_cycle_boundaries
from plot_l4l5_intact_ar_all_cycles_centered_mean import estimate_cycle_consistent_lag
from plot_levels_all_interventions_lb_ar_cycles_centered_mean_stacked import (
    AXIS_CONFIG,
    LEVEL_ROOT,
    intervention_folders,
    load_motor,
    load_optotrak,
)


LEVEL = "L2-L3"
INTERVENTION_FOLDER = "8. PLL Release"
OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_STEM = "L2-L3_PLL_release_AR_cycles_1_2_3_centered_mean"
OUT_PNG = OUT_DIR / f"{OUT_STEM}.png"
OUT_CSV = OUT_DIR / f"{OUT_STEM}.csv"
OUT_RANGE_CSV = OUT_DIR / f"{OUT_STEM}_cycle_ranges.csv"
CYCLE_COLORS = ["tab:blue", "tab:orange", "tab:green"]


def main() -> None:
    level_dir = LEVEL_ROOT / LEVEL
    folder = next(path for path in intervention_folders(level_dir) if path.name == INTERVENTION_FOLDER)
    config = AXIS_CONFIG["AR"]
    motor_time, moment, motor_file = load_motor(folder, config["moment_col"])
    opto_time, angle, opto_file = load_optotrak(folder, config["angle_col"])
    boundaries = fe_cycle_boundaries(motor_time, moment)
    if len(boundaries) != 4:
        raise RuntimeError(f"Expected three complete AR motor cycles, found {len(boundaries) - 1}.")

    lag, trough_lags = estimate_cycle_consistent_lag(motor_time, moment, opto_time, angle)
    requested_opto_time = motor_time - lag
    valid = (requested_opto_time >= opto_time[0]) & (requested_opto_time <= opto_time[-1])
    aligned_angle = np.full(motor_time.shape, np.nan)
    aligned_angle[valid] = np.interp(requested_opto_time[valid], opto_time, angle)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    curve_rows = []
    range_rows = []

    for cycle_number, color, (cycle_start, cycle_end) in zip(
        range(1, 4),
        CYCLE_COLORS,
        zip(boundaries[:-1], boundaries[1:]),
    ):
        cycle_mask = (motor_time >= cycle_start) & (motor_time < cycle_end)
        keep = valid & cycle_mask
        cycle_angle = aligned_angle[keep]
        cycle_moment = moment[keep]
        total_cycle_rows = int(cycle_mask.sum())
        if len(cycle_angle) < 20:
            raise RuntimeError(f"AR cycle {cycle_number} has too few synchronized samples.")

        sample_coverage = len(cycle_angle) / total_cycle_rows
        coverage_start = max(cycle_start, opto_time[0] + lag)
        coverage_end = min(cycle_end, opto_time[-1] + lag)
        time_coverage = max(0.0, coverage_end - coverage_start) / (cycle_end - cycle_start)
        has_full_coverage = sample_coverage >= 0.999 and time_coverage >= 0.999

        if has_full_coverage:
            curve_type = "centered_loading_unloading_mean"
            curve_angle, curve_moment, _, _ = centered_mean_curve(cycle_angle, cycle_moment)
            line_style = "-"
            line_width = 2.2
            line_label = f"Cycle {cycle_number} (centered mean)"
        else:
            curve_type = "partial_observed_trace"
            curve_angle = cycle_angle
            curve_moment = cycle_moment
            line_style = "--"
            line_width = 1.6
            line_label = f"Cycle {cycle_number} (partial raw, {sample_coverage:.0%} samples)"

        ax.plot(
            curve_angle,
            curve_moment,
            color=color,
            lw=line_width,
            ls=line_style,
            label=line_label,
        )
        for point_index, (angle_value, moment_value) in enumerate(zip(curve_angle, curve_moment)):
            curve_rows.append({
                "level": LEVEL,
                "intervention": "PLL Release",
                "cycle": cycle_number,
                "curve_type": curve_type,
                "point_index": point_index,
                "angle_deg": angle_value,
                "moment_mz_nm": moment_value,
                "motor_angle_lag_s": lag,
                "alignment_method": "corresponding Mz/Rx troughs",
                "opto_sample_coverage_fraction": sample_coverage,
                "opto_time_coverage_fraction": time_coverage,
                "motor_file": motor_file,
                "optotrak_file": opto_file,
                "n_cycle_rows": len(cycle_angle),
            })
        range_rows.append({
            "level": LEVEL,
            "intervention": "PLL Release",
            "cycle": cycle_number,
            "curve_type": curve_type,
            "cycle_start_s": cycle_start,
            "cycle_end_s": cycle_end,
            "n_motor_rows": total_cycle_rows,
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
    ax.set_xlabel("AR angle change, ΔRx [deg]")
    ax.set_ylabel("Mz [Nm]")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-10, 10)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", title="L2-L3 PLL Release")
    fig.suptitle("L2-L3 PLL Release: Axial Rotation", fontsize=12)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    pd.DataFrame(curve_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(range_rows).to_csv(OUT_RANGE_CSV, index=False)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_RANGE_CSV}")
    print(
        f"motor={motor_file}, optotrak={opto_file}, cycle-consistent lag={lag:+.3f}s, "
        f"trough lags={np.round(trough_lags, 3).tolist()}"
    )
    print(pd.DataFrame(range_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
