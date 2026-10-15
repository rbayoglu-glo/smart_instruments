#%%
"""Plot centered loading/unloading FE means for all intact L4-L5 cycles."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_intact_l4l5_fe_raw_and_referenced import detect_full_cycles_on_aligned_angle
from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import centered_mean_curve
from plot_l4l5_all_interventions_fe_cycle3_mean import (
    LEVEL_DIR,
    align_optotrak_to_motor,
    load_motor,
    load_optotrak,
    numeric_book_number,
    numeric_opto_number,
)


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_STEM = "L4-L5_intact_FE_cycles_1_2_3_centered_mean"
OUT_PNG = OUT_DIR / f"{OUT_STEM}.png"
OUT_CSV = OUT_DIR / f"{OUT_STEM}.csv"
OUT_RANGE_CSV = OUT_DIR / f"{OUT_STEM}_cycle_ranges.csv"
CYCLE_COLORS = ["tab:blue", "tab:orange", "tab:green"]


def main() -> None:
    folder = LEVEL_DIR / "1. Intact"
    motor_files = sorted((folder / "Motor").glob("Book*.xlsx"), key=numeric_book_number)
    optotrak_files = sorted((folder / "Optotrak").glob("*.xls"), key=numeric_opto_number)
    motor_file = motor_files[0].name
    optotrak_file = optotrak_files[0].name

    motor_time, motor_angle, moment = load_motor(folder, {})
    opto_time, opto_angle = load_optotrak(folder, {})
    aligned_angle, measured, lag, correlation = align_optotrak_to_motor(
        motor_time,
        motor_angle,
        opto_time,
        opto_angle,
    )
    aligned_valid = measured & np.isfinite(aligned_angle)
    _, boundaries = detect_full_cycles_on_aligned_angle(
        motor_time[aligned_valid],
        aligned_angle[aligned_valid],
    )
    if len(boundaries) != 4:
        raise RuntimeError(f"Expected three complete FE cycles, found {len(boundaries) - 1}.")

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    mean_rows = []
    range_rows = []

    for cycle_number, color, (cycle_start, cycle_end) in zip(
        range(1, 4),
        CYCLE_COLORS,
        zip(boundaries[:-1], boundaries[1:]),
    ):
        has_full_coverage = (
            cycle_start >= opto_time[0] + lag
            and cycle_end <= opto_time[-1] + lag
        )
        if not has_full_coverage:
            raise RuntimeError(f"FE cycle {cycle_number} is outside synchronized Optotrak coverage.")

        keep = aligned_valid & (motor_time >= cycle_start) & (motor_time < cycle_end)
        cycle_angle = aligned_angle[keep]
        cycle_moment = moment[keep]
        if len(cycle_angle) < 20:
            raise RuntimeError(f"FE cycle {cycle_number} has too few synchronized samples.")

        mean_angle, mean_moment, _, _ = centered_mean_curve(cycle_angle, cycle_moment)
        ax.plot(
            mean_angle,
            mean_moment,
            color=color,
            lw=2.2,
            label=f"Cycle {cycle_number}",
        )
        for phase_index, (angle_value, moment_value) in enumerate(zip(mean_angle, mean_moment)):
            mean_rows.append({
                "cycle": cycle_number,
                "phase_index": phase_index,
                "angle_mean_deg": angle_value,
                "moment_mx_mean_nm": moment_value,
                "motor_angle_lag_s": lag,
                "motor_angle_lag_correlation": correlation,
                "motor_file": motor_file,
                "optotrak_file": optotrak_file,
                "n_cycle_rows": len(cycle_angle),
            })
        range_rows.append({
            "cycle": cycle_number,
            "cycle_start_s": cycle_start,
            "cycle_end_s": cycle_end,
            "n_cycle_rows": len(cycle_angle),
            "raw_angle_min_deg": float(np.min(cycle_angle)),
            "raw_angle_max_deg": float(np.max(cycle_angle)),
            "centered_mean_angle_min_deg": float(np.min(mean_angle)),
            "centered_mean_angle_max_deg": float(np.max(mean_angle)),
            "moment_min_nm": float(np.min(cycle_moment)),
            "moment_max_nm": float(np.max(cycle_moment)),
        })

    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_xlabel("FE angle change, ΔRy [deg]")
    ax.set_ylabel("Mx [Nm]")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", title="Intact L4-L5")
    ax.text(
        8.0,
        1.0,
        "Extension",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )
    ax.text(
        -2.0,
        -9.0,
        "Flexion",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )
    fig.suptitle("L4-L5 Intact Flexion/Extension: All Cycles", fontsize=12)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    pd.DataFrame(mean_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(range_rows).to_csv(OUT_RANGE_CSV, index=False)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_RANGE_CSV}")
    print(f"motor={motor_file}, optotrak={optotrak_file}, lag={lag:+.2f}s, r={correlation:.3f}")
    print(pd.DataFrame(range_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
