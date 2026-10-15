#%%
"""Plot raw intact L4-L5 FE cycles with colored start/end markers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from export_intact_l4l5_fe_raw_and_referenced import detect_full_cycles_on_aligned_angle
from plot_l4l5_all_interventions_fe_cycle3_mean import (
    LEVEL_DIR,
    align_optotrak_to_motor,
    load_motor,
    load_optotrak,
)


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_STEM = "L4-L5_intact_FE"
OUT_CSV = OUT_DIR / f"{OUT_STEM}.csv"
OUT_RANGE_CSV = OUT_DIR / f"{OUT_STEM}_cycle_ranges.csv"
CYCLE_COLORS = ["tab:blue", "tab:orange", "tab:green"]


def main() -> None:
    folder = LEVEL_DIR / "1. Intact"
    motor_time, motor_angle, moment = load_motor(folder, {})
    opto_time, opto_angle = load_optotrak(folder, {})
    aligned_angle, measured, lag, correlation = align_optotrak_to_motor(
        motor_time,
        motor_angle,
        opto_time,
        opto_angle,
    )
    valid = measured & np.isfinite(aligned_angle) & np.isfinite(moment)
    _, boundaries = detect_full_cycles_on_aligned_angle(motor_time[valid], aligned_angle[valid])
    if len(boundaries) != 4:
        raise RuntimeError(f"Expected three complete FE cycles, found {len(boundaries) - 1}.")

    data_rows = []
    range_rows = []
    output_pngs = []

    for cycle_number, color, (cycle_start, cycle_end) in zip(
        range(1, 4),
        CYCLE_COLORS,
        zip(boundaries[:-1], boundaries[1:]),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
        cycle_mask = (motor_time >= cycle_start) & (motor_time < cycle_end)
        keep = valid & cycle_mask
        indices = np.flatnonzero(keep)
        total_rows = int(cycle_mask.sum())
        if indices.size < 20:
            raise RuntimeError(f"FE cycle {cycle_number} has too few aligned samples.")

        cycle_angle = aligned_angle[indices]
        cycle_moment = moment[indices]
        coverage = len(indices) / total_rows
        ax.plot(
            cycle_angle,
            cycle_moment,
            color=color,
            lw=1.5,
            label=f"Cycle {cycle_number}",
        )
        ax.scatter(
            cycle_angle[0],
            cycle_moment[0],
            marker="o",
            s=54,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            zorder=4,
        )
        ax.scatter(
            cycle_angle[-1],
            cycle_moment[-1],
            marker="X",
            s=54,
            color=color,
            linewidths=1.2,
            zorder=4,
        )

        for point_index, sample_index in enumerate(indices):
            data_rows.append({
                "cycle": cycle_number,
                "point_index": point_index,
                "motor_sample_index": int(sample_index),
                "motor_time_s": motor_time[sample_index],
                "requested_optotrak_time_s": motor_time[sample_index] - lag,
                "fe_angle_deg_uncentered": aligned_angle[sample_index],
                "moment_mx_nm_uncentered": moment[sample_index],
                "is_cycle_start": point_index == 0,
                "is_cycle_end": point_index == len(indices) - 1,
                "motor_angle_lag_s": lag,
                "motor_angle_lag_correlation": correlation,
                "sample_coverage_fraction": coverage,
            })
        range_rows.append({
            "cycle": cycle_number,
            "cycle_start_boundary_s": cycle_start,
            "cycle_end_boundary_s": cycle_end,
            "start_sample_time_s": motor_time[indices[0]],
            "end_sample_time_s": motor_time[indices[-1]],
            "sample_count": len(indices),
            "motor_sample_count_in_cycle": total_rows,
            "sample_coverage_fraction": coverage,
            "start_angle_deg_uncentered": float(cycle_angle[0]),
            "start_mx_nm_uncentered": float(cycle_moment[0]),
            "end_angle_deg_uncentered": float(cycle_angle[-1]),
            "end_mx_nm_uncentered": float(cycle_moment[-1]),
            "angle_min_deg_uncentered": float(np.min(cycle_angle)),
            "angle_max_deg_uncentered": float(np.max(cycle_angle)),
            "mx_min_nm_uncentered": float(np.min(cycle_moment)),
            "mx_max_nm_uncentered": float(np.max(cycle_moment)),
        })

        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.set_xlabel("FE angle change, ΔRy [deg]")
        ax.set_ylabel("Mx [Nm]")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(alpha=0.25)
        marker_legend = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black", label="Start"),
            Line2D([], [], marker="X", linestyle="none", color="black", label="End"),
        ]
        line_handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=line_handles + marker_legend,
            loc="best",
            title="Cycle markers",
            fontsize=8,
        )
        fig.suptitle(
            f"L4-L5 Intact Flexion/Extension: Cycle {cycle_number} (raw, uncentered)",
            fontsize=12,
        )
        out_png = OUT_DIR / f"{OUT_STEM}_cycle{cycle_number}_raw_start_end.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        output_pngs.append(out_png)

    pd.DataFrame(data_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(range_rows).to_csv(OUT_RANGE_CSV, index=False)
    for out_png in output_pngs:
        print(f"saved {out_png}")
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_RANGE_CSV}")
    print(f"Motor/Optotrak lag={lag:+.3f}s, r={correlation:.5f}")
    print(pd.DataFrame(range_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
