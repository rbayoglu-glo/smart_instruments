#%%
"""Plot stacked centered LB and AR mean curves for complete cycles."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import centered_mean_curve
from plot_l4l5_all_interventions_lb_ar_cycle3_centered_mean import (
    AXES,
    COLORS,
    FILE_OVERRIDES,
    LEVEL_DIR,
    align_angle_to_motor,
    fe_cycle_boundaries,
    load_motor,
    load_optotrak,
)


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
AXIS_CYCLES = {
    "LB": (1, 2, 3),
    "AR": (2, 3),
}


def intervention_folders() -> list[Path]:
    return sorted(
        [path for path in LEVEL_DIR.iterdir() if path.is_dir() and path.name[:1].isdigit()],
        key=lambda path: int(path.name.split(".")[0]),
    )


def load_cases(axis_name: str) -> list[dict]:
    cases = []
    config = AXES[axis_name]
    for color, folder in zip(COLORS, intervention_folders()):
        name = re.sub(r"^\d+\.\s*", "", folder.name)
        selected_config = dict(config)
        selected_config.update(FILE_OVERRIDES.get(axis_name, {}).get(folder.name, {}))
        motor_time, moment, motor_file = load_motor(folder, selected_config)
        opto_time, angle, opto_file = load_optotrak(folder, selected_config)
        boundaries = fe_cycle_boundaries(motor_time, moment)
        aligned_angle, aligned_valid, lag, correlation = align_angle_to_motor(
            motor_time,
            moment,
            opto_time,
            angle,
            boundaries[1] - boundaries[0],
        )
        cases.append({
            "name": name,
            "color": color,
            "motor_time": motor_time,
            "moment": moment,
            "opto_time": opto_time,
            "aligned_angle": aligned_angle,
            "aligned_valid": aligned_valid,
            "boundaries": boundaries,
            "lag": lag,
            "correlation": correlation,
            "motor_file": motor_file,
            "opto_file": opto_file,
        })
    return cases


def plot_axis(axis_name: str, cycle_numbers: tuple[int, ...]) -> None:
    config = AXES[axis_name]
    cycle_tag = "_".join(str(cycle) for cycle in cycle_numbers)
    out_stem = f"L4-L5_all_interventions_{axis_name}_cycles_{cycle_tag}_centered_mean_stacked"
    out_png = OUT_DIR / f"{out_stem}.png"
    out_csv = OUT_DIR / f"{out_stem}.csv"
    out_range_csv = OUT_DIR / f"{out_stem}_angle_ranges.csv"
    cases = load_cases(axis_name)

    fig, axes = plt.subplots(
        len(cycle_numbers),
        1,
        figsize=(8.5, 4.5 * len(cycle_numbers)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    mean_rows = []
    range_rows = []

    for cycle_index, (cycle_number, ax) in enumerate(zip(cycle_numbers, axes)):
        for case in cases:
            if len(case["boundaries"]) <= cycle_number:
                raise RuntimeError(
                    f"{case['name']} {axis_name}: cannot isolate cycle {cycle_number}."
                )
            cycle_start, cycle_end = case["boundaries"][cycle_number - 1:cycle_number + 1]
            opto_available_start = case["opto_time"][0] + case["lag"]
            opto_available_end = case["opto_time"][-1] + case["lag"]
            if cycle_start < opto_available_start or cycle_end > opto_available_end:
                raise RuntimeError(
                    f"{case['name']} {axis_name} cycle {cycle_number}: "
                    "Optotrak coverage does not span the complete Motor interval."
                )
            keep = (
                case["aligned_valid"]
                & (case["motor_time"] >= cycle_start)
                & (case["motor_time"] < cycle_end)
            )
            cycle_angle = case["aligned_angle"][keep]
            cycle_moment = case["moment"][keep]
            if len(cycle_angle) < 20:
                raise RuntimeError(
                    f"{case['name']} {axis_name}: too few cycle-{cycle_number} samples."
                )

            mean_angle, mean_moment, _, _ = centered_mean_curve(cycle_angle, cycle_moment)
            ax.plot(
                mean_angle,
                mean_moment,
                color=case["color"],
                lw=2.0,
                label=case["name"],
            )
            for phase, (angle_value, moment_value) in enumerate(zip(mean_angle, mean_moment)):
                mean_rows.append({
                    "cycle": cycle_number,
                    "intervention": case["name"],
                    "phase_index": phase,
                    "angle_mean_deg": angle_value,
                    "moment_mean_nm": moment_value,
                    "motor_angle_lag_s": case["lag"],
                    "motor_angle_lag_correlation": case["correlation"],
                    "n_cycle_rows": len(cycle_angle),
                })
            range_rows.append({
                "cycle": cycle_number,
                "intervention": case["name"],
                "cycle_extracted_angle_min_deg": float(np.min(cycle_angle)),
                "cycle_extracted_angle_max_deg": float(np.max(cycle_angle)),
                "centered_mean_curve_angle_min_deg": float(np.min(mean_angle)),
                "centered_mean_curve_angle_max_deg": float(np.max(mean_angle)),
            })

        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.set_title(f"Cycle {cycle_number}", loc="left", fontsize=10)
        ax.set_ylabel(config["moment_label"])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(alpha=0.25)
        ax.text(
            8.0,
            1.0,
            "Left",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        ax.text(
            -2.0,
            -9.0,
            "Right",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        if cycle_index == 0:
            ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel(config["angle_label"])
    title = config["title"]
    if axis_name == "AR":
        title += " (complete synchronized cycles)"
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    pd.DataFrame(mean_rows).to_csv(out_csv, index=False)
    pd.DataFrame(range_rows).to_csv(out_range_csv, index=False)
    print(f"saved {out_png}")
    print(f"saved {out_csv}")
    print(f"saved {out_range_csv}")


def main() -> None:
    for axis_name, cycle_numbers in AXIS_CYCLES.items():
        plot_axis(axis_name, cycle_numbers)


if __name__ == "__main__":
    main()
