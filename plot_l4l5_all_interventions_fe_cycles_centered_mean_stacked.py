#%%
"""Plot centered loading/unloading mean FE curves for cycles 1-3."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_intact_l4l5_fe_raw_and_referenced import detect_full_cycles_on_aligned_angle
from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import centered_mean_curve
from plot_l4l5_all_interventions_fe_cycle3_mean import (
    COLORS,
    FE_FILE_OVERRIDES,
    LEVEL_DIR,
    align_optotrak_to_motor,
    load_motor,
    load_optotrak,
)


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "L4-L5_all_interventions_FE_cycles_1_2_3_centered_mean_stacked.png"
OUT_CSV = OUT_DIR / "L4-L5_all_interventions_FE_cycles_1_2_3_centered_mean_stacked.csv"
OUT_ANGLE_RANGE_CSV = OUT_DIR / "L4-L5_all_interventions_FE_cycles_1_2_3_angle_ranges.csv"


def intervention_folders() -> list[Path]:
    return sorted(
        [path for path in LEVEL_DIR.iterdir() if path.is_dir() and path.name[:1].isdigit()],
        key=lambda path: int(path.name.split(".")[0]),
    )


def main() -> None:
    folders = intervention_folders()
    cases = []

    for folder in folders:
        name = folder.name.split(".", 1)[1].strip()
        config = FE_FILE_OVERRIDES.get(folder.name, {})
        motor_time, motor_angle, moment = load_motor(folder, config)
        opto_time, opto_angle = load_optotrak(folder, config)
        aligned_angle, measured, lag, correlation = align_optotrak_to_motor(
            motor_time, motor_angle, opto_time, opto_angle
        )
        aligned_valid = measured & np.isfinite(aligned_angle)
        _, boundaries = detect_full_cycles_on_aligned_angle(
            motor_time[aligned_valid], aligned_angle[aligned_valid]
        )
        cases.append({
            "name": name,
            "color": COLORS[len(cases)],
            "motor_time": motor_time,
            "moment": moment,
            "aligned_angle": aligned_angle,
            "aligned_valid": aligned_valid,
            "boundaries": boundaries,
            "lag": lag,
            "correlation": correlation,
        })

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.5, 13.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    mean_rows = []
    angle_range_rows = []

    for cycle_index, ax in enumerate(axes):
        cycle_number = cycle_index + 1
        for case in cases:
            cycle_start, cycle_end = case["boundaries"][cycle_index:cycle_index + 2]
            keep = (
                case["aligned_valid"]
                & (case["motor_time"] >= cycle_start)
                & (case["motor_time"] < cycle_end)
            )
            cycle_angle = case["aligned_angle"][keep]
            cycle_moment = case["moment"][keep]
            if len(cycle_angle) < 20:
                raise RuntimeError(
                    f"{case['name']}: too few aligned cycle-{cycle_number} samples "
                    f"({len(cycle_angle)})."
                )

            mean_angle, mean_moment, _, _ = centered_mean_curve(cycle_angle, cycle_moment)
            ax.plot(
                mean_angle,
                mean_moment,
                color=case["color"],
                lw=2.0,
                label=case["name"],
            )
            for phase, (x_value, y_value) in enumerate(zip(mean_angle, mean_moment)):
                mean_rows.append({
                    "cycle": cycle_number,
                    "intervention": case["name"],
                    "phase_index": phase,
                    "angle_mean_deg": x_value,
                    "moment_mx_mean_nm": y_value,
                    "motor_angle_lag_s": case["lag"],
                    "motor_angle_lag_correlation": case["correlation"],
                    "n_cycle_rows": len(cycle_angle),
                })
            angle_range_rows.append({
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
        ax.set_ylabel("Mx [Nm]")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(alpha=0.25)
        ax.text(
            9.0,
            1.0,
            "Extension",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        ax.text(
            -1.5,
            -9.0,
            "Flexion",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        if cycle_index == 0:
            ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Angle [deg]")
    fig.suptitle("L4-L5", fontsize=12)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    pd.DataFrame(mean_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(angle_range_rows).to_csv(OUT_ANGLE_RANGE_CSV, index=False)

    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_ANGLE_RANGE_CSV}")
    for case in cases:
        print(
            f"{case['name']}: lag={case['lag']:+.2f}s, "
            f"r={case['correlation']:.3f}, boundaries="
            f"{np.round(case['boundaries'], 3).tolist()}"
        )


if __name__ == "__main__":
    main()
