#%%
"""Plot cycle-3 FE centered means for all interventions and levels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import loading_ramp_curve, loading_window_start
from plot_levels_all_interventions_fe_cycles_centered_mean_stacked import LEVEL_ROOT, load_cases


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_STEM = "all_levels_FE_cycle3_centered_mean_stacked"
LEVELS = ("T12-L1", "L2-L3", "L4-L5")
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


def plot_fe_cycle3(levels: tuple[str, ...], out_stem: str, title: str) -> None:
    out_png = OUT_DIR / f"{out_stem}.png"
    out_csv = OUT_DIR / f"{out_stem}.csv"
    out_range_csv = OUT_DIR / f"{out_stem}_angle_ranges.csv"
    fig, axes = plt.subplots(
        len(levels),
        1,
        figsize=(9.5, 17.0 / 3 * len(levels)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    curve_rows = []
    range_rows = []

    for level, ax in zip(levels, axes):
        cases = load_cases(level)
        for intervention_order, case in enumerate(cases, start=1):
            usable_cycles = [cycle for cycle in case["available_cycles"] if cycle <= 3]
            if not usable_cycles:
                raise RuntimeError(f"{level} {case['name']}: no fully synchronized FE cycle up to 3.")
            cycle_number = max(usable_cycles)
            cycle_start, cycle_end = case["boundaries"][cycle_number - 1:cycle_number + 1]
            cycle_start = loading_window_start(case["motor_time"], case["moment"], cycle_start)
            keep = (
                case["aligned_valid"]
                & (case["motor_time"] >= cycle_start)
                & (case["motor_time"] < cycle_end)
            )
            cycle_angle = case["aligned_angle"][keep]
            cycle_moment = case["moment"][keep]
            if len(cycle_angle) < 20:
                raise RuntimeError(f"{level} {case['name']}: too few FE cycle-3 samples.")

            mean_angle, mean_moment = loading_ramp_curve(cycle_angle, cycle_moment)
            label = f"{intervention_order}. {case['name']}"
            if cycle_number != 3:
                label += f" (cycle {cycle_number})"
            color = INTERVENTION_COLORS.get(case["name"], "0.25")
            ax.plot(
                mean_angle,
                mean_moment,
                color=color,
                lw=2.0,
                label=label,
            )
            for phase_index, (angle_value, moment_value) in enumerate(zip(mean_angle, mean_moment)):
                curve_rows.append({
                    "level": level,
                    "intervention_order": intervention_order,
                    "intervention": case["name"],
                    "cycle": cycle_number,
                    "phase_index": phase_index,
                    "angle_mean_deg": angle_value,
                    "moment_mx_mean_nm": moment_value,
                    "motor_angle_lag_s": case["lag"],
                    "motor_angle_lag_correlation": case["correlation"],
                    "motor_file": case["motor_file"],
                    "optotrak_file": case["opto_file"],
                    "n_cycle_rows": len(cycle_angle),
                })
            range_rows.append({
                "level": level,
                "intervention_order": intervention_order,
                "intervention": case["name"],
                "cycle": cycle_number,
                "n_cycle_rows": len(cycle_angle),
                "cycle_angle_min_deg": float(np.min(cycle_angle)),
                "cycle_angle_max_deg": float(np.max(cycle_angle)),
                "centered_mean_angle_min_deg": float(np.min(mean_angle)),
                "centered_mean_angle_max_deg": float(np.max(mean_angle)),
                "moment_min_nm": float(np.min(cycle_moment)),
                "moment_max_nm": float(np.max(cycle_moment)),
            })

        ax.axvline(0.0, color="0.65", lw=0.8)
        ax.axhline(0.0, color="0.65", lw=0.8)
        ax.set_title(level, loc="left", fontsize=10)
        ax.set_ylabel("Mx [Nm]")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(alpha=0.25)
        ax.text(
            8.0,
            1.0,
            "Extension",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        ax.text(
            -1.5,
            -9.0,
            "Flexion",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
        )
        ax.legend(loc="upper left", fontsize=7, ncol=2, title="Intervention order")

    axes[-1].set_xlabel("FE angle change, ΔRy [deg]")
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
    plot_fe_cycle3(LEVELS, OUT_STEM, "Flexion/Extension Cycle 3: All Levels")


if __name__ == "__main__":
    main()
