#%%
"""Compare Motor Angle and Optotrak FE angle across complete FE cycles."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
INPUT_XLSX = OUT_DIR / "L4-L5_intact_FE_raw_and_referenced.xlsx"
OUT_PNG = OUT_DIR / "L4-L5_intact_FE_motor_vs_optotrak_angle_all_full_cycles.png"
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}


def cycle_boundaries(moment: np.ndarray, time_s: np.ndarray) -> list[float]:
    threshold = 0.10 * float(np.nanmax(np.abs(moment)))
    boundaries = [float(time_s[0])]
    previous_start = 0
    for index in range(moment.size - 1):
        if not (moment[index] <= 0.0 < moment[index + 1]):
            continue
        trough = float(np.nanmin(moment[previous_start:index + 2]))
        if trough > -threshold:
            continue
        fraction = -moment[index] / (moment[index + 1] - moment[index])
        boundaries.append(float(time_s[index] + fraction * (time_s[index + 1] - time_s[index])))
        previous_start = index + 1
    boundaries.append(float(time_s[-1]) + 1e-12)
    return boundaries


def nearest_angle(angle_time: np.ndarray, angle: np.ndarray, motor_time: np.ndarray) -> np.ndarray:
    right = np.searchsorted(angle_time, motor_time, side="left").clip(1, len(angle_time) - 1)
    left = right - 1
    use_left = np.abs(motor_time - angle_time[left]) <= np.abs(angle_time[right] - motor_time)
    nearest = np.where(use_left, left, right)
    error = np.abs(angle_time[nearest] - motor_time)
    measured = error <= 0.005
    output = np.full(motor_time.shape, np.nan)
    output[measured] = angle[nearest[measured]]
    return output


def main() -> None:
    aligned = pd.read_excel(INPUT_XLSX, sheet_name="angle_aligned_to_motor")
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 13.0), sharey=True, constrained_layout=True)
    panels = [
        (0, "motor_angle_deg", "Motor Angle [deg]", "Mx vs Motor Angle"),
        (1, "fe_angle_sign_corrected_deg", "Optotrak sign-corrected FE angle [deg]", "Mx vs Optotrak FE angle"),
    ]
    plotted_counts = {}
    for cycle in [1, 2, 3]:
        data = aligned[(aligned["full_opto_cycle"] == cycle) & aligned["optotrack_angle_measured"]].copy()
        if data.empty:
            raise RuntimeError(f"No aligned rows found for complete cycle {cycle}.")
        moment = data["moment_mx_nm"].to_numpy(float)
        color = COLORS[cycle]
        for column_index, x_column, xlabel, title in panels:
            ax = axes[cycle - 1, column_index]
            x_values = data[x_column].to_numpy(float)
            keep = np.isfinite(x_values) & np.isfinite(moment)
            x = x_values[keep]
            y = moment[keep]
            plotted_counts.setdefault(cycle, []).append(int(keep.sum()))
            ax.plot(x, y, color=color, lw=1.25, label=f"full cycle {cycle}")
            if x.size:
                ax.scatter(x[0], y[0], color=color, marker="o", s=30, zorder=3, label="start")
                ax.scatter(x[-1], y[-1], color=color, marker="x", s=38, linewidth=1.1, zorder=3, label="end")
            ax.axvline(0.0, color="0.65", lw=0.8)
            ax.axhline(0.0, color="0.65", lw=0.8)
            ax.set_xlabel(xlabel)
            ax.set_title(f"Cycle {cycle}: {title}", loc="left")
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)

    for row in axes:
        row[0].set_ylabel("Book1 Mx [N m]\n(flexion negative, extension positive)")
    fig.suptitle(
        "L4-L5 intact: complete Optotrak-defined FE cycles 1-3 (stacked)\n"
        "Exact Book1 moments; Optotrak aligned using Motor-Angle timing; no regression or averaging",
        fontsize=12,
    )
    out_png = OUT_PNG
    try:
        fig.savefig(out_png, dpi=180)
    except OSError:
        out_png = OUT_DIR / "L4-L5_intact_FE_motor_vs_optotrak_angle_all_full_cycles_alt.png"
        fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"plotted rows per cycle [Motor, Optotrak]: {plotted_counts}")


if __name__ == "__main__":
    main()
