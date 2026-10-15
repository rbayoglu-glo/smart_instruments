#%%
"""Plot intact L4-L5 FE moment against first-row-subtracted Optotrak angle."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
INPUT_XLSX = OUT_DIR / "L4-L5_intact_FE_raw_and_referenced.xlsx"
OUT_PNG = OUT_DIR / "L4-L5_intact_FE_moment_vs_angle_cycles.png"

MOMENT_SHEET = "raw_motor_Mx"
ANGLE_SHEET = "angle_first_row_subtracted"


def cycle_boundaries(moment: np.ndarray, time_s: np.ndarray) -> list[float]:
    """Find three FE cycle boundaries from substantial negative-to-positive Mx crossings."""
    threshold = 0.10 * float(np.nanmax(np.abs(moment)))
    boundaries = [float(time_s[0])]
    previous_start = 0
    for index in range(moment.size - 1):
        if not (moment[index] <= 0.0 < moment[index + 1]):
            continue
        trough = float(np.nanmin(moment[previous_start : index + 2]))
        if trough > -threshold:
            continue
        fraction = -moment[index] / (moment[index + 1] - moment[index])
        boundaries.append(float(time_s[index] + fraction * (time_s[index + 1] - time_s[index])))
        previous_start = index + 1
    boundaries.append(float(time_s[-1]) + 1e-12)
    return boundaries


def main() -> None:
    motor = pd.read_excel(INPUT_XLSX, sheet_name=MOMENT_SHEET)
    angle_data = pd.read_excel(INPUT_XLSX, sheet_name=ANGLE_SHEET)

    motor_time = motor["motor_time_s"].to_numpy(float)
    moment = motor["moment_mx_nm"].to_numpy(float)
    angle_time = angle_data["optotrak_time_s"].to_numpy(float)
    angle = angle_data["fe_angle_first_row_subtracted_sign_corrected_deg"].to_numpy(float)

    boundaries = cycle_boundaries(moment, motor_time)
    if len(boundaries) < 4:
        raise RuntimeError(f"Expected three FE cycles; found {len(boundaries) - 1}.")

    fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    plotted = {}

    for cycle in range(1, 4):
        start, end = boundaries[cycle - 1], boundaries[cycle]
        keep = (motor_time >= start) & (motor_time < end)
        cycle_time = motor_time[keep]
        cycle_moment = moment[keep]
        if cycle_time.size == 0:
            continue

        # Pair each Motor sample to its nearest measured Optotrak row only.
        right = np.searchsorted(angle_time, cycle_time, side="left").clip(1, len(angle_time) - 1)
        left = right - 1
        use_left = np.abs(cycle_time - angle_time[left]) <= np.abs(angle_time[right] - cycle_time)
        nearest = np.where(use_left, left, right)
        pairing_error = np.abs(angle_time[nearest] - cycle_time)
        measured = pairing_error <= 0.005
        if not measured.any():
            continue

        x = angle[nearest[measured]]
        y = cycle_moment[measured]
        ax.plot(x, y, color=colors[cycle], lw=1.3, label=f"Cycle {cycle} (n={len(x)})")
        ax.scatter(x[0], y[0], color=colors[cycle], marker="o", s=28, zorder=3)
        ax.scatter(x[-1], y[-1], color=colors[cycle], marker="x", s=34, linewidth=1.1, zorder=3)
        plotted[cycle] = int(len(x))

    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_xlabel("Optotrack sign-corrected FE angle [deg]")
    ax.set_ylabel("Book1 Mx [N m]\n(flexion negative, extension positive)")
    ax.set_title(
        "L4-L5 intact: FE moment versus tracking angle\n"
        "Measured cycles only; no Motor Angle, lag, interpolation, regression, or average curve",
        loc="left",
        fontsize=11,
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    print(f"saved {OUT_PNG}")
    print(f"plotted rows per cycle: {plotted}")


if __name__ == "__main__":
    main()
