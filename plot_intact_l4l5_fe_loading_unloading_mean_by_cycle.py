#%%
"""Plot loading/unloading-centered mean FE curves separately by cycle."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
INPUT_XLSX = OUT_DIR / "L4-L5_intact_FE_raw_and_referenced.xlsx"
OUT_PNG = OUT_DIR / "L4-L5_intact_FE_loading_unloading_mean_by_cycle.png"
OUT_CSV = OUT_DIR / "L4-L5_intact_FE_loading_unloading_mean_by_cycle.csv"
CYCLES = [1, 2, 3]
MEAN_POINTS = 500
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}


def cyclic_segment(values: np.ndarray, start: int, end: int) -> np.ndarray:
    if start <= end:
        indices = np.arange(start, end + 1)
    else:
        indices = np.concatenate((np.arange(start, values.size), np.arange(0, end + 1)))
    return values[indices]


def monotonic_by_moment(moment: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(moment)
    moment_sorted = moment[order]
    angle_sorted = angle[order]
    moment_unique, inverse = np.unique(moment_sorted, return_inverse=True)
    angle_sum = np.zeros(moment_unique.size, dtype=float)
    count = np.zeros(moment_unique.size, dtype=float)
    np.add.at(angle_sum, inverse, angle_sorted)
    np.add.at(count, inverse, 1.0)
    return moment_unique, angle_sum / np.maximum(count, 1.0)


def moving_average(values: np.ndarray, window: int = 11) -> np.ndarray:
    if values.size < 5:
        return values
    window = min(window, values.size if values.size % 2 else values.size - 1)
    if window < 3:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def centered_mean_curve(angle: np.ndarray, moment: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    min_index = int(np.argmin(moment))
    max_index = int(np.argmax(moment))
    if min_index == max_index:
        raise ValueError("Cycle does not contain distinct moment extrema.")

    moment_a = cyclic_segment(moment, min_index, max_index)
    angle_a = cyclic_segment(angle, min_index, max_index)
    moment_b = cyclic_segment(moment, max_index, min_index)
    angle_b = cyclic_segment(angle, max_index, min_index)
    moment_a, angle_a = monotonic_by_moment(moment_a, angle_a)
    moment_b, angle_b = monotonic_by_moment(moment_b, angle_b)

    low = max(moment_a.min(), moment_b.min())
    high = min(moment_a.max(), moment_b.max())
    common_moment = np.linspace(low, high, MEAN_POINTS)
    mean_angle = 0.5 * (
        np.interp(common_moment, moment_a, angle_a)
        + np.interp(common_moment, moment_b, angle_b)
    )
    mean_angle = moving_average(mean_angle)

    angle_at_zero = float(np.interp(0.0, common_moment, mean_angle)) if low <= 0.0 <= high else float(mean_angle[np.argmin(np.abs(common_moment))])
    moment_at_zero = 0.0 if low <= 0.0 <= high else float(common_moment[np.argmin(np.abs(common_moment))])
    return mean_angle - angle_at_zero, common_moment - moment_at_zero, angle_at_zero, moment_at_zero


def loading_window_start(motor_time: np.ndarray, moment: np.ndarray, cycle_start: float) -> float:
    crossings = np.flatnonzero(np.sign(moment[:-1]) * np.sign(moment[1:]) <= 0.0)
    before = crossings[motor_time[crossings + 1] <= cycle_start]
    return float(motor_time[before[-1]]) if before.size else float(cycle_start)


def loading_ramp_from_zero(angle: np.ndarray, moment: np.ndarray, peak_index: int) -> tuple[np.ndarray, np.ndarray]:
    peak_sign = np.sign(moment[peak_index])
    zero_side = np.flatnonzero(peak_sign * moment[:peak_index] <= 0.0)
    if zero_side.size == 0:
        raise ValueError("Loading ramp does not start at zero moment before its peak.")
    start = int(zero_side[-1])
    moment_start, moment_next = moment[start], moment[start + 1]
    fraction = -moment_start / (moment_next - moment_start)
    angle_zero = angle[start] + fraction * (angle[start + 1] - angle[start])
    ramp_moment = np.concatenate(([0.0], moment[start + 1:peak_index + 1]))
    ramp_angle = np.concatenate(([angle_zero], angle[start + 1:peak_index + 1]))
    return monotonic_by_moment(ramp_moment, ramp_angle)


def loading_ramp_curve(angle: np.ndarray, moment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    segments = []
    for peak_index in (int(np.argmin(moment)), int(np.argmax(moment))):
        ramp_moment, ramp_angle = loading_ramp_from_zero(angle, moment, peak_index)
        grid_moment = np.linspace(ramp_moment[0], ramp_moment[-1], MEAN_POINTS)
        grid_angle = moving_average(np.interp(grid_moment, ramp_moment, ramp_angle))
        grid_angle = grid_angle - float(np.interp(0.0, grid_moment, grid_angle))
        segments.append((grid_moment, grid_angle))

    (flexion_moment, flexion_angle), (extension_moment, extension_angle) = segments
    return np.concatenate((flexion_angle, extension_angle[1:])), np.concatenate((flexion_moment, extension_moment[1:]))


def main() -> None:
    data = pd.read_excel(INPUT_XLSX, sheet_name="angle_aligned_to_motor")
    data = data[data["full_opto_cycle"].isin(CYCLES) & data["optotrack_angle_measured"]].copy()
    data = data.sort_values("motor_time_s")

    panels = [
        ("motor_angle_deg", "Motor Angle [deg]", "Mx vs Motor Angle"),
        ("fe_angle_sign_corrected_deg", "Optotrak sign-corrected FE angle [deg]", "Mx vs Optotrak FE angle"),
    ]
    fig, axes = plt.subplots(len(CYCLES), 2, figsize=(14.0, 12.5), sharey=True, constrained_layout=True)
    csv_rows = []

    for row, cycle in enumerate(CYCLES):
        subset = data[data["full_opto_cycle"] == cycle]
        for col, (angle_column, xlabel, title) in enumerate(panels):
            ax = axes[row, col]
            angle = subset[angle_column].to_numpy(float)
            moment = subset["moment_mx_nm"].to_numpy(float)
            raw_valid = np.isfinite(angle) & np.isfinite(moment)
            angle = angle[raw_valid]
            moment = moment[raw_valid]
            mean_angle, mean_moment, angle_offset, moment_offset = centered_mean_curve(angle, moment)
            centered_angle = angle - angle_offset
            centered_moment = moment - moment_offset

            ax.plot(centered_angle, centered_moment, color=COLORS[cycle], lw=0.9, alpha=0.35, label=f"measured cycle {cycle}")
            ax.plot(mean_angle, mean_moment, color="tab:red", lw=2.2, label="Avg (loading/unloading), centered")
            ax.axvline(0.0, color="0.65", lw=0.8)
            ax.axhline(0.0, color="0.65", lw=0.8)
            ax.set_title(f"Cycle {cycle}: {title}", loc="left", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)

            if col == 0:
                ax.set_ylabel("Mx [N m]\n(flexion negative, extension positive)")

            if col == 0:
                for phase, (x_value, y_value) in enumerate(zip(mean_angle, mean_moment)):
                    csv_rows.append({
                        "cycle": cycle,
                        "phase_index": phase,
                        "motor_angle_mean_deg": x_value,
                        "moment_mean_nm": y_value,
                    })
            else:
                for phase, (x_value, y_value) in enumerate(zip(mean_angle, mean_moment)):
                    csv_rows[phase + (row * MEAN_POINTS)]["optotrak_angle_mean_deg"] = x_value

    fig.suptitle(
        "L4-L5 intact: loading/unloading-centered mean curves by cycle\n"
        "Red = average of the two moment-ordered branches; no polynomial regression",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_CSV}")
    print(f"cycles used={CYCLES}, mean points per curve={MEAN_POINTS}")


if __name__ == "__main__":
    main()
