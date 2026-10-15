#%%
"""Plot centered loading/unloading mean FE curves for all L4-L5 interventions."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import centered_mean_curve
from export_intact_l4l5_fe_raw_and_referenced import detect_full_cycles_on_aligned_angle


LAB_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016")
LEVEL_DIR = LAB_DIR / "segment" / "6dof" / "L4-L5"
OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "L4-L5_all_interventions_FE_cycle3_centered_mean.png"
OUT_CSV = OUT_DIR / "L4-L5_all_interventions_FE_cycle3_centered_mean.csv"
OUT_ANGLE_RANGE_CSV = OUT_DIR / "L4-L5_all_interventions_FE_cycle3_angle_ranges.csv"

OPTO_FE_SIGN = -1.0
LAG_LIMIT_S = 20.0
LAG_STEP_S = 0.05
CORRELATION_GRID_STEP_S = 0.08
ANGLE_MATCH_MAX_ERROR_S = 0.005
MEDIAN_WINDOW = 31
PEAK_PROMINENCE_DEG = 0.75
PEAK_MIN_SEPARATION_S = 5.0
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]

# The folder contains an explicit note that this intervention has FE/LB
# swapped. Use its second Motor/Optotrak files for the FE comparison.
FE_FILE_OVERRIDES = {
    "3. Partial facetectomy 2": {"motor_index": 1, "opto_index": 1, "moment_col": "Mx (Nm)"},
}


def numeric_book_number(path: Path) -> int:
    return int(re.search(r"(\d+)", path.stem).group(1))


def numeric_opto_number(path: Path) -> int:
    return int(re.search(r"_(\d+)_cal", path.name).group(1))


def load_motor(folder: Path, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((folder / "Motor").glob("Book*.xlsx"), key=numeric_book_number)
    path = paths[config.get("motor_index", 0)]
    data = pd.read_excel(path)
    moment_col = config.get("moment_col", "Mx (Nm)")
    time_s = pd.to_numeric(data["Time (s)"], errors="coerce").to_numpy(float)
    motor_angle = pd.to_numeric(data["Angle (deg)"], errors="coerce").to_numpy(float)
    moment = pd.to_numeric(data[moment_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_s) & np.isfinite(motor_angle) & np.isfinite(moment)
    return time_s[valid], motor_angle[valid], moment[valid]


def load_optotrak(folder: Path, config: dict) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted((folder / "Optotrak").glob("*.xls"), key=numeric_opto_number)
    path = paths[config.get("opto_index", 0)]
    # The .xls extension contains tab-delimited text; index_col=False prevents
    # pandas from treating the first time field as an implicit index.
    data = pd.read_csv(path, sep="\t", index_col=False, usecols=range(7))
    time_s = pd.to_numeric(data["time"], errors="coerce").to_numpy(float)
    raw_ry = pd.to_numeric(data["Relative2(RB1, RB8) deg [Ry]"], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_s) & np.isfinite(raw_ry)
    time_s = time_s[valid]
    angle = OPTO_FE_SIGN * (raw_ry[valid] - raw_ry[valid][0])
    return time_s, angle


def estimate_lag(motor_time: np.ndarray, motor_angle: np.ndarray, opto_time: np.ndarray, opto_angle: np.ndarray) -> tuple[float, float]:
    grid = np.arange(0.0, min(motor_time[-1], opto_time[-1]), CORRELATION_GRID_STEP_S)
    motor_grid = np.interp(grid, motor_time, motor_angle)
    opto_grid = np.interp(grid, opto_time, opto_angle)
    motor_grid = (motor_grid - motor_grid.mean()) / (motor_grid.std() + 1e-12)
    opto_grid = (opto_grid - opto_grid.mean()) / (opto_grid.std() + 1e-12)
    best = (-np.inf, 0.0)
    for lag in np.arange(-LAG_LIMIT_S, LAG_LIMIT_S + LAG_STEP_S / 2, LAG_STEP_S):
        shifted = np.interp(grid + lag, grid, motor_grid, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted)
        if valid.sum() < 20:
            continue
        corr = float(np.corrcoef(shifted[valid], opto_grid[valid])[0, 1])
        if corr > best[0]:
            best = (corr, float(lag))
    return best[1], best[0]


def align_optotrak_to_motor(
    motor_time: np.ndarray,
    motor_angle: np.ndarray,
    opto_time: np.ndarray,
    opto_angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    lag, correlation = estimate_lag(motor_time, motor_angle, opto_time, opto_angle)
    requested = motor_time - lag
    right = np.searchsorted(opto_time, requested, side="left").clip(1, len(opto_time) - 1)
    left = right - 1
    use_left = np.abs(requested - opto_time[left]) <= np.abs(opto_time[right] - requested)
    nearest = np.where(use_left, left, right)
    error = opto_time[nearest] - requested
    measured = np.abs(error) <= ANGLE_MATCH_MAX_ERROR_S
    aligned = np.full(motor_time.shape, np.nan)
    aligned[measured] = opto_angle[nearest[measured]]
    return aligned, measured, lag, correlation


def full_cycle_boundaries(time_s: np.ndarray, angle: np.ndarray) -> list[float]:
    valid = np.isfinite(angle)
    time_s = time_s[valid]
    angle = angle[valid]
    smooth = pd.Series(angle).rolling(MEDIAN_WINDOW, center=True, min_periods=1).median().to_numpy(float)
    dt = float(np.nanmedian(np.diff(time_s)))
    troughs, _ = find_peaks(-smooth, prominence=PEAK_PROMINENCE_DEG, distance=max(1, int(round(PEAK_MIN_SEPARATION_S / dt))))
    if troughs.size < 3:
        raise RuntimeError(f"Only {len(troughs)} FE troughs found.")
    boundaries = [float(time_s[0])]
    for trough in troughs[:3]:
        indices = np.arange(time_s.size - 1)
        crossings = indices[(indices > trough) & (smooth[:-1] < 0.0) & (smooth[1:] >= 0.0)]
        if crossings.size:
            i = int(crossings[0])
            frac = -smooth[i] / (smooth[i + 1] - smooth[i])
            boundaries.append(float(time_s[i] + frac * (time_s[i + 1] - time_s[i])))
        elif trough == troughs[2] and abs(float(smooth[-1])) <= 0.5:
            boundaries.append(float(time_s[-1]))
        else:
            break
    if len(boundaries) != 4:
        raise RuntimeError(f"Could not find three complete cycle boundaries: {boundaries}")
    return boundaries


def main() -> None:
    folders = sorted(
        [p for p in LEVEL_DIR.iterdir() if p.is_dir() and re.match(r"^\d+\.", p.name)],
        key=lambda p: int(p.name.split(".")[0]),
    )
    results = []
    mean_rows = []
    angle_range_rows = []
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)

    for color, folder in zip(COLORS, folders):
        name = re.sub(r"^\d+\.\s*", "", folder.name)
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
        cycle_start, cycle_end = boundaries[2], boundaries[3]
        keep = aligned_valid & (motor_time >= cycle_start) & (motor_time < cycle_end)
        cycle_angle = aligned_angle[keep]
        cycle_moment = moment[keep]
        if len(cycle_angle) < 20:
            raise RuntimeError(f"{name}: too few aligned cycle-3 samples ({len(cycle_angle)}).")

        mean_angle, mean_moment, angle_offset, moment_offset = centered_mean_curve(cycle_angle, cycle_moment)
        ax.plot(mean_angle, mean_moment, color=color, lw=2.2, label=name)
        angle_range_rows.append({
            "intervention": name,
            "cycle3_extracted_angle_min_deg": float(np.min(cycle_angle)),
            "cycle3_extracted_angle_max_deg": float(np.max(cycle_angle)),
            "centered_mean_curve_angle_min_deg": float(np.min(mean_angle)),
            "centered_mean_curve_angle_max_deg": float(np.max(mean_angle)),
        })
        for phase, (x_value, y_value) in enumerate(zip(mean_angle, mean_moment)):
            mean_rows.append({
                "intervention": name,
                "phase_index": phase,
                "angle_mean_deg": x_value,
                "moment_mx_mean_nm": y_value,
                "motor_angle_lag_s": lag,
                "motor_angle_lag_correlation": correlation,
                "n_cycle3_rows": len(cycle_angle),
            })
        results.append((name, len(cycle_angle), lag, correlation))

    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Mx [Nm]")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
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
    fig.suptitle("L4-L5 Flexion/Extension", fontsize=12)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    pd.DataFrame(mean_rows).to_csv(OUT_CSV, index=False)
    angle_ranges = pd.DataFrame(angle_range_rows)
    angle_ranges.to_csv(OUT_ANGLE_RANGE_CSV, index=False)

    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_ANGLE_RANGE_CSV}")
    print("Angle ranges [deg]:")
    print(angle_ranges.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    for result in results:
        print(f"{result[0]}: cycle3 rows={result[1]}, lag={result[2]:+.2f}s, r={result[3]:.3f}")


if __name__ == "__main__":
    main()
