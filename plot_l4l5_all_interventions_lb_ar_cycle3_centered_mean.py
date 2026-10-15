#%%
"""Plot centered cycle-3 LB and AR curves for all L4-L5 interventions."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from plot_intact_l4l5_fe_loading_unloading_mean_by_cycle import centered_mean_curve
from plot_intact_l4l5_flexion_extension_6dof import fe_cycle_boundaries


LAB_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016")
LEVEL_DIR = LAB_DIR / "segment" / "6dof" / "L4-L5"
OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
CORRELATION_GRID_STEP_S = 0.08
LAG_STEP_S = 0.05
BROAD_LAG_LIMIT_S = 60.0
MIN_ALIGNMENT_CORRELATION = 0.90
MIN_BROAD_OVERLAP_FRACTION = 0.40
MOMENT_TROUGH_PROMINENCE_NM = 1.0
ANGLE_TROUGH_PROMINENCE_DEG = 0.3
MIN_TROUGH_SEPARATION_S = 2.0

AXES = {
    "LB": {
        "motor_index": 1,
        "moment_col": "My (Nm)",
        "opto_index": 1,
        "angle_col": "Relative2(RB1, RB8) deg [Rz]",
        "title": "L4-L5 Lateral Bending",
        "angle_label": "LB angle change, ΔRz [deg]",
        "moment_label": "My [Nm]",
        "cycle_number": 3,
    },
    "AR": {
        "motor_index": 2,
        "moment_col": "Mz (Nm)",
        "opto_index": 2,
        "angle_col": "Relative2(RB1, RB8) deg [Rx]",
        "title": "L4-L5 Axial Rotation",
        "angle_label": "AR angle change, ΔRx [deg]",
        "moment_label": "Mz [Nm]",
        "cycle_number": 3,
    },
}

# Partial facetectomy 2 has FE and LB in swapped file slots. Its AR files
# remain in the third slot.
FILE_OVERRIDES = {
    "LB": {
        "3. Partial facetectomy 2": {"motor_index": 0, "opto_index": 0},
    },
    "AR": {},
}


def numeric_book_number(path: Path) -> int:
    return int(re.search(r"(\d+)", path.stem).group(1))


def numeric_opto_number(path: Path) -> int:
    return int(re.search(r"_(\d+)_cal", path.name).group(1))


def intervention_folders() -> list[Path]:
    return sorted(
        [path for path in LEVEL_DIR.iterdir() if path.is_dir() and path.name[:1].isdigit()],
        key=lambda path: int(path.name.split(".")[0]),
    )


def load_motor(folder: Path, config: dict) -> tuple[np.ndarray, np.ndarray, str]:
    paths = sorted((folder / "Motor").glob("Book*.xlsx"), key=numeric_book_number)
    path = paths[config["motor_index"]]
    data = pd.read_excel(path)
    time_s = pd.to_numeric(data["Time (s)"], errors="coerce").to_numpy(float)
    moment = pd.to_numeric(data[config["moment_col"]], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_s) & np.isfinite(moment)
    return time_s[valid], moment[valid], path.name


def load_optotrak(folder: Path, config: dict) -> tuple[np.ndarray, np.ndarray, str]:
    paths = sorted((folder / "Optotrak").glob("*.xls"), key=numeric_opto_number)
    path = paths[config["opto_index"]]
    data = pd.read_csv(path, sep="\t", index_col=False, usecols=range(7))
    time_s = pd.to_numeric(data["time"], errors="coerce").to_numpy(float)
    raw_angle = pd.to_numeric(data[config["angle_col"]], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_s) & np.isfinite(raw_angle)
    time_s = time_s[valid]
    raw_angle = raw_angle[valid]
    return time_s, raw_angle - raw_angle[0], path.name


def _estimate_lag_with_limit(
    motor_time: np.ndarray,
    moment: np.ndarray,
    opto_time: np.ndarray,
    angle: np.ndarray,
    lag_limit_s: float,
) -> tuple[float, float, float]:
    grid = np.arange(0.0, min(motor_time[-1], opto_time[-1]), CORRELATION_GRID_STEP_S)
    motor_grid = np.interp(grid, motor_time, moment)
    angle_grid = np.interp(grid, opto_time, angle)
    motor_grid = (motor_grid - motor_grid.mean()) / (motor_grid.std() + 1e-12)
    angle_grid = (angle_grid - angle_grid.mean()) / (angle_grid.std() + 1e-12)
    best = (-np.inf, 0.0, 0.0)
    for lag in np.arange(-lag_limit_s, lag_limit_s + LAG_STEP_S / 2, LAG_STEP_S):
        shifted_motor = np.interp(grid + lag, grid, motor_grid, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted_motor)
        if valid.sum() < 20:
            continue
        correlation = float(np.corrcoef(shifted_motor[valid], angle_grid[valid])[0, 1])
        if correlation > best[0]:
            best = (correlation, float(lag), float(valid.mean()))
    return best


def estimate_bounded_lag(
    motor_time: np.ndarray,
    moment: np.ndarray,
    opto_time: np.ndarray,
    angle: np.ndarray,
    lag_limit_s: float,
) -> tuple[float, float]:
    best = _estimate_lag_with_limit(
        motor_time,
        moment,
        opto_time,
        angle,
        lag_limit_s,
    )
    if best[0] < MIN_ALIGNMENT_CORRELATION and lag_limit_s < BROAD_LAG_LIMIT_S:
        broad = _estimate_lag_with_limit(
            motor_time,
            moment,
            opto_time,
            angle,
            BROAD_LAG_LIMIT_S,
        )
        if broad[2] >= MIN_BROAD_OVERLAP_FRACTION and broad[0] > best[0]:
            return broad[1], broad[0]
    return best[1], best[0]


def estimate_cycle_consistent_lag(
    motor_time: np.ndarray,
    moment: np.ndarray,
    opto_time: np.ndarray,
    angle: np.ndarray,
) -> tuple[float, np.ndarray]:
    moment_smooth = pd.Series(moment).rolling(5, center=True, min_periods=1).median().to_numpy(float)
    angle_smooth = pd.Series(angle).rolling(31, center=True, min_periods=1).median().to_numpy(float)
    motor_dt = float(np.median(np.diff(motor_time)))
    opto_dt = float(np.median(np.diff(opto_time)))
    moment_troughs, _ = find_peaks(
        -moment_smooth,
        prominence=MOMENT_TROUGH_PROMINENCE_NM,
        distance=max(1, int(round(MIN_TROUGH_SEPARATION_S / motor_dt))),
    )
    angle_troughs, _ = find_peaks(
        -angle_smooth,
        prominence=ANGLE_TROUGH_PROMINENCE_DEG,
        distance=max(1, int(round(MIN_TROUGH_SEPARATION_S / opto_dt))),
    )
    if moment_troughs.size < 3 or angle_troughs.size < 3:
        raise RuntimeError(
            "Cycle-consistent alignment requires three moment and angle troughs; "
            f"found {moment_troughs.size} and {angle_troughs.size}."
        )

    trough_lags = motor_time[moment_troughs[:3]] - opto_time[angle_troughs[:3]]
    lag = float(np.median(trough_lags))
    if np.max(np.abs(trough_lags - lag)) > 1.0:
        raise RuntimeError(f"Corresponding cycle trough lags are inconsistent: {trough_lags}.")
    return lag, trough_lags


def align_angle_to_motor(
    motor_time: np.ndarray,
    moment: np.ndarray,
    opto_time: np.ndarray,
    angle: np.ndarray,
    lag_limit_s: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    lag, _ = estimate_cycle_consistent_lag(motor_time, moment, opto_time, angle)
    requested = motor_time - lag
    valid = (requested >= opto_time[0]) & (requested <= opto_time[-1])
    aligned = np.full(motor_time.shape, np.nan)
    aligned[valid] = np.interp(requested[valid], opto_time, angle)
    correlation = (
        float(np.corrcoef(moment[valid], aligned[valid])[0, 1])
        if valid.sum() > 1
        else float("nan")
    )
    return aligned, valid, lag, correlation


def plot_axis(axis_name: str, config: dict) -> None:
    cycle_number = config["cycle_number"]
    out_stem = f"L4-L5_all_interventions_{axis_name}_cycle{cycle_number}_centered_mean"
    out_png = OUT_DIR / f"{out_stem}.png"
    out_csv = OUT_DIR / f"{out_stem}.csv"
    out_range_csv = OUT_DIR / f"{out_stem}_angle_ranges.csv"

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    mean_rows = []
    range_rows = []
    results = []

    for color, folder in zip(COLORS, intervention_folders()):
        name = re.sub(r"^\d+\.\s*", "", folder.name)
        selected_config = dict(config)
        selected_config.update(FILE_OVERRIDES.get(axis_name, {}).get(folder.name, {}))
        motor_time, moment, motor_file = load_motor(folder, selected_config)
        opto_time, angle, opto_file = load_optotrak(folder, selected_config)
        boundaries = fe_cycle_boundaries(motor_time, moment)
        if len(boundaries) <= cycle_number:
            raise RuntimeError(
                f"{name} {axis_name}: found only {len(boundaries) - 1} complete cycles; "
                f"cannot isolate cycle {cycle_number}."
            )
        aligned_angle, aligned_valid, lag, correlation = align_angle_to_motor(
            motor_time,
            moment,
            opto_time,
            angle,
            boundaries[1] - boundaries[0],
        )
        cycle_start, cycle_end = boundaries[cycle_number - 1:cycle_number + 1]
        opto_available_start = opto_time[0] + lag
        opto_available_end = opto_time[-1] + lag
        if cycle_start < opto_available_start or cycle_end > opto_available_end:
            raise RuntimeError(
                f"{name} {axis_name} cycle {cycle_number}: Optotrak coverage does not "
                f"span the complete Motor interval ({cycle_start:.3f}-{cycle_end:.3f} s)."
            )
        keep = aligned_valid & (motor_time >= cycle_start) & (motor_time < cycle_end)
        cycle_angle = aligned_angle[keep]
        cycle_moment = moment[keep]
        if len(cycle_angle) < 20:
            raise RuntimeError(f"{name} {axis_name}: too few cycle-{cycle_number} samples.")

        mean_angle, mean_moment, _, _ = centered_mean_curve(cycle_angle, cycle_moment)
        ax.plot(mean_angle, mean_moment, color=color, lw=2.2, label=name)
        for phase, (angle_value, moment_value) in enumerate(zip(mean_angle, mean_moment)):
            mean_rows.append({
                "intervention": name,
                "phase_index": phase,
                "angle_mean_deg": angle_value,
                "moment_mean_nm": moment_value,
                "motor_angle_lag_s": lag,
                "motor_angle_lag_correlation": correlation,
                "n_cycle_rows": len(cycle_angle),
            })
        range_rows.append({
            "intervention": name,
            "cycle_extracted_angle_min_deg": float(np.min(cycle_angle)),
            "cycle_extracted_angle_max_deg": float(np.max(cycle_angle)),
            "centered_mean_curve_angle_min_deg": float(np.min(mean_angle)),
            "centered_mean_curve_angle_max_deg": float(np.max(mean_angle)),
        })
        results.append((name, motor_file, opto_file, len(cycle_angle), lag, correlation))

    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_xlabel(config["angle_label"])
    ax.set_ylabel(config["moment_label"])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
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
    fig.suptitle(config["title"], fontsize=12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    pd.DataFrame(mean_rows).to_csv(out_csv, index=False)
    pd.DataFrame(range_rows).to_csv(out_range_csv, index=False)
    print(f"saved {out_png}")
    print(f"saved {out_csv}")
    print(f"saved {out_range_csv}")
    for result in results:
        print(
            f"{result[0]}: motor={result[1]}, optotrak={result[2][-7:]}, "
            f"cycle{cycle_number} rows={result[3]}, lag={result[4]:+.2f}s, r={result[5]:.3f}"
        )


def main() -> None:
    for axis_name, config in AXES.items():
        plot_axis(axis_name, config)


if __name__ == "__main__":
    main()
