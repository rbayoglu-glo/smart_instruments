#%%
"""Plot intact L4-L5 FE moment against direct Optotrack FE angle samples."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


LAB_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016")
TEST_DIR = LAB_DIR / "segment" / "6dof" / "L4-L5" / "1. Intact"
MOTOR_FILE = TEST_DIR / "Motor" / "Book1.xlsx"
OPTOTRAK_FILE = TEST_DIR / "Optotrak" / "Lamina spreader L4-L5_2026_09_17_095853_001_cal.xls"
OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)

MOTOR_TIME_COL = "Time (s)"
MOTOR_MOMENT_COL = "Mx (Nm)"
OPTOTRAK_TIME_COL = "time"
OPTOTRAK_ANGLE_COL = "angle"
RAW_MATCH_MAX_ERROR_S = 0.005
LAG_LIMIT_S = 20.0
LAG_STEP_S = 0.05
CORRELATION_GRID_STEP_S = 0.08
CYCLE_NUMBER = 3
NEGATIVE_LOBE_THRESHOLD_FRACTION = 0.10


def load_motor() -> tuple[np.ndarray, np.ndarray]:
    motor = pd.read_excel(MOTOR_FILE)
    required = {MOTOR_TIME_COL, MOTOR_MOMENT_COL}
    missing = required.difference(motor.columns)
    if missing:
        raise ValueError(f"Motor workbook is missing required columns: {sorted(missing)}")
    time_s = pd.to_numeric(motor[MOTOR_TIME_COL], errors="coerce").to_numpy(float)
    moment_nm = pd.to_numeric(motor[MOTOR_MOMENT_COL], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_s) & np.isfinite(moment_nm)
    return time_s[valid], moment_nm[valid]


def load_optotrak_fe() -> tuple[np.ndarray, np.ndarray]:
    # These Optotrack exports are tab-delimited text with a trailing tab per row.
    # The user-provided `angle` field is already referenced to the first frame.
    optotrak = pd.read_csv(OPTOTRAK_FILE, sep="\t", usecols=range(8))
    required = {OPTOTRAK_TIME_COL, OPTOTRAK_ANGLE_COL}
    missing = required.difference(optotrak.columns)
    if missing:
        raise ValueError(f"Optotrack export is missing required columns: {sorted(missing)}")
    time_s = pd.to_numeric(optotrak[OPTOTRAK_TIME_COL], errors="coerce").to_numpy(float)
    angle_deg = pd.to_numeric(optotrak[OPTOTRAK_ANGLE_COL], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_s) & np.isfinite(angle_deg)
    time_s = time_s[valid]
    angle_deg = angle_deg[valid]
    if time_s.size == 0:
        raise ValueError("Optotrack FE file contains no valid time/angle rows.")
    return time_s, angle_deg


def fe_cycle_boundaries(time_s: np.ndarray, moment_nm: np.ndarray) -> list[float]:
    """Use substantial negative-to-positive Mx crossings to delimit FE cycles."""
    negative_threshold = NEGATIVE_LOBE_THRESHOLD_FRACTION * float(np.nanmax(np.abs(moment_nm)))
    boundaries = [float(time_s[0])]
    previous_boundary_index = 0

    for index in range(moment_nm.size - 1):
        if not (moment_nm[index] <= 0.0 < moment_nm[index + 1]):
            continue
        prior_trough = float(np.nanmin(moment_nm[previous_boundary_index:index + 2]))
        if prior_trough > -negative_threshold:
            continue
        fraction = -moment_nm[index] / (moment_nm[index + 1] - moment_nm[index])
        crossing_time = time_s[index] + fraction * (time_s[index + 1] - time_s[index])
        boundaries.append(float(crossing_time))
        previous_boundary_index = index + 1

    return boundaries


def estimate_lag(
    motor_time_s: np.ndarray,
    moment_nm: np.ndarray,
    opto_time_s: np.ndarray,
    fe_angle_deg: np.ndarray,
) -> tuple[float, float]:
    """Find lag maximizing overlap correlation; positive means motor is later."""
    grid_end = min(motor_time_s[-1], opto_time_s[-1])
    grid = np.arange(0.0, grid_end, CORRELATION_GRID_STEP_S)
    moment_grid = np.interp(grid, motor_time_s, moment_nm)
    angle_grid = np.interp(grid, opto_time_s, fe_angle_deg)
    moment_grid = (moment_grid - moment_grid.mean()) / (moment_grid.std() + 1e-12)
    angle_grid = (angle_grid - angle_grid.mean()) / (angle_grid.std() + 1e-12)

    best_lag = 0.0
    best_corr = -np.inf
    for lag_s in np.arange(-LAG_LIMIT_S, LAG_LIMIT_S + LAG_STEP_S / 2, LAG_STEP_S):
        # Compare motor at t + lag with Optotrack angle at t.
        shifted_moment = np.interp(grid + lag_s, grid, moment_grid, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted_moment)
        if valid.sum() < 20:
            continue
        corr = float(np.corrcoef(shifted_moment[valid], angle_grid[valid])[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_lag = float(lag_s)
    return best_lag, best_corr


def main() -> None:
    motor_time_s, moment_nm = load_motor()
    opto_time_s, fe_angle_deg = load_optotrak_fe()

    boundaries = fe_cycle_boundaries(motor_time_s, moment_nm)
    if len(boundaries) <= CYCLE_NUMBER:
        raise RuntimeError(
            f"Cannot isolate FE cycle {CYCLE_NUMBER}: found only {len(boundaries) - 1} complete cycles."
        )
    cycle_start_s = boundaries[CYCLE_NUMBER - 1]
    cycle_end_s = boundaries[CYCLE_NUMBER]

    # Pair direct raw Optotrack samples at the shared test-time base. The motor
    # and Optotrack clocks require no lag correction and no interpolation.
    motor_samples = (motor_time_s >= cycle_start_s) & (motor_time_s <= cycle_end_s)
    paired_time_s = motor_time_s[motor_samples]
    paired_moment_nm = moment_nm[motor_samples]
    if paired_time_s.size == 0:
        raise RuntimeError(f"FE cycle {CYCLE_NUMBER} has no Motor samples.")
    right = np.searchsorted(opto_time_s, paired_time_s, side="left").clip(1, opto_time_s.size - 1)
    left = right - 1
    choose_left = np.abs(paired_time_s - opto_time_s[left]) <= np.abs(opto_time_s[right] - paired_time_s)
    nearest = np.where(choose_left, left, right)
    pairing_error_s = opto_time_s[nearest] - paired_time_s
    paired_angle_deg = np.full(paired_time_s.shape, np.nan)
    measured = np.abs(pairing_error_s) <= RAW_MATCH_MAX_ERROR_S
    paired_angle_deg[measured] = fe_angle_deg[nearest[measured]]
    paired = pd.DataFrame({
        "cycle": CYCLE_NUMBER,
        "motor_time_s": paired_time_s,
        "optotrack_time_s": np.where(measured, opto_time_s[nearest], np.nan),
        "optotrack_pairing_error_s": np.where(measured, pairing_error_s, np.nan),
        "moment_mx_nm": paired_moment_nm,
        "fe_angle_deg": paired_angle_deg,
        "optotrack_angle_measured": measured,
    })
    out_csv = OUT_DIR / f"L4-L5_intact_FE_cycle{CYCLE_NUMBER}_moment_vs_optotrack_angle.csv"
    paired.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    plot_valid = np.isfinite(paired_angle_deg)
    points = np.column_stack([paired_angle_deg[plot_valid], paired_moment_nm[plot_valid]])
    if points.shape[0] > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        collection = LineCollection(segments, cmap="viridis", linewidth=1.5, alpha=0.75)
        collection.set_array(paired_time_s[plot_valid][:-1])
        ax.add_collection(collection)
        colorbar = fig.colorbar(collection, ax=ax)
        colorbar.set_label("elapsed motor time [s]")
    ax.scatter(paired_angle_deg[plot_valid], paired_moment_nm[plot_valid], c="0.2", s=8, alpha=0.35, zorder=2)
    ax.scatter(paired_angle_deg[plot_valid][0], paired_moment_nm[plot_valid][0], color="tab:green", s=38, zorder=3, label="start")
    ax.scatter(paired_angle_deg[plot_valid][-1], paired_moment_nm[plot_valid][-1], color="tab:red", s=38, zorder=3, label="end")
    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_xlabel("FE angle, direct Optotrack angle [deg]")
    ax.set_ylabel("Mx moment [N·m] (flexion negative, extension positive)")
    cycle_duration_s = cycle_end_s - cycle_start_s
    coverage = float(np.mean(measured))
    ax.set_title(
        f"L4-L5 intact: FE cycle {CYCLE_NUMBER}, Mx vs direct Optotrack angle\n"
        f"Optotrack `angle` column; motor Angle unused; no lag or interpolation; measured coverage={coverage:.0%}",
        loc="left",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    out_png = OUT_DIR / f"L4-L5_intact_FE_cycle{CYCLE_NUMBER}_moment_vs_optotrack_angle.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"motor rows={len(motor_time_s)}, Optotrack rows={len(opto_time_s)}, paired rows={len(paired)}")
    print(f"Optotrack direct angle range=[{fe_angle_deg.min():+.3f}, {fe_angle_deg.max():+.3f}] deg")
    print(
        f"cycle {CYCLE_NUMBER} motor interval={cycle_start_s:.3f}-{cycle_end_s:.3f} s; "
        f"direct raw Optotrack coverage={coverage:.1%}"
    )
    print(f"saved {out_png}")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()