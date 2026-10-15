#%%
"""Export unsynchronized intact L4-L5 FE raw and referenced source data."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


TEST_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016\segment\6dof\L4-L5\1. Intact")
OUT_DIR = Path(__file__).resolve().parent / "plot_outputs"
OUT_DIR.mkdir(exist_ok=True)

MOTOR_FILE = TEST_DIR / "Motor" / "Book1.xlsx"
OPTOTRAK_FILE = TEST_DIR / "Optotrak" / "Lamina spreader L4-L5_2026_09_17_095853_001_cal.xls"
OUT_XLSX = OUT_DIR / "L4-L5_intact_FE_raw_and_referenced.xlsx"
OUT_MOTOR_CSV = OUT_DIR / "L4-L5_intact_FE_raw_motor_Mx.csv"
OUT_OPTO_RAW_CSV = OUT_DIR / "L4-L5_intact_FE_raw_optotrak_Ry.csv"
OUT_OPTO_REFERENCED_CSV = OUT_DIR / "L4-L5_intact_FE_optotrak_Ry_first_row_subtracted.csv"
OUT_RESAMPLED_CSV = OUT_DIR / "L4-L5_intact_FE_moment_vs_optotrack_angle_resampled.csv"
OUT_CYCLES_CSV = OUT_DIR / "L4-L5_intact_FE_optotrack_cycle_boundaries.csv"
OUT_ALIGNED_CSV = OUT_DIR / "L4-L5_intact_FE_moment_vs_optotrack_angle_aligned.csv"
OUT_ALIGNMENT_CSV = OUT_DIR / "L4-L5_intact_FE_motor_angle_optotrak_alignment.csv"
OUT_FULL_CYCLES_CSV = OUT_DIR / "L4-L5_intact_FE_full_cycle_boundaries.csv"

OPTO_MEDIAN_WINDOW = 31
OPTO_MOTION_LOOKAHEAD_S = 0.32
OPTO_MOTION_DEPARTURE_DEG = 0.38
OPTO_PEAK_PROMINENCE_DEG = 0.75
OPTO_PEAK_MIN_SEPARATION_S = 5.0
OPTO_FE_SIGN = -1.0
ANGLE_LAG_LIMIT_S = 20.0
ANGLE_LAG_STEP_S = 0.05
ANGLE_CORRELATION_GRID_STEP_S = 0.08
ANGLE_MATCH_MAX_ERROR_S = 0.005


def load_motor_raw() -> pd.DataFrame:
    motor = pd.read_excel(MOTOR_FILE)
    return pd.DataFrame({
        "motor_sample_index": range(1, len(motor) + 1),
        "motor_time_s": pd.to_numeric(motor["Time (s)"], errors="coerce"),
        "motor_angle_deg": pd.to_numeric(motor["Angle (deg)"], errors="coerce"),
        "moment_mx_nm": pd.to_numeric(motor["Mx (Nm)"], errors="coerce"),
    })


def load_optotrak_raw() -> pd.DataFrame:
    # The .xls extension contains tab-delimited text. The current saved source
    # has seven fields; the referenced angle is created on the third sheet.
    optotrak = pd.read_csv(OPTOTRAK_FILE, sep="\t", usecols=range(7))
    return pd.DataFrame({
        "optotrak_sample_index": range(1, len(optotrak) + 1),
        "optotrak_time_s": pd.to_numeric(optotrak["time"], errors="coerce"),
        "fe_angle_ry_raw_deg": pd.to_numeric(optotrak["Relative2(RB1, RB8) deg [Ry]"], errors="coerce"),
        "fe_angle_ry_sign_corrected_deg": OPTO_FE_SIGN * pd.to_numeric(
            optotrak["Relative2(RB1, RB8) deg [Ry]"], errors="coerce"
        ),
    })


def detect_optotrak_cycles(time_s: np.ndarray, angle_deg: np.ndarray) -> tuple[pd.DataFrame, list[float]]:
    """Find motion onset and positive FE peaks on the native Optotrak trace."""
    dt = float(np.nanmedian(np.diff(time_s)))
    smooth = pd.Series(angle_deg).rolling(OPTO_MEDIAN_WINDOW, center=True, min_periods=1).median().to_numpy(float)

    motion_start = None
    for index in range(time_s.size):
        future_index = int(np.searchsorted(time_s, time_s[index] + OPTO_MOTION_LOOKAHEAD_S))
        if future_index >= time_s.size:
            break
        departure = abs(float(smooth[future_index] - smooth[index]))
        if departure >= OPTO_MOTION_DEPARTURE_DEG:
            motion_start = index
            break
    if motion_start is None:
        raise RuntimeError("Could not identify a sustained Optotrak motion onset.")

    peaks, _ = find_peaks(
        smooth,
        prominence=OPTO_PEAK_PROMINENCE_DEG,
        distance=max(1, int(round(OPTO_PEAK_MIN_SEPARATION_S / dt))),
    )
    peaks = peaks[peaks > motion_start]
    if peaks.size < 3:
        raise RuntimeError(f"Expected three Optotrak FE peaks after onset; found {len(peaks)}.")

    boundaries = [float(time_s[motion_start]), *[float(time_s[index]) for index in peaks[:3]]]
    cycle_ids = np.full(time_s.shape, np.nan)
    for cycle in range(1, 4):
        cycle_ids[(time_s >= boundaries[cycle - 1]) & (time_s < boundaries[cycle])] = cycle

    cycle_rows = []
    for cycle in range(1, 4):
        start_index = motion_start if cycle == 1 else int(peaks[cycle - 2])
        end_index = int(peaks[cycle - 1])
        cycle_rows.append({
            "cycle": cycle,
            "start_time_s": boundaries[cycle - 1],
            "end_time_s": boundaries[cycle],
            "start_angle_deg": float(angle_deg[start_index]),
            "end_angle_deg": float(angle_deg[end_index]),
        })
    return pd.DataFrame(cycle_rows), boundaries


def estimate_motor_angle_lag(
    motor_time_s: np.ndarray,
    motor_angle_deg: np.ndarray,
    opto_time_s: np.ndarray,
    opto_angle_deg: np.ndarray,
) -> tuple[float, float]:
    """Estimate lag using Motor Angle and Optotrack angle trajectories.

    A negative lag means the Optotrack trajectory is shifted earlier onto the
    Motor time axis by that amount.
    """
    grid = np.arange(
        0.0,
        min(motor_time_s[-1], opto_time_s[-1]),
        ANGLE_CORRELATION_GRID_STEP_S,
    )
    motor_grid = np.interp(grid, motor_time_s, motor_angle_deg)
    opto_grid = np.interp(grid, opto_time_s, opto_angle_deg)
    motor_grid = (motor_grid - motor_grid.mean()) / (motor_grid.std() + 1e-12)
    opto_grid = (opto_grid - opto_grid.mean()) / (opto_grid.std() + 1e-12)

    best_lag = 0.0
    best_corr = -np.inf
    for lag_s in np.arange(-ANGLE_LAG_LIMIT_S, ANGLE_LAG_LIMIT_S + ANGLE_LAG_STEP_S / 2, ANGLE_LAG_STEP_S):
        shifted_motor = np.interp(grid + lag_s, grid, motor_grid, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted_motor)
        if valid.sum() < 20:
            continue
        corr = float(np.corrcoef(shifted_motor[valid], opto_grid[valid])[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_lag = float(lag_s)
    return best_lag, best_corr


def detect_full_cycles_on_aligned_angle(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
) -> tuple[pd.DataFrame, list[float]]:
    """Find complete cycles: test start to neutral return after each trough."""
    dt = float(np.nanmedian(np.diff(time_s)))
    smooth = pd.Series(angle_deg).rolling(5, center=True, min_periods=1).median().to_numpy(float)
    troughs, _ = find_peaks(
        -smooth,
        prominence=OPTO_PEAK_PROMINENCE_DEG,
        distance=max(1, int(round(OPTO_PEAK_MIN_SEPARATION_S / dt))),
    )
    if troughs.size < 3:
        raise RuntimeError(f"Expected three aligned FE troughs; found {len(troughs)}.")

    boundaries = [float(time_s[0])]
    for trough in troughs[:3]:
        candidates = np.flatnonzero(
            (np.arange(time_s.size - 1) > trough)
            & (smooth[:-1] < 0.0)
            & (smooth[1:] >= 0.0)
        )
        if candidates.size:
            index = int(candidates[0])
            fraction = -smooth[index] / (smooth[index + 1] - smooth[index])
            boundaries.append(float(time_s[index] + fraction * (time_s[index + 1] - time_s[index])))
        elif trough == troughs[2] and abs(float(smooth[-1])) <= 0.5:
            boundaries.append(float(time_s[-1]))
        else:
            break

    if len(boundaries) != 4:
        raise RuntimeError(f"Could not define three complete aligned FE cycles: {boundaries}")

    rows = pd.DataFrame({
        "cycle": [1, 2, 3],
        "start_time_s": boundaries[:-1],
        "end_time_s": boundaries[1:],
        "start_angle_deg": [float(np.interp(boundaries[i], time_s, smooth)) for i in range(3)],
        "end_angle_deg": [float(np.interp(boundaries[i + 1], time_s, smooth)) for i in range(3)],
    })
    return rows, boundaries


def main() -> None:
    motor_raw = load_motor_raw()
    optotrak_raw = load_optotrak_raw()
    first_angle = float(optotrak_raw["fe_angle_ry_raw_deg"].iloc[0])
    optotrak_referenced = optotrak_raw[["optotrak_sample_index", "optotrak_time_s", "fe_angle_ry_raw_deg"]].copy()
    optotrak_referenced["fe_angle_first_row_subtracted_deg"] = (
        optotrak_referenced["fe_angle_ry_raw_deg"] - first_angle
    )
    optotrak_referenced["fe_angle_first_row_subtracted_sign_corrected_deg"] = (
        OPTO_FE_SIGN * optotrak_referenced["fe_angle_first_row_subtracted_deg"]
    )
    optotrak_referenced["first_row_reference_deg"] = first_angle
    cycle_summary, opto_boundaries = detect_optotrak_cycles(
        optotrak_referenced["optotrak_time_s"].to_numpy(float),
        optotrak_referenced["fe_angle_first_row_subtracted_deg"].to_numpy(float),
    )
    optotrak_referenced["opto_cycle"] = np.nan
    for row in cycle_summary.itertuples(index=False):
        mask = (
            (optotrak_referenced["optotrak_time_s"] >= row.start_time_s)
            & (optotrak_referenced["optotrak_time_s"] < row.end_time_s)
        )
        optotrak_referenced.loc[mask, "opto_cycle"] = row.cycle

    motor_time = motor_raw["motor_time_s"].to_numpy(float)
    motor_angle = motor_raw["motor_angle_deg"].to_numpy(float)
    opto_time = optotrak_raw["optotrak_time_s"].to_numpy(float)
    if motor_time.min() < opto_time.min() or motor_time.max() > opto_time.max():
        raise ValueError("Motor time range is outside Optotrack coverage; resampling would extrapolate.")
    resampled = pd.DataFrame({
        "motor_sample_index": motor_raw["motor_sample_index"],
        "motor_time_s": motor_raw["motor_time_s"],
        "motor_angle_deg": motor_raw["motor_angle_deg"],
        "moment_mx_nm": motor_raw["moment_mx_nm"],
        "fe_angle_first_row_subtracted_resampled_deg": np.interp(
            motor_time,
            opto_time,
            optotrak_referenced["fe_angle_first_row_subtracted_deg"].to_numpy(float),
        ),
    })
    resampled["opto_cycle"] = np.nan
    resampled["fe_angle_first_row_subtracted_sign_corrected_resampled_deg"] = (
        OPTO_FE_SIGN * resampled["fe_angle_first_row_subtracted_resampled_deg"]
    )
    for row in cycle_summary.itertuples(index=False):
        mask = (resampled["motor_time_s"] >= row.start_time_s) & (resampled["motor_time_s"] < row.end_time_s)
        resampled.loc[mask, "opto_cycle"] = row.cycle

    # Align Optotrack to Motor time using the measured Motor Angle trajectory.
    opto_angle_corrected = optotrak_referenced["fe_angle_first_row_subtracted_sign_corrected_deg"].to_numpy(float)
    angle_lag_s, angle_lag_correlation = estimate_motor_angle_lag(
        motor_time, motor_angle, opto_time, opto_angle_corrected
    )
    opto_time_for_motor = motor_time - angle_lag_s
    right = np.searchsorted(opto_time, opto_time_for_motor, side="left").clip(1, len(opto_time) - 1)
    left = right - 1
    choose_left = np.abs(opto_time_for_motor - opto_time[left]) <= np.abs(opto_time[right] - opto_time_for_motor)
    nearest = np.where(choose_left, left, right)
    pairing_error_s = opto_time[nearest] - opto_time_for_motor
    measured = np.abs(pairing_error_s) <= ANGLE_MATCH_MAX_ERROR_S
    aligned_angle = np.full(motor_time.shape, np.nan)
    aligned_raw_angle = np.full(motor_time.shape, np.nan)
    aligned_opto_time = np.full(motor_time.shape, np.nan)
    aligned_source_row = np.full(motor_time.shape, np.nan)
    aligned_angle[measured] = opto_angle_corrected[nearest[measured]]
    aligned_raw_angle[measured] = optotrak_raw["fe_angle_ry_raw_deg"].to_numpy(float)[nearest[measured]]
    aligned_opto_time[measured] = opto_time[nearest[measured]]
    aligned_source_row[measured] = optotrak_raw["optotrak_sample_index"].to_numpy(int)[nearest[measured]]
    aligned_boundaries = [boundary + angle_lag_s for boundary in opto_boundaries]
    aligned_cycle = np.full(motor_time.shape, np.nan)
    for cycle in range(1, 4):
        mask = (motor_time >= aligned_boundaries[cycle - 1]) & (motor_time < aligned_boundaries[cycle])
        aligned_cycle[mask] = cycle
    aligned = pd.DataFrame({
        "opto_cycle": aligned_cycle,
        "motor_sample_index": motor_raw["motor_sample_index"],
        "motor_time_s": motor_time,
        "motor_angle_deg": motor_angle,
        "moment_mx_nm": motor_raw["moment_mx_nm"],
        "optotrak_source_row": aligned_source_row,
        "optotrak_time_s": aligned_opto_time,
        "optotrak_pairing_error_s": np.where(measured, pairing_error_s, np.nan),
        "fe_angle_ry_raw_deg": aligned_raw_angle,
        "fe_angle_sign_corrected_deg": aligned_angle,
        "optotrack_angle_measured": measured,
    })
    full_cycle_summary, full_boundaries = detect_full_cycles_on_aligned_angle(
        motor_time, aligned_angle
    )
    aligned["full_opto_cycle"] = np.nan
    for row in full_cycle_summary.itertuples(index=False):
        mask = (motor_time >= row.start_time_s) & (motor_time < row.end_time_s)
        aligned.loc[mask, "full_opto_cycle"] = row.cycle
    resampled["full_opto_cycle"] = np.nan
    for row in full_cycle_summary.itertuples(index=False):
        mask = (resampled["motor_time_s"] >= row.start_time_s) & (resampled["motor_time_s"] < row.end_time_s)
        resampled.loc[mask, "full_opto_cycle"] = row.cycle
    alignment_summary = pd.DataFrame([{
        "angle_lag_s": angle_lag_s,
        "angle_lag_correlation": angle_lag_correlation,
        "matched_motor_rows": int(measured.sum()),
        "total_motor_rows": len(motor_time),
        "max_accepted_pairing_error_s": float(np.nanmax(np.abs(pairing_error_s[measured]))) if measured.any() else np.nan,
    }])

    # These CSVs keep the different source sampling grids separate.
    motor_raw.to_csv(OUT_MOTOR_CSV, index=False)
    optotrak_raw.to_csv(OUT_OPTO_RAW_CSV, index=False)
    optotrak_referenced.to_csv(OUT_OPTO_REFERENCED_CSV, index=False)
    resampled.to_csv(OUT_RESAMPLED_CSV, index=False)
    cycle_summary.to_csv(OUT_CYCLES_CSV, index=False)
    aligned.to_csv(OUT_ALIGNED_CSV, index=False)
    alignment_summary.to_csv(OUT_ALIGNMENT_CSV, index=False)
    full_cycle_summary.to_csv(OUT_FULL_CYCLES_CSV, index=False)

    # Workbook sheets are intentionally separate: no row-wise Motor/Optotrak
    # pairing is implied because the sources have different clocks/rates.
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        motor_raw.to_excel(writer, sheet_name="raw_motor_Mx", index=False)
        optotrak_raw.to_excel(writer, sheet_name="raw_optotrak_Ry", index=False)
        optotrak_referenced.to_excel(writer, sheet_name="angle_first_row_subtracted", index=False)
        resampled.to_excel(writer, sheet_name="angle_resampled_to_motor", index=False)
        aligned.to_excel(writer, sheet_name="angle_aligned_to_motor", index=False)
        alignment_summary.to_excel(writer, sheet_name="angle_alignment_summary", index=False)
        full_cycle_summary.to_excel(writer, sheet_name="full_cycle_boundaries", index=False)

    print(f"Motor raw rows: {len(motor_raw)}")
    print(f"Optotrak raw rows: {len(optotrak_raw)}")
    print(f"Resampled angle rows: {len(resampled)}")
    print(f"Optotrak cycle boundaries: {opto_boundaries}")
    print(f"Motor-angle alignment lag: {angle_lag_s:+.2f} s, correlation={angle_lag_correlation:.3f}")
    print(f"Full-cycle boundaries: {full_boundaries}")
    print(f"First-row angle reference: {first_angle:+.6f} deg")
    print(f"saved {OUT_XLSX}")
    print(f"saved {OUT_MOTOR_CSV}")
    print(f"saved {OUT_OPTO_RAW_CSV}")
    print(f"saved {OUT_OPTO_REFERENCED_CSV}")
    print(f"saved {OUT_RESAMPLED_CSV}")
    print(f"saved {OUT_CYCLES_CSV}")
    print(f"saved {OUT_ALIGNED_CSV}")
    print(f"saved {OUT_ALIGNMENT_CSV}")
    print(f"saved {OUT_FULL_CYCLES_CSV}")


if __name__ == "__main__":
    main()