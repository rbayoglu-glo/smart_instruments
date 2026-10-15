"""Run the shared intact-level left/right analysis on configured data."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEGACY_ANALYSIS_DIR = Path(
    r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis"
)
sys.path.insert(0, str(LEGACY_ANALYSIS_DIR))

from plot_sync_check_all_levels_sides_stacked_true_raw_filtered import (  # noqa: E402
    filter_force,
    filter_theta,
    load_force_true_raw,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline import (  # noqa: E402
    compute_force_baseline,
    compute_theta0_clean,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart import (  # noqa: E402
    THORACIC_MIN_SYNC_CORRELATION,
    detect_ramps,
    find_force_peaks,
    find_thoracic_sync_lag,
    ramp_start_from_peak,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_shpose_only import (  # noqa: E402
    compute_theta_from_sh_pose,
)
from lamina_spreader_lr import FORCE_COL, FORCE_SIGN, find_lag  # noqa: E402


SEGMENT_ROOT = Path(
    r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016\segment\spreader"
)
FORCE_DIR = SEGMENT_ROOT / "Force" / "lamspreader"
TRACKING_DIR = SEGMENT_ROOT / "Displacement" / "gns-sensorhub"
OUT_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis\output")

TRIALS = {
    "L4-L5 Left": {
        "force_file": "Session #80.csv",
        "tracking_folder": "2026-12-17_[01-56-11]",
    },
    "L4-L5 Right": {
        "force_file": "Session #81.csv",
        "tracking_folder": "2026-12-17_[02-05-40]",
    },
}

MOMENT_ARM_MM = 139.0
FORCE_FIT_LOW_N = 22.0
FORCE_FIT_HIGH_N = 67.0
FORCE_FIT_MIN_HIGH_N = 50.0
MIN_STIFFNESS_R2 = 0.95
FORCE_BASELINE_WINDOW_S = 1.0
FORCE_BASELINE_GUARD_S = 0.25
THETA_BASELINE_MIN_COVERAGE = 0.40
BASELINE_SEARCH_STEP_S = 0.25
BASELINE_FORCE_MEDIAN_TOL_N = 2.0
BASELINE_FORCE_ROBUST_RANGE_N = 5.0
BASELINE_THETA_MEDIAN_TOL_DEG = 0.30
BASELINE_THETA_ROBUST_RANGE_DEG = 0.60
PLOT_OUT_DIR = Path(
    r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis\output"
)
LEVEL_NAME = "L4-L5"
OUTPUT_PREFIX = ""
RELATIVE_TIME_MODE = False


def wide_lag(
    force_t: np.ndarray,
    force: np.ndarray,
    theta_t: np.ndarray,
    theta: np.ndarray,
) -> tuple[float, float]:
    grid = np.arange(0.0, min(force_t[-1], theta_t[-1]), 0.1)
    force_grid = np.interp(grid, force_t, force)
    theta_grid = np.interp(grid, theta_t, np.nan_to_num(theta, nan=np.nanmedian(theta)))
    force_grid = (force_grid - np.mean(force_grid)) / (np.std(force_grid) + 1e-9)
    theta_grid = (theta_grid - np.mean(theta_grid)) / (np.std(theta_grid) + 1e-9)

    best_score = -np.inf
    best_lag = 0.0
    for lag in np.arange(-200.0, 200.0 + 0.5, 0.5):
        shifted_force = np.interp(grid + lag, grid, force_grid, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted_force)
        if valid.sum() < 20:
            continue
        score = float(np.corrcoef(theta_grid[valid], shifted_force[valid])[0, 1])
        if score > best_score:
            best_score = score
            best_lag = float(lag)
    return best_lag, best_score


def load_force(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    force_df = pd.read_csv(path)
    force_t = force_df["Time"].to_numpy(float) / 1e6
    force_raw = FORCE_SIGN * pd.to_numeric(force_df[FORCE_COL], errors="coerce").to_numpy(float)
    force_fs = 1.0 / np.nanmedian(np.diff(force_t))
    return force_t, force_raw, filter_force(force_raw, force_fs)


def pre_cycle_force_baseline(
    force_filtered: np.ndarray,
    force_t_sync: np.ndarray,
    first_start_idx: int,
) -> tuple[float, float, float, float]:
    dt = float(np.nanmedian(np.diff(force_t_sync)))
    window_end = float(force_t_sync[first_start_idx] - FORCE_BASELINE_GUARD_S)
    window_start = window_end - FORCE_BASELINE_WINDOW_S
    in_window = (force_t_sync >= window_start) & (force_t_sync < window_end)
    samples = force_filtered[in_window]
    min_samples = max(3, int(0.8 * FORCE_BASELINE_WINDOW_S / dt))
    if samples.size < min_samples or not np.isfinite(samples).any():
        raise RuntimeError("insufficient finite samples in the pre-cycle force baseline window")
    baseline = float(np.nanmedian(samples))
    sigma_mad = float(1.4826 * np.nanmedian(np.abs(samples - baseline)))
    return baseline, window_start, window_end, sigma_mad


def pre_cycle_theta_baseline(
    theta_filtered: np.ndarray,
    theta_t: np.ndarray,
    window_start: float,
    window_end: float,
) -> float:
    dt = float(np.nanmedian(np.diff(theta_t)))
    in_window = (theta_t >= window_start) & (theta_t < window_end)
    samples = theta_filtered[in_window]
    samples = samples[np.isfinite(samples)]
    min_samples = max(3, int(np.ceil(THETA_BASELINE_MIN_COVERAGE * (window_end - window_start) / dt)))
    if samples.size < min_samples:
        raise RuntimeError("insufficient finite samples in the pre-cycle theta baseline window")
    return float(np.median(samples))


def select_stable_baseline_window(
    force_filtered: np.ndarray,
    force_t_sync: np.ndarray,
    theta_filtered: np.ndarray,
    theta_t: np.ndarray,
    first_start_idx: int,
    seed_force_baseline: float,
    seed_theta_baseline: float,
) -> tuple[float, float, float, float, float]:
    """Select the latest stable neutral window before the first ramp."""
    force_dt = float(np.nanmedian(np.diff(force_t_sync)))
    theta_dt = float(np.nanmedian(np.diff(theta_t)))
    first_start_t = float(force_t_sync[first_start_idx])
    latest_end = first_start_t - FORCE_BASELINE_GUARD_S
    earliest_start = float(force_t_sync[0])
    if latest_end - earliest_start < FORCE_BASELINE_WINDOW_S:
        raise RuntimeError("insufficient pre-ramp time for stable baseline-window search")

    min_force_samples = max(3, int(np.ceil(0.8 * FORCE_BASELINE_WINDOW_S / force_dt)))
    min_theta_samples = max(
        3, int(np.ceil(THETA_BASELINE_MIN_COVERAGE * FORCE_BASELINE_WINDOW_S / theta_dt))
    )
    windows: list[tuple[float, float, float, float, float]] = []
    for window_end in np.arange(
        earliest_start + FORCE_BASELINE_WINDOW_S,
        latest_end + 1e-9,
        BASELINE_SEARCH_STEP_S,
    ):
        window_start = float(window_end - FORCE_BASELINE_WINDOW_S)
        force_mask = (
            (force_t_sync >= window_start)
            & (force_t_sync < window_end)
            & np.isfinite(force_filtered)
        )
        theta_mask = (
            (theta_t >= window_start)
            & (theta_t < window_end)
            & np.isfinite(theta_filtered)
        )
        if force_mask.sum() < min_force_samples or theta_mask.sum() < min_theta_samples:
            continue

        force_percentiles = np.nanpercentile(force_filtered[force_mask], [10.0, 50.0, 90.0])
        theta_percentiles = np.nanpercentile(theta_filtered[theta_mask], [10.0, 50.0, 90.0])
        if abs(force_percentiles[1] - seed_force_baseline) > BASELINE_FORCE_MEDIAN_TOL_N:
            continue
        if force_percentiles[2] - force_percentiles[0] > BASELINE_FORCE_ROBUST_RANGE_N:
            continue
        if abs(theta_percentiles[1] - seed_theta_baseline) > BASELINE_THETA_MEDIAN_TOL_DEG:
            continue
        if theta_percentiles[2] - theta_percentiles[0] > BASELINE_THETA_ROBUST_RANGE_DEG:
            continue

        force_baseline = float(force_percentiles[1])
        theta_baseline = float(theta_percentiles[1])
        sigma_mad = float(
            1.4826
            * np.nanmedian(np.abs(force_filtered[force_mask] - force_baseline))
        )
        windows.append(
            (force_baseline, window_start, float(window_end), sigma_mad, theta_baseline)
        )

    if not windows:
        raise RuntimeError("no stable neutral baseline window found before the first ramp")
    return windows[-1]


def first_force_foot_index(force_t: np.ndarray, force_dynamic: np.ndarray) -> int:
    """Find a force-foot anchor when no cycle passes the full detector."""
    peaks = find_force_peaks(force_t, force_dynamic, float(np.nanmax(force_dynamic)))
    previous_peak = -1
    for peak in peaks:
        peak = int(peak)
        lookback = max(
            int(np.searchsorted(force_t, force_t[peak] - 30.0)),
            previous_peak + 1,
            0,
        )
        previous_peak = peak
        found = ramp_start_from_peak(force_dynamic, force_t, peak, lookback)
        if found is not None:
            return int(found[0])
    raise RuntimeError("no force-foot anchor found before baseline selection")


def split_synchronizable_trials() -> tuple[dict, dict[str, float]]:
    """Split TRIALS into those whose force/theta correlation passes the global sync gate."""
    usable: dict = {}
    rejected: dict[str, float] = {}
    for trial_name, config in TRIALS.items():
        tracking = compute_theta_from_sh_pose(TRACKING_DIR / config["tracking_folder"], valid_only=False)
        force_t, _, force_filtered = load_force(FORCE_DIR / config["force_file"])
        theta_t = (tracking["t_true_est"] - tracking["t_true_est"].min()).dt.total_seconds().to_numpy()
        theta_raw = np.where(
            tracking["valid"].to_numpy(bool), tracking["theta_deg"].to_numpy(float), np.nan
        )
        theta_filtered = filter_theta(theta_raw, 1.0 / np.nanmedian(np.diff(theta_t)))
        _, correlation = find_thoracic_sync_lag(force_t, force_filtered, theta_t, theta_filtered)
        if correlation >= THORACIC_MIN_SYNC_CORRELATION:
            usable[trial_name] = config
        else:
            rejected[trial_name] = correlation
    return usable, rejected


def process_trial(trial_name: str) -> tuple[dict, pd.DataFrame]:
    config = TRIALS[trial_name]
    tracking = compute_theta_from_sh_pose(
        TRACKING_DIR / config["tracking_folder"], valid_only=False
    )
    if RELATIVE_TIME_MODE:
        force_t, force_raw, force_filtered = load_force(FORCE_DIR / config["force_file"])
        theta_t = (
            tracking["t_true_est"] - tracking["t_true_est"].min()
        ).dt.total_seconds().to_numpy() + float(
            config.get("tracking_start_offset_s", 0.0)
        )
    else:
        force = load_force_true_raw(FORCE_DIR / config["force_file"])
        t0 = min(tracking["t_true_est"].min(), force["t_true_est"].min())
        theta_t = (tracking["t_true_est"] - t0).dt.total_seconds().to_numpy()
        force_t = (force["t_true_est"] - t0).dt.total_seconds().to_numpy()
    tracking_valid = tracking["valid"].to_numpy(bool)
    theta_raw = np.where(tracking_valid, tracking["theta_deg"].to_numpy(float), np.nan)
    theta_fs = 1.0 / np.nanmedian(np.diff(theta_t)) if theta_t.size >= 2 else np.nan
    theta_filtered = filter_theta(theta_raw, theta_fs)

    if not RELATIVE_TIME_MODE:
        force_raw = force["force_dir"].to_numpy(float)
        force_fs = 1.0 / np.nanmedian(np.diff(force_t)) if force_t.size >= 2 else np.nan
        force_filtered = filter_force(force_raw, force_fs)
    theta_ok = np.isfinite(theta_filtered)
    kin_for_lag = pd.DataFrame(
        {"t_rel": theta_t[theta_ok], "theta_deg": theta_filtered[theta_ok]}
    )
    force_for_lag = pd.DataFrame({"t_rel": force_t, "force": force_filtered})
    sync_warning = None
    if RELATIVE_TIME_MODE:
        lag, lag_score = find_thoracic_sync_lag(
            force_t,
            force_filtered,
            theta_t,
            theta_filtered,
        )
        if lag_score < THORACIC_MIN_SYNC_CORRELATION:
            raise RuntimeError(
                f"{trial_name}: force/theta synchronization unresolved "
                f"(best correlation {lag_score:.2f} < {THORACIC_MIN_SYNC_CORRELATION:.2f})"
            )
    else:
        lag = find_lag(force_for_lag, kin_for_lag)
        lag_score = float("nan")
    force_t_sync = force_t - lag
    measured_t = theta_t[tracking_valid & np.isfinite(theta_raw)]
    track_span = (float(measured_t[0]), float(measured_t[-1]))

    def process_with_baseline(force_baseline: float, theta0: float, require_force_hold: bool) -> dict:
        force_dynamic = force_filtered - force_baseline
        fmax = float(np.nanmax(force_dynamic))
        theta_dynamic = theta_filtered - theta0

        detection_valid = np.isfinite(theta_dynamic)
        detection_t = theta_t[detection_valid]
        detection_theta = theta_dynamic[detection_valid]
        theta_dt = float(np.nanmedian(np.diff(detection_t)))
        theta_on_force = np.interp(force_t_sync, detection_t, detection_theta, left=np.nan, right=np.nan)
        nearest = np.clip(np.searchsorted(detection_t, force_t_sync), 1, detection_t.size - 1)
        time_gap = np.minimum(
            np.abs(force_t_sync - detection_t[nearest - 1]),
            np.abs(force_t_sync - detection_t[nearest]),
        )
        measured_on_force = time_gap <= 3.0 * theta_dt
        candidates = detect_ramps(
            force_t_sync,
            force_dynamic,
            theta_on_force,
            measured_on_force,
            track_span,
            fmax,
            require_force_hold=require_force_hold,
        )
        return {
            "force_dynamic": force_dynamic,
            "theta0": theta0,
            "theta_dynamic": theta_dynamic,
            "theta_on_force": theta_on_force,
            "candidates": candidates,
        }

    initial_baseline = compute_force_baseline(force_filtered)
    initial_dynamic = force_filtered - initial_baseline
    # seed for anchor search only; final theta0 is sampled from the pre-cycle window
    initial_theta0 = compute_theta0_clean(
        theta_t,
        np.interp(theta_t, force_t_sync, initial_dynamic, left=np.nan, right=np.nan),
        theta_filtered,
        float(np.nanmax(initial_dynamic)),
    )
    initial = process_with_baseline(initial_baseline, initial_theta0, require_force_hold=False)
    processing_warning = sync_warning
    if not initial["candidates"]:
        if not RELATIVE_TIME_MODE:
            raise RuntimeError(f"No initial loading cycle found for {trial_name} to anchor the baseline.")
        baseline_anchor_idx = first_force_foot_index(force_t_sync, initial_dynamic)
        processing_warning = "no_initial_cycle_passed_detector; force-foot baseline anchor used"
    else:
        baseline_anchor_idx = int(initial["candidates"][0]["i0"])
    force_baseline, baseline_start, baseline_end, baseline_sigma_mad, theta0 = (
        select_stable_baseline_window(
            force_filtered,
            force_t_sync,
            theta_filtered,
            theta_t,
            baseline_anchor_idx,
            initial_baseline,
            initial_theta0,
        )
    )
    final = process_with_baseline(force_baseline, theta0, require_force_hold=True)
    if not final["candidates"]:
        if not RELATIVE_TIME_MODE:
            raise RuntimeError(f"No loading cycles found for {trial_name} after stable baseline selection.")
        processing_warning = (
            processing_warning or "stable baseline selected, but no cycle passed all detector gates"
        )
    force_dynamic = final["force_dynamic"]
    theta0 = final["theta0"]
    theta_dynamic = final["theta_dynamic"]
    theta_on_force = final["theta_on_force"]
    candidates = final["candidates"]

    rows: list[pd.DataFrame] = []
    for cycle, candidate in enumerate(candidates, start=1):
        cycle = int(candidate.get("cycle_id", cycle))
        samples = np.arange(candidate["i0"], candidate["pk"] + 1, dtype=int)
        valid = np.isfinite(force_dynamic[samples]) & np.isfinite(theta_on_force[samples])
        samples = samples[valid]
        if samples.size == 0:
            continue
        tip_mm = MOMENT_ARM_MM * np.radians(theta_on_force[samples])
        rows.append(
            pd.DataFrame(
                {
                    "trial": trial_name,
                    "cycle": cycle,
                    "time_s": force_t_sync[samples],
                    "force_N": force_dynamic[samples],
                    "theta_deg": theta_on_force[samples],
                    "tip_displacement_mm": tip_mm,
                }
            )
        )

    info = {
        "lag_s": lag,
        "lag_score": lag_score,
        "initial_force_baseline_N": initial_baseline,
        "force_baseline_N": force_baseline,
        "force_baseline_window_s": (baseline_start, baseline_end),
        "force_baseline_sigma_mad_N": baseline_sigma_mad,
        "theta0_deg": theta0,
        "force_t_sync": force_t_sync,
        "force_dynamic": force_dynamic,
        "theta_t": theta_t,
        "theta_dynamic": theta_dynamic,
        "theta_on_force": theta_on_force,
        "n_cycles": len(candidates),
        "candidates": candidates,
        "processing_warning": processing_warning,
    }
    if rows:
        data = pd.concat(rows, ignore_index=True)
    else:
        data = pd.DataFrame(
            columns=[
                "trial",
                "cycle",
                "time_s",
                "force_N",
                "theta_deg",
                "tip_displacement_mm",
            ]
        )
    return info, data


def line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def fit_cycle_force_range(sub: pd.DataFrame) -> dict:
    x = sub["tip_displacement_mm"].to_numpy(float)
    y = sub["force_N"].to_numpy(float)
    peak_force = float(np.nanmax(y))
    fit_high_n = min(FORCE_FIT_HIGH_N, peak_force)
    base = {
        "trial": sub["trial"].iloc[0],
        "cycle": int(sub["cycle"].iloc[0]),
        "cycle_peak_force_n": peak_force,
        "fit_force_low_n": FORCE_FIT_LOW_N,
        "r2_threshold": MIN_STIFFNESS_R2,
    }
    if peak_force < FORCE_FIT_MIN_HIGH_N:
        return {
            **base,
            "fit_force_high_n": np.nan,
            "ok": False,
            "reason": "max_force_below_50N",
        }
    low = np.flatnonzero((y[:-1] < FORCE_FIT_LOW_N) & (y[1:] >= FORCE_FIT_LOW_N))
    high = np.flatnonzero((y[:-1] < fit_high_n) & (y[1:] >= fit_high_n))
    base["fit_force_high_n"] = fit_high_n
    if low.size == 0 or not high[high >= low[0]].size:
        return {**base, "ok": False, "reason": "missing_force_fit_interval"}
    start = int(low[0])
    end = int(high[high >= start][0] + 1)
    xx, yy = x[start : end + 1], y[start : end + 1]
    if xx.size < 2 or np.ptp(xx) <= 1e-9:
        return {**base, "ok": False, "reason": "degenerate_force_fit_interval"}
    slope, intercept, r2 = line_fit(xx, yy)
    fit_ok = bool(r2 > MIN_STIFFNESS_R2)
    return {
        **base,
        "ok": fit_ok,
        "reason": "ok" if fit_ok else "r2_below_0.95",
        "slope_force_range_n_per_mm": slope,
        "intercept_n": intercept,
        "r2": r2,
        "span_mm": float(np.ptp(xx)),
        "fit_start": start,
        "fit_end": end,
    }


def plot_trial(data: pd.DataFrame, fits: list[dict], trial_name: str, out_path: Path) -> None:
    cycles = [fit["cycle"] for fit in fits]
    ncols = 3
    nrows = int(np.ceil(len(cycles) / ncols))
    fig = plt.figure(figsize=(4.8 * ncols, 3.7 * (nrows + 1)), constrained_layout=True)
    grid = fig.add_gridspec(nrows + 1, ncols)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(nrows * ncols)]
    stiffness_ax = fig.add_subplot(grid[nrows, :])

    for ax, fit in zip(axes, fits):
        sub = data[data["cycle"] == fit["cycle"]]
        cycle_color = "tab:green" if fit["ok"] else "tab:red"
        ax.plot(
            sub["tip_displacement_mm"],
            sub["force_N"],
            color="tab:blue",
            lw=1.1,
        )
        has_fit = (
            "fit_start" in fit
            and "fit_end" in fit
            and np.isfinite(fit.get("slope_force_range_n_per_mm", np.nan))
        )
        if has_fit:
            start, end = fit["fit_start"], fit["fit_end"]
            xx = sub["tip_displacement_mm"].to_numpy(float)[start : end + 1]
            yy = sub["force_N"].to_numpy(float)[start : end + 1]
            ax.plot(xx, yy, color=cycle_color, lw=2.0)
            ax.plot(
                xx,
                fit["slope_force_range_n_per_mm"] * xx + fit["intercept_n"],
                color=cycle_color,
                ls=":",
                lw=1.0,
            )
            title = (
                f"Cycle {fit['cycle']} | {'TRUE-LINEAR' if fit['ok'] else 'FLAGGED'} | "
                f"{fit['fit_force_low_n']:.0f}-"
                f"{fit['fit_force_high_n']:.0f} N | k={fit['slope_force_range_n_per_mm']:.2f} N/mm\n"
                f"R2={fit['r2']:.3f}"
            )
        else:
            title = f"Cycle {fit['cycle']} | N/A ({fit['reason']}, peak={fit['cycle_peak_force_n']:.1f} N)"
        ax.axhline(FORCE_FIT_LOW_N, color="0.75", lw=0.7, ls="--")
        if np.isfinite(fit["fit_force_high_n"]):
            ax.axhline(fit["fit_force_high_n"], color="0.75", lw=0.7, ls="--")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("tip displacement [mm]")
        ax.set_ylabel("force [N]")
        ax.grid(alpha=0.25)

    for ax in axes[len(cycles) :]:
        ax.axis("off")

    stiffness = np.asarray(
        [
            fit.get("slope_force_range_n_per_mm", np.nan) if fit["ok"] else np.nan
            for fit in fits
        ],
        dtype=float,
    )
    flagged_stiffness = np.asarray(
        [
            fit.get("slope_force_range_n_per_mm", np.nan)
            if not fit["ok"]
            else np.nan
            for fit in fits
        ],
        dtype=float,
    )
    stiffness_ax.plot(
        cycles, stiffness, color="tab:green", marker="o", ms=4, lw=1.4,
        label="true-linear fit (R2 > 0.95)",
    )
    if np.isfinite(flagged_stiffness).any():
        stiffness_ax.plot(
            cycles, flagged_stiffness, color="tab:red", marker="x", ms=6,
            linestyle="None", label="flagged fit (R2 <= 0.95)",
        )
    valid = np.isfinite(stiffness)
    if valid.any():
        mean = float(np.nanmean(stiffness))
        sd = float(np.nanstd(stiffness, ddof=1)) if valid.sum() > 1 else 0.0
        stiffness_ax.axhline(
            mean, color="tab:green", ls="--", lw=1.2,
            label=f"true-linear mean={mean:.2f} N/mm (n={valid.sum()})",
        )
        stiffness_ax.axhline(mean + sd, color="tab:red", ls=":", lw=1.0)
        stiffness_ax.axhline(mean - sd, color="tab:red", ls=":", lw=1.0)
        stiffness_ax.fill_between(
            cycles, mean - sd, mean + sd, color="tab:green", alpha=0.10,
            label=f"SD={sd:.2f} across true-linear fits",
        )
        stiffness_ax.legend(loc="upper left", fontsize=8)
    elif np.isfinite(flagged_stiffness).any():
        stiffness_ax.legend(loc="upper left", fontsize=8)
    stiffness_ax.set_title("Cycle-wise force-tip stiffness (true-linear: R2 > 0.95)", fontsize=10)
    stiffness_ax.set(xlabel="cycle number", ylabel="k [N/mm]")
    stiffness_ax.set_xticks(cycles)
    stiffness_ax.grid(alpha=0.25)
    fig.suptitle(
        f"{trial_name}: force vs tip displacement loading cycles\n"
        "Filtered baseline-subtracted signals; tip displacement = 139 mm x theta; "
        "fit range = 22 N to min(67 N, cycle peak); true-linear requires R2 > 0.95",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_fits: list[pd.DataFrame] = []

    for trial_name in TRIALS:
        print(f"=== {trial_name} ===")
        info, data = process_trial(trial_name)
        if data.empty and not RELATIVE_TIME_MODE:
            raise RuntimeError(f"No accepted loading cycles for {trial_name}.")
        fits = [fit_cycle_force_range(sub) for _, sub in data.groupby("cycle", sort=True)]
        stem = OUTPUT_PREFIX + trial_name.replace(" ", "_")
        data.to_csv(OUT_DIR / f"{stem}_force_tip_loading_cycles.csv", index=False)
        metrics = pd.DataFrame(fits)
        if metrics.empty:
            metrics = pd.DataFrame(
                columns=[
                    "trial", "cycle", "cycle_peak_force_n", "fit_force_low_n",
                    "r2_threshold", "fit_force_high_n", "ok", "reason",
                    "slope_force_range_n_per_mm", "intercept_n", "r2",
                    "span_mm", "fit_start", "fit_end",
                ]
            )
        metrics.to_csv(OUT_DIR / f"{stem}_translational_stiffness_by_cycle.csv", index=False)
        step = 5 if trial_name.endswith("Left") else 6
        plot_trial(
            data,
            fits,
            trial_name,
            PLOT_OUT_DIR / f"Step_{step:02d}_{stem}_force_tip_loading_cycles.png",
        )
        all_fits.append(metrics)
        print(
            f"  lag={info['lag_s']:+.2f}s score={info['lag_score']:.3f}; "
            f"{len(fits)} detected cycles; {int(metrics['ok'].sum())} true-linear fits "
            f"(R2 > {MIN_STIFFNESS_R2:.2f})"
        )
        if info.get("processing_warning"):
            print(f"  WARNING: {info['processing_warning']}")

    pd.concat(all_fits, ignore_index=True).to_csv(
        OUT_DIR / f"{OUTPUT_PREFIX}{LEVEL_NAME}_translational_stiffness_by_cycle.csv", index=False
    )
    print(f"saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()