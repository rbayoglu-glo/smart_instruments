#%%
"""Stacked true-raw force/theta sync view for all level/side trials, filtered.

Definition for this plot:
- Force: DAQ channel with direction correction only (-1 * force), no baseline/preload
  subtraction, then MAD despiking + Butterworth low-pass.
- Theta: raw theta_deg from tracking solve (no theta0 subtraction), then median filter +
  MAD despiking + Butterworth low-pass.
- Time alignment: wall-clock alignment + lag refinement from force/theta cross-correlation
  (computed on the filtered signals below).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt

from lamina_spreader_lr import (
    FORCE_COL,
    FORCE_DIR,
    FORCE_SIGN,
    OUT_DIR,
    TRACK_DIR,
    TRIALS,
    break_tracking_gaps,
    find_lag,
    parse_force_anchor,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_shpose_only import (
    compute_theta_from_sh_pose,
)


FORCE_YLIM_N = (-25.0, 150.0)
THETA_YLIM_DEG = (-4.0, 7.0)

THETA_MEDIAN_KERNEL = 5
THETA_LOWPASS_CUTOFF_HZ = 1.2
FORCE_LOWPASS_CUTOFF_HZ = 1.5
BUTTER_ORDER = 2
THETA_DESPIKE_WINDOW = 15
THETA_DESPIKE_Z = 3.5
FORCE_DESPIKE_WINDOW = 11
FORCE_DESPIKE_Z = 3.5


def _odd_kernel_size(k: int) -> int:
    k = max(3, int(k))
    return k if k % 2 == 1 else k + 1


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.interpolate(limit_direction="both").to_numpy(dtype=float)


def _rolling_median_mad(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    w = _odd_kernel_size(window)
    s = pd.Series(np.asarray(x, dtype=float))
    med = s.rolling(w, center=True, min_periods=1).median().to_numpy(dtype=float)
    mad = (
        (s - pd.Series(med))
        .abs()
        .rolling(w, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    return med, mad


def despike_mad(x: np.ndarray, window: int, z: float) -> np.ndarray:
    """Replace isolated spikes using rolling median + MAD threshold."""
    y = _interpolate_nans(np.asarray(x, dtype=float))
    med, mad = _rolling_median_mad(y, window)
    sigma = np.maximum(1.4826 * mad, 1e-6)
    resid = np.abs(y - med)
    out = y.copy()
    out[resid > float(z) * sigma] = med[resid > float(z) * sigma]
    return out


def butter_lowpass_filter(x: np.ndarray, fs_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()

    y = _interpolate_nans(x)
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return y

    nyq = 0.5 * fs_hz
    if not np.isfinite(nyq) or nyq <= 0 or cutoff_hz <= 0:
        return y

    wn = min(0.99, float(cutoff_hz / nyq))
    if wn <= 0:
        return y

    if y.size < max(9, 3 * int(order) + 1):
        return y

    b, a = butter(int(order), wn, btype="low")
    try:
        yf = filtfilt(b, a, y)
    except ValueError:
        return y

    yf[~np.isfinite(x)] = np.nan
    return yf


def filter_theta(theta: np.ndarray, fs_hz: float) -> np.ndarray:
    k = _odd_kernel_size(THETA_MEDIAN_KERNEL)
    th = np.asarray(theta, dtype=float)

    filled = _interpolate_nans(th)
    med = medfilt(filled, kernel_size=k)
    med[~np.isfinite(th)] = np.nan
    dsp = despike_mad(med, THETA_DESPIKE_WINDOW, THETA_DESPIKE_Z)
    dsp[~np.isfinite(th)] = np.nan

    return butter_lowpass_filter(dsp, fs_hz, THETA_LOWPASS_CUTOFF_HZ, BUTTER_ORDER)


def filter_force(force: np.ndarray, fs_hz: float) -> np.ndarray:
    f_dsp = despike_mad(force, FORCE_DESPIKE_WINDOW, FORCE_DESPIKE_Z)
    f_dsp[~np.isfinite(force)] = np.nan
    return butter_lowpass_filter(f_dsp, fs_hz, FORCE_LOWPASS_CUTOFF_HZ, BUTTER_ORDER)


def load_force_true_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Time", FORCE_COL])
    df["t_s"] = df["Time"] / 1e6
    df["force_dir"] = FORCE_SIGN * pd.to_numeric(df[FORCE_COL], errors="coerce")

    anchor_est = parse_force_anchor(path)
    df["t_true_est"] = pd.to_datetime(anchor_est) + pd.to_timedelta(df["t_s"], unit="s")
    return df[["t_s", "force_dir", "t_true_est"]]


def main() -> None:
    trial_names = list(TRIALS.keys())
    n = len(trial_names)

    fig, axes = plt.subplots(n, 1, figsize=(14.0, 2.25 * n), squeeze=False)
    axes = axes.ravel()

    for i, (name, ax) in enumerate(zip(trial_names, axes)):
        print(f"=== {name} ===")

        kin = compute_theta_from_sh_pose(TRACK_DIR / TRIALS[name], valid_only=False)
        force = load_force_true_raw(FORCE_DIR / f"{name}.csv")

        if kin.empty or force.empty:
            ax.set_title(f"{name}   no data")
            ax.grid(alpha=0.3)
            continue

        t0 = min(kin["t_true_est"].min(), force["t_true_est"].min())
        kin_t = (kin["t_true_est"] - t0).dt.total_seconds().to_numpy()
        force_t = (force["t_true_est"] - t0).dt.total_seconds().to_numpy()

        kin_fs = 1.0 / np.nanmedian(np.diff(kin_t)) if kin_t.size >= 2 else np.nan
        force_fs = 1.0 / np.nanmedian(np.diff(force_t)) if force_t.size >= 2 else np.nan

        theta_raw = np.where(
            kin["valid"].to_numpy(),
            kin["theta_deg"].to_numpy(),
            np.nan,
        )
        theta_filt = filter_theta(theta_raw, kin_fs)
        force_filt = filter_force(force["force_dir"].to_numpy(), force_fs)

        theta_ok = np.isfinite(theta_filt)
        kin_for_lag = pd.DataFrame({"t_rel": kin_t[theta_ok], "theta_deg": theta_filt[theta_ok]})
        force_for_lag = pd.DataFrame({"t_rel": force_t, "force": force_filt})
        lag = find_lag(force_for_lag, kin_for_lag)
        force_t_sync = force_t - lag

        h_force, = ax.plot(
            force_t_sync,
            force_filt,
            lw=1.0,
            color="tab:blue",
            label="force (-1 x raw, despiked + low-pass), lag-synced",
        )
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(*FORCE_YLIM_N)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        kin_t_brk, theta_brk = break_tracking_gaps(kin_t, theta_filt)
        h_theta, = ax2.plot(
            kin_t_brk, theta_brk, lw=0.95, color="tab:orange", label="theta (median + despiked + low-pass)"
        )
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls=":", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(*THETA_YLIM_DEG)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax.set_title(f"{name}   filtered SH-pose-only sync check (lag {lag:+.2f} s)", fontsize=10)

        if i == 0:
            handles = [h_force, h_theta]
            ax.legend(handles, [h.get_label() for h in handles], loc="upper right", fontsize=8)

        if i == n - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        "Lamina spreader filtered sync check (stacked, SH pose only) - intact levels/sides\n"
        "Force: direction-corrected + MAD despike + Butterworth low-pass. "
        "Theta: median filter + MAD despike + Butterworth low-pass (no theta0 subtraction).",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = Path(OUT_DIR) / "sync_check_all_levels_sides_stacked_true_raw_filtered.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
