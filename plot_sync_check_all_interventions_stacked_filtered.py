from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt

import analyze_interventions as ai
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"
CASE_ORDER = [
    "2 - Intact Holding Longer",
    "3 - PUBF Left",
    "4 - FUF Left",
    "5 - FBF",
    "6 - Posterior Release",
    "7 - SPO",
]

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
    sigma = 1.4826 * mad
    # Keep a small floor to avoid division by tiny values in very flat sections.
    sigma = np.maximum(sigma, 1e-6)
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
    if not np.isfinite(nyq) or nyq <= 0:
        return y

    if cutoff_hz <= 0:
        return y

    # Avoid unstable settings when cutoff approaches Nyquist.
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


def load_synced_case(file_name: str) -> dict:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag0 = ls.find_lag(force, kin)
    lag, _ = ls.refine_lag(kin, force, lag0)
    d, _, _ = ls.sync(kin, force, lag)

    t = d["t_sync"].to_numpy(dtype=float)
    f = d["force"].to_numpy(dtype=float)
    if "theta_sm" in d.columns:
        th = d["theta_sm"].to_numpy(dtype=float)
    else:
        th = d["theta_dist"].to_numpy(dtype=float)
        print("warning: theta_sm not found; falling back to theta_dist")

    if t.size >= 2:
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = np.nan
    fs_hz = float(1.0 / dt) if np.isfinite(dt) and dt > 0 else np.nan

    f_dsp = despike_mad(f, FORCE_DESPIKE_WINDOW, FORCE_DESPIKE_Z)
    f_dsp[~np.isfinite(f)] = np.nan
    f_filt = butter_lowpass_filter(f_dsp, fs_hz, FORCE_LOWPASS_CUTOFF_HZ, BUTTER_ORDER)
    th_filt = filter_theta(th, fs_hz)

    return {
        "lag": float(lag),
        "fs_hz": fs_hz,
        "t": t,
        "f": f,
        "th": th,
        "f_filt": f_filt,
        "th_filt": th_filt,
    }


def ordered_cases() -> list[tuple[str, str]]:
    tracker = ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    out = []
    for file_name in CASE_ORDER:
        sub = tracker[tracker["File Name"] == file_name]
        if sub.empty:
            continue
        out.append((file_name, str(sub.iloc[0]["Intervention"])))
    return out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    cases = ordered_cases()
    if not cases:
        raise RuntimeError("No intervention cases resolved from tracker.")

    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(14.0, 2.25 * n), squeeze=False)
    axes = axes.ravel()

    for i, ((file_name, intervention), ax) in enumerate(zip(cases, axes)):
        print(f"=== {file_name} | {intervention} ===")
        case = load_synced_case(file_name)

        t = case["t"]
        f_filt = case["f_filt"]
        th_filt = case["th_filt"]

        h_force, = ax.plot(t, f_filt, lw=1.1, color="tab:blue", label="force (Butterworth)")
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax.twinx()
        h_theta, = ax2.plot(
            t,
            th_filt,
            lw=1.1,
            color="tab:orange",
            label="theta (un-subtracted, median + Butterworth)",
        )
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        fs_txt = f"{case['fs_hz']:.2f} Hz" if np.isfinite(case["fs_hz"]) else "N/A"
        ax.set_title(
            f"{file_name} | {intervention}   filtered sync check (lag {case['lag']:+.2f} s, fs {fs_txt})",
            fontsize=10,
        )
        ax.grid(alpha=0.3)

        if i == 0:
            handles = [h_force, h_theta]
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, loc="upper right", fontsize=8)

        if i == n - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        "Lamina spreader sync check (filtered) - all interventions\n"
        f"force: MAD despike (w={FORCE_DESPIKE_WINDOW}, z={FORCE_DESPIKE_Z:.1f}) + "
        f"Butterworth low-pass (order={BUTTER_ORDER}, fc={FORCE_LOWPASS_CUTOFF_HZ:.1f} Hz); "
        f"theta: median (k={THETA_MEDIAN_KERNEL}) + MAD despike "
        f"(w={THETA_DESPIKE_WINDOW}, z={THETA_DESPIKE_Z:.1f}) + Butterworth low-pass "
        f"(order={BUTTER_ORDER}, fc={THETA_LOWPASS_CUTOFF_HZ:.1f} Hz)",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / "sync_check_all_interventions_stacked_filtered.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
