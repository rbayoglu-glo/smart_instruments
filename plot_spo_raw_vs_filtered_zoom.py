from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_sync_check_all_interventions_stacked_filtered as filt


OUT_DIR = Path(__file__).parent / "output"
FILE_NAME = "7 - SPO"
ZOOM_HALF_WINDOW_S = 15.0
THETA_FORCE_MASK_N = 5.0


def _norm_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    s = np.nanpercentile(d, 95)
    if not np.isfinite(s) or s <= 1e-12:
        s = np.nanmax(d)
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return d / s


def _plot_comparison(
    t: np.ndarray,
    f_raw: np.ndarray,
    f_filt: np.ndarray,
    th_raw: np.ndarray,
    th_filt: np.ndarray,
    t0: float,
    t1: float,
    title: str,
    out_path: Path,
) -> None:
    zoom = (t >= t0) & (t <= t1)

    fig, ax = plt.subplots(2, 2, figsize=(14.5, 7.5), sharex="col")

    # Full-range force
    ax[0, 0].plot(t, f_raw, color="0.75", lw=1.0, label="force raw")
    ax[0, 0].plot(t, f_filt, color="tab:blue", lw=1.2, label="force filtered")
    ax[0, 0].axvspan(t0, t1, color="tab:blue", alpha=0.12, lw=0)
    ax[0, 0].set_ylabel("force [N]")
    ax[0, 0].set_title("Force vs Time (full)")
    ax[0, 0].grid(alpha=0.25)
    ax[0, 0].legend(fontsize=8, loc="best")

    # Zoom force
    ax[0, 1].plot(t[zoom], f_raw[zoom], color="0.75", lw=1.0, label="force raw")
    ax[0, 1].plot(t[zoom], f_filt[zoom], color="tab:blue", lw=1.2, label="force filtered")
    ax[0, 1].set_ylabel("force [N]")
    ax[0, 1].set_title("Force vs Time (zoom)")
    ax[0, 1].grid(alpha=0.25)
    ax[0, 1].legend(fontsize=8, loc="best")

    # Full-range theta
    ax[1, 0].plot(t, th_raw, color="0.75", lw=1.0, label="theta raw")
    ax[1, 0].plot(t, th_filt, color="tab:orange", lw=1.2, label="theta filtered")
    ax[1, 0].axvspan(t0, t1, color="tab:orange", alpha=0.12, lw=0)
    ax[1, 0].set_ylabel("theta [deg]")
    ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].set_title("Theta vs Time (full)")
    ax[1, 0].grid(alpha=0.25)
    ax[1, 0].legend(fontsize=8, loc="best")

    # Zoom theta
    ax[1, 1].plot(t[zoom], th_raw[zoom], color="0.75", lw=1.0, label="theta raw")
    ax[1, 1].plot(t[zoom], th_filt[zoom], color="tab:orange", lw=1.2, label="theta filtered")
    ax[1, 1].set_ylabel("theta [deg]")
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_title("Theta vs Time (zoom)")
    ax[1, 1].grid(alpha=0.25)
    ax[1, 1].legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    case = filt.load_synced_case(FILE_NAME)
    t = case["t"]
    f_raw = case["f"]
    f_filt = case["f_filt"]
    th_raw = case["th"]
    th_filt = case["th_filt"]

    score_force = _norm_residual(f_raw, f_filt)
    score_theta = _norm_residual(th_raw, th_filt)
    score = np.nanmax(np.vstack([score_force, score_theta]), axis=0)
    i_spike = int(np.nanargmax(score))

    t0 = float(t[i_spike] - ZOOM_HALF_WINDOW_S)
    t1 = float(t[i_spike] + ZOOM_HALF_WINDOW_S)

    # Baseline comparison image (no masking).
    out = OUT_DIR / "spo_raw_vs_filtered_zoom.png"
    title = (
        f"SPO raw vs filtered comparison | zoom centered at t={t[i_spike]:.2f}s | "
        f"window=[{t0:.2f}, {t1:.2f}] s"
    )
    _plot_comparison(t, f_raw, f_filt, th_raw, th_filt, t0, t1, title, out)
    print(f"saved {out}")

    # Theta-masked comparison image: hide theta where force is below threshold.
    mask_loaded = np.isfinite(f_raw) & (f_raw >= THETA_FORCE_MASK_N)
    th_raw_masked = th_raw.copy()
    th_filt_masked = th_filt.copy()
    th_raw_masked[~mask_loaded] = np.nan
    th_filt_masked[~mask_loaded] = np.nan

    out_masked = OUT_DIR / "spo_raw_vs_filtered_zoom_theta_masked_force_ge5N.png"
    title_masked = (
        f"SPO raw vs filtered comparison (theta masked where force < {THETA_FORCE_MASK_N:.1f} N) | "
        f"zoom centered at t={t[i_spike]:.2f}s | window=[{t0:.2f}, {t1:.2f}] s"
    )
    _plot_comparison(
        t,
        f_raw,
        f_filt,
        th_raw_masked,
        th_filt_masked,
        t0,
        t1,
        title_masked,
        out_masked,
    )
    print(f"saved {out_masked}")
    print(f"spike_center_s={t[i_spike]:.3f}")
    print(f"zoom_window_s={t0:.3f},{t1:.3f}")


if __name__ == "__main__":
    main()
