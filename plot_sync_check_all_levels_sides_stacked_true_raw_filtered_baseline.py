#%%
"""Stacked filtered true-raw force/theta sync view, with baselines subtracted.

Builds on plot_sync_check_all_levels_sides_stacked_true_raw_filtered.py (force:
direction-corrected + despiked + low-pass; theta: median + despiked + low-pass)
and adds a single global baseline subtraction to each signal:

- Force baseline: median of the first FORCE_BASELINE_N filtered samples
  (initial unloaded window), subtracted from the whole filtered force trace.
- Theta baseline (theta0): robust median of filtered theta over low-force
  samples (force below THETA0_FORCE_FRACTION of trial peak), excluding any
  contiguous low-force block whose own median theta looks like a genuine
  non-neutral hold rather than a true neutral dwell.

No per-cycle preload removal and no ramp-start snapping yet - that is a later
step; this one only removes the two global baselines.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lamina_spreader_lr import (
    FORCE_DIR,
    OUT_DIR,
    TRACK_DIR,
    TRIALS,
    break_tracking_gaps,
    find_lag,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered import (
    filter_force,
    filter_theta,
    load_force_true_raw,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_shpose_only import (
    compute_theta_from_sh_pose,
)


FORCE_YLIM_N = (-25.0, 150.0)
THETA_YLIM_DEG = (-4.0, 7.0)

FORCE_BASELINE_N = 200

THETA0_FORCE_FRACTION = 0.05
THETA0_BLOCK_GAP_S = 1.0
THETA0_BLOCK_EXCLUDE_THETA_DEG = 1.0


def compute_force_baseline(force_filt: np.ndarray, n: int = FORCE_BASELINE_N) -> float:
    """Median of the initial unloaded window of the (already despiked) force trace."""
    window = force_filt[: min(n, force_filt.size)]
    return float(np.nanmedian(window))


def compute_theta0_clean(t: np.ndarray, f: np.ndarray, th: np.ndarray, fmax: float) -> float:
    """Robust theta0: median over low-force samples, dropping non-neutral blocks."""
    mask = np.isfinite(f) & (f < THETA0_FORCE_FRACTION * fmax) & np.isfinite(th)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return float(np.nanmedian(th))

    tt = t[idx]
    gaps = np.diff(tt, prepend=tt[0] - THETA0_BLOCK_GAP_S - 1.0)
    block_id = np.cumsum(gaps > THETA0_BLOCK_GAP_S)

    keep = np.ones(idx.size, dtype=bool)
    for b in np.unique(block_id):
        sel = block_id == b
        block_theta = th[idx[sel]]
        if abs(float(np.median(block_theta))) > THETA0_BLOCK_EXCLUDE_THETA_DEG:
            keep[sel] = False

    clean_idx = idx[keep]
    if clean_idx.size == 0:
        return float(np.nanmedian(th[idx]))
    return float(np.median(th[clean_idx]))


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

        force_baseline = compute_force_baseline(force_filt)
        force_dyn = force_filt - force_baseline

        f_on_kin = np.interp(kin_t, force_t_sync, force_dyn, left=np.nan, right=np.nan)
        fmax = float(np.nanmax(f_on_kin)) if np.any(np.isfinite(f_on_kin)) else np.nan
        theta0 = compute_theta0_clean(kin_t, f_on_kin, theta_filt, fmax) if np.isfinite(fmax) else 0.0
        theta_dyn = theta_filt - theta0

        print(f"  force baseline: {force_baseline:+.2f} N, theta0: {theta0:+.2f} deg")

        h_force, = ax.plot(
            force_t_sync,
            force_dyn,
            lw=1.0,
            color="tab:blue",
            label="force (baseline-subtracted)",
        )
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(*FORCE_YLIM_N)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        kin_t_brk, theta_brk = break_tracking_gaps(kin_t, theta_dyn)
        h_theta, = ax2.plot(
            kin_t_brk, theta_brk, lw=0.95, color="tab:orange", label="theta (theta0-subtracted)"
        )
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls=":", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(*THETA_YLIM_DEG)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax.set_title(
            f"{name}   filtered + baseline-subtracted SH-pose-only sync check "
            f"(lag {lag:+.2f} s, force0 {force_baseline:+.2f} N, theta0 {theta0:+.2f} deg)",
            fontsize=10,
        )

        if i == 0:
            handles = [h_force, h_theta]
            ax.legend(handles, [h.get_label() for h in handles], loc="upper right", fontsize=8)

        if i == n - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        "Lamina spreader filtered + baseline-subtracted sync check (stacked, SH pose only) - intact levels/sides\n"
        "Force: initial-window baseline subtracted. Theta: theta0 (low-force, non-neutral-excluded) subtracted. "
        "No per-cycle preload removal or ramp-start snapping yet.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = Path(OUT_DIR) / "sync_check_all_levels_sides_stacked_true_raw_filtered_baseline.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
