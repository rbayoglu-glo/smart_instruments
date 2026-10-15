#%%
"""Stacked filtered + baseline-subtracted sync view, final step: ramp-start
detection with theta snap (no per-cycle preload removal).

Builds on plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline.py
(force: despiked + low-pass + global initial-window baseline removed; theta:
despiked + low-pass + global theta0 removed) and adds:

- Force and theta merged onto one timeline (kin/theta grid is master; force
  interpolated onto it), matching lamina_spreader.py's sync() convention.
- Cycle/ramp detection on the merged, baseline-subtracted force
  (segment_cycles + ramp_bounds, reused from lamina_spreader_lr.py).
- Optional ramp-start snap to theta~0 only when the preceding low-force state
    is itself neutral; an anatomical gap may have nonzero theta at zero force.

The plotted force/theta traces are exactly the global baseline-subtracted
signals from the previous step - only ramp-start markers are added, nothing
is re-referenced per cycle, so no real data is ever hidden.

Same global thresholds are used for every level/side - no per-trial tuning.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline import (
    compute_force_baseline,
    compute_theta0_clean,
)
from plot_sync_check_all_levels_sides_stacked_true_raw_shpose_only import (
    compute_theta_from_sh_pose,
)


FORCE_YLIM_N = (-25.0, 150.0)
THETA_YLIM_DEG = (-4.0, 7.0)

# One distraction = one prominent force peak. Prominence (not an absolute
# threshold crossing) is what separates back-to-back cycles when the operator
# does not fully release between them.
PEAK_MIN_FORCE_FRAC = 0.15
PEAK_PROMINENCE_FRAC = 0.15
PEAK_MIN_SEPARATION_S = 3.0

# Ramp start = foot of the rise leading into that peak.
FOOT_FRAC = 0.06
MAX_LOOKBACK_S = 30.0

MIN_RISE_N = 10.0
MIN_RISE_SEC = 1.2
# a distraction must actually open the level; force alone cannot distinguish a
# real ramp from the tool being handled or pressed against tissue
MIN_RISE_THETA_DEG = 1.0
# a re-grip or partial release mid-ramp shows up as theta falling back
MAX_RAMP_THETA_DRAWDOWN_DEG = 1.0
# an aborted engagement attempt loads then releases before the real climb
EARLY_RAMP_FORCE_FRACTION = 0.5
MAX_EARLY_FORCE_DRAWDOWN_FRAC = 0.08
MIN_THETA_COVERAGE = 0.70
POST_PEAK_MIN_THETA_COVERAGE = 0.80
# theta lags force, so its plateau is found after the force peak
THETA_PEAK_SEARCH_SEC = 2.0
THETA_DWELL_TOL_DEG = 0.35
MIN_THETA_DWELL_SEC = 0.50
POST_PEAK_FORCE_HOLD_FRACTION = 0.50
MIN_POST_PEAK_FORCE_HOLD_SEC = 3.0
# a second force rise and drop before release means the cycle is not one distraction
MIN_SECONDARY_RISE_SEP_S = 1.0

THETA_ZERO_SNAP_WINDOW_SEC = 1.5
THETA_ZERO_FORCE_FRACTION = 0.25
THETA_NEUTRAL_TOL_DEG = 0.35

# force and tracking have independent clocks, so the lag is searched over the whole recording
THORACIC_LAG_SEARCH_S = 200.0
THORACIC_LAG_STEP_S = 0.1
THORACIC_SYNC_GRID_S = 0.1
THORACIC_SYNC_MIN_OVERLAP_S = 20.0
THORACIC_SYNC_WINSOR_PCT = (1.0, 99.0)
THORACIC_MIN_SYNC_CORRELATION = 0.80


def find_force_peaks(t: np.ndarray, f: np.ndarray, fmax: float) -> np.ndarray:
    """One peak per distraction, separated by prominence rather than by a
    threshold crossing, so cycles run together when the operator pauses little."""
    dt = float(np.nanmedian(np.diff(t))) if t.size >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    peaks, _ = find_peaks(
        np.nan_to_num(f, nan=-np.inf),
        height=PEAK_MIN_FORCE_FRAC * fmax,
        prominence=PEAK_PROMINENCE_FRAC * fmax,
        distance=max(1, int(round(PEAK_MIN_SEPARATION_S / dt))),
    )
    return peaks


def find_thoracic_sync_lag(
    force_t: np.ndarray,
    force: np.ndarray,
    theta_t: np.ndarray,
    theta: np.ndarray,
) -> tuple[float, float]:
    """Lag maximizing force/theta correlation; `force_t - lag` is on the theta timeline.

    Both signals are winsorized so handling spikes cannot dominate the score, and
    only measured theta samples (no interpolated tracking gaps) are compared.
    Returns (lag_s, correlation).
    """
    force_ok = np.isfinite(force)
    theta_ok = np.isfinite(theta)
    if force_ok.sum() < 3 or theta_ok.sum() < 3:
        return 0.0, 0.0
    f_t, f = force_t[force_ok], force[force_ok]
    th_t, th = theta_t[theta_ok], theta[theta_ok]
    f = np.clip(f, *np.percentile(f, THORACIC_SYNC_WINSOR_PCT))
    th = np.clip(th, *np.percentile(th, THORACIC_SYNC_WINSOR_PCT))

    theta_dt = float(np.median(np.diff(th_t)))
    grid = np.arange(th_t[0], th_t[-1], THORACIC_SYNC_GRID_S)
    nearest = np.clip(np.searchsorted(th_t, grid), 1, th_t.size - 1)
    gap = np.minimum(np.abs(grid - th_t[nearest - 1]), np.abs(grid - th_t[nearest]))
    measured = gap <= 3.0 * theta_dt
    theta_grid = np.interp(grid, th_t, th)
    min_samples = int(THORACIC_SYNC_MIN_OVERLAP_S / THORACIC_SYNC_GRID_S)

    best_corr, best_lag = -np.inf, 0.0
    for lag in np.arange(
        -THORACIC_LAG_SEARCH_S,
        THORACIC_LAG_SEARCH_S + THORACIC_LAG_STEP_S / 2,
        THORACIC_LAG_STEP_S,
    ):
        t = grid + lag
        use = measured & (t >= f_t[0]) & (t <= f_t[-1])
        if use.sum() < min_samples:
            continue
        corr = np.corrcoef(theta_grid[use], np.interp(t[use], f_t, f))[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr, best_lag = float(corr), float(lag)
    return best_lag, max(best_corr, 0.0)


def ramp_start_from_peak(f: np.ndarray, t: np.ndarray, pk: int, lo: int) -> tuple[int, float] | None:
    """Foot of the rise into `pk`: last sample still at the pre-ramp force level.

    Walking back to the trough alone lands too early when the previous cycle is
    still relaxing; a fixed look-back lands too late on slow rises. Anchoring on
    a small fraction of this cycle's own rise avoids both.
    """
    if lo >= pk:
        return None
    imin = lo + int(np.nanargmin(f[lo:pk]))
    rise = float(f[pk] - f[imin])
    if not np.isfinite(rise) or rise < MIN_RISE_N:
        return None
    foot = f[imin] + FOOT_FRAC * rise
    seg = np.arange(imin, pk + 1)
    below = seg[np.nan_to_num(f[seg], nan=np.inf) <= foot]
    i0 = int(below[-1]) if below.size else int(imin)
    return i0, rise


def early_force_drawdown_frac(f: np.ndarray, i0: int, pk: int) -> float:
    """Largest force fall-back in the low-force part of the ramp, per unit rise.

    Confined to the early phase so a dip between twin peaks near full load does
    not read as an aborted attempt.
    """
    rise = float(f[pk] - f[i0])
    if not np.isfinite(rise) or rise <= 0:
        return 0.0
    seg = np.arange(i0, pk + 1)
    early = f[seg[f[seg] <= f[i0] + EARLY_RAMP_FORCE_FRACTION * rise]]
    if early.size < 2:
        return 0.0
    return float(np.max(np.maximum.accumulate(early) - early)) / rise


def theta_drawdown(th: np.ndarray, i0: int, pk: int) -> float:
    """Largest fall-back of theta below its running maximum across the ramp."""
    seg = th[i0:pk + 1]
    seg = seg[np.isfinite(seg)]
    if seg.size < 2:
        return 0.0
    return float(np.max(np.maximum.accumulate(seg) - seg))


def post_peak_force_hold_90pct_s(t: np.ndarray, f: np.ndarray, pk: int) -> float:
    threshold = POST_PEAK_FORCE_HOLD_FRACTION * float(f[pk])
    post = f[pk + 1:]
    below = np.flatnonzero(~np.isfinite(post) | (post < threshold))
    end = pk + 1 + int(below[0]) if below.size else f.size - 1
    return max(0.0, float(t[end] - t[pk]))


def has_theta_hold_after_peak(
    t: np.ndarray,
    th: np.ndarray,
    th_measured: np.ndarray,
    pk: int,
) -> bool:
    """Require theta to dwell near its maximum, i.e. a real hold at distraction.

    A cycle released immediately at the force peak shows a sharp theta apex
    instead of a plateau.
    """
    if not th_measured[pk] or not np.isfinite(th[pk]):
        return False

    search_end = int(np.searchsorted(t, t[pk] + THETA_PEAK_SEARCH_SEC, side="right"))
    seg = np.arange(pk, max(search_end, pk + 1), dtype=int)
    seg = seg[np.isfinite(th[seg])]
    if seg.size == 0:
        return False

    apex = int(seg[int(np.argmax(th[seg]))])
    floor = th[apex] - THETA_DWELL_TOL_DEG

    lo = apex
    while lo > 0 and np.isfinite(th[lo - 1]) and th[lo - 1] >= floor:
        lo -= 1
    hi = apex
    while hi < th.size - 1 and np.isfinite(th[hi + 1]) and th[hi + 1] >= floor:
        hi += 1

    if float(t[hi] - t[lo]) < MIN_THETA_DWELL_SEC:
        return False

    # coverage only over the required hold, since a long dwell can span dropouts
    hold_end = min(hi, int(np.searchsorted(t, t[apex] + MIN_THETA_DWELL_SEC, side="right")) - 1)
    return float(np.mean(th_measured[apex:hold_end + 1])) >= POST_PEAK_MIN_THETA_COVERAGE


def has_secondary_rise_in_cycle(
    t: np.ndarray,
    f: np.ndarray,
    i0: int,
    pk: int,
    next_pk: int,
) -> bool:
    """True when force rises and drops again between this peak and the cycle release.

    Release is the first return to the pre-ramp level; if force never releases before
    the next peak, the cycle ends at the trough preceding that peak.
    """
    foot = f[i0] + FOOT_FRAC * (f[pk] - f[i0])
    post = np.nan_to_num(f[pk + 1:], nan=np.inf)
    released = np.flatnonzero(post <= foot)
    if released.size:
        end = pk + 1 + int(released[0])
    elif next_pk > pk + 1:
        end = pk + 1 + int(np.argmin(post[:next_pk - pk - 1]))
    else:
        return False

    dt = float(np.nanmedian(np.diff(t))) if t.size >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    seg = np.nan_to_num(f[i0:end + 1], nan=-np.inf)
    rises, _ = find_peaks(
        seg,
        prominence=MIN_RISE_N,
        distance=max(1, int(round(MIN_SECONDARY_RISE_SEP_S / dt))),
    )
    return rises.size > 1


def detect_ramps(
    t: np.ndarray,
    f: np.ndarray,
    th: np.ndarray,
    th_measured: np.ndarray,
    track_span: tuple[float, float],
    fmax: float,
    require_force_hold: bool = True,
) -> list[dict]:
    """Ramp starts/peaks for every distraction, same global rules for all trials."""
    peaks = find_force_peaks(t, f, fmax)
    track_lo, track_hi = track_span
    cands: list[dict] = []
    prev_pk = -1

    for k, pk in enumerate(peaks):
        pk = int(pk)
        lo = max(int(np.searchsorted(t, t[pk] - MAX_LOOKBACK_S)), prev_pk + 1, 0)
        prev_pk = pk

        found = ramp_start_from_peak(f, t, pk, lo)
        if found is None:
            continue
        i0, rise = found

        i0 = snap_start_to_theta_zero(t, f, th, i0, pk)
        if float(t[pk] - t[i0]) < MIN_RISE_SEC:
            continue
        if t[pk] < track_lo or t[i0] > track_hi:
            continue

        coverage = float(np.mean(th_measured[i0:pk + 1])) if pk > i0 else 0.0
        if coverage < MIN_THETA_COVERAGE:
            continue
        dtheta = float(th[pk] - th[i0])
        if not (dtheta >= MIN_RISE_THETA_DEG):
            continue
        if theta_drawdown(th, i0, pk) > MAX_RAMP_THETA_DRAWDOWN_DEG:
            continue
        if early_force_drawdown_frac(f, i0, pk) > MAX_EARLY_FORCE_DRAWDOWN_FRAC:
            continue
        if not has_theta_hold_after_peak(t, th, th_measured, pk):
            continue
        next_pk = int(peaks[k + 1]) if k + 1 < peaks.size else f.size
        if has_secondary_rise_in_cycle(t, f, i0, pk, next_pk):
            continue

        force_hold_s = post_peak_force_hold_90pct_s(t, f, pk)
        if require_force_hold and force_hold_s < MIN_POST_PEAK_FORCE_HOLD_SEC:
            continue

        hi = int(peaks[k + 1]) if k + 1 < peaks.size else f.size
        e = pk + 1 + int(np.nanargmin(f[pk + 1:hi])) if hi > pk + 1 else f.size

        cands.append(
            {
                "cycle_id": k + 1,
                "i0": i0,
                "pk": pk,
                "e": int(e),
                "fpk": float(f[pk]),
                "rise_N": rise,
                "theta_cov": coverage,
                "post_peak_force_hold_90pct_s": force_hold_s,
            }
        )

    return cands


def snap_start_to_theta_zero(
    t: np.ndarray, f: np.ndarray, th: np.ndarray, i0: int, pk: int
) -> int:
    """Snap to theta zero only when the preceding low-force state is neutral."""
    if pk <= i0 + 2 or not np.isfinite(th[i0]):
        return int(i0)

    dt = float(np.nanmedian(np.diff(t))) if t.size >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.03
    w = max(3, int(round(THETA_ZERO_SNAP_WINDOW_SEC / dt)))

    lo = max(0, int(i0) - w)
    hi = int(i0)
    if hi <= lo + 1:
        return int(i0)

    idx = np.arange(lo, hi + 1, dtype=int)
    finite = np.isfinite(th[idx]) & np.isfinite(f[idx])
    idx = idx[finite]
    if idx.size == 0:
        return int(i0)

    f0 = float(f[i0])
    fpk = float(f[pk])
    rise = max(fpk - f0, 1e-9)
    early_cap = f0 + THETA_ZERO_FORCE_FRACTION * rise
    early = idx[f[idx] <= early_cap]
    if early.size >= 3:
        idx = early

    theta_center = float(np.nanmedian(th[idx])) if idx.size else np.nan
    if not np.isfinite(theta_center) or abs(theta_center) > THETA_NEUTRAL_TOL_DEG:
        return int(i0)

    return int(idx[int(np.argmin(np.abs(th[idx])))])


def process_trial(name: str) -> dict:
    kin = compute_theta_from_sh_pose(TRACK_DIR / TRIALS[name], valid_only=False)
    force = load_force_true_raw(FORCE_DIR / f"{name}.csv")

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

    # theta0 is estimated on the theta grid (same convention as the baseline step)
    f_on_kin = np.interp(kin_t, force_t_sync, force_dyn, left=np.nan, right=np.nan)
    fmax = float(np.nanmax(f_on_kin)) if np.any(np.isfinite(f_on_kin)) else np.nan
    theta0 = compute_theta0_clean(kin_t, f_on_kin, theta_filt, fmax) if np.isfinite(fmax) else 0.0
    theta_dyn = theta_filt - theta0

    # detection runs on the full force timeline so no force sample is ever dropped;
    # theta is bridged across tracking gaps only for the theta-based ramp checks
    fin = np.isfinite(theta_dyn)
    kin_valid_t = kin_t[fin]
    th_det = (
        np.interp(force_t_sync, kin_valid_t, theta_dyn[fin], left=np.nan, right=np.nan)
        if fin.any()
        else np.full_like(force_t_sync, np.nan)
    )

    if kin_valid_t.size >= 2:
        # "measured" = a tracked theta sample sits within a few frames of this instant;
        # dropouts are missing rows, so only time distance can reveal them
        kin_dt = float(np.nanmedian(np.diff(kin_valid_t)))
        j = np.clip(np.searchsorted(kin_valid_t, force_t_sync), 1, kin_valid_t.size - 1)
        gap = np.minimum(
            np.abs(force_t_sync - kin_valid_t[j - 1]), np.abs(kin_valid_t[j] - force_t_sync)
        )
        th_measured = gap <= 3.0 * kin_dt
        track_span = (float(kin_valid_t[0]), float(kin_valid_t[-1]))
    else:
        th_measured = np.zeros_like(force_t_sync, dtype=bool)
        track_span = (np.inf, -np.inf)

    t = force_t_sync
    f = force_dyn
    fmax_det = float(np.nanmax(f)) if f.size else np.nan

    cands = detect_ramps(t, f, th_det, th_measured, track_span, fmax_det) if np.isfinite(fmax_det) else []
    starts = np.array([c["i0"] for c in cands], dtype=int)

    for c in cands:
        print(
            f"  cycle {c['cycle_id']:2d}: start {t[c['i0']]:7.2f}s  peak {t[c['pk']]:7.2f}s  "
            f"rise {t[c['pk']] - t[c['i0']]:5.2f}s  F0 {f[c['i0']]:6.1f}N  "
            f"Fpk {c['fpk']:6.1f}N  theta0 {th_det[c['i0']]:+5.2f}deg  "
            f"dtheta {th_det[c['pk']] - th_det[c['i0']]:+5.2f}deg  cov {c['theta_cov']:.2f}"
        )

    return {
        "lag": float(lag),
        "force_baseline": force_baseline,
        "theta0": theta0,
        "force_t": t,
        "f_dyn": f,
        "kin_t": kin_t,
        "th": theta_dyn,
        "th_det": th_det,
        "cands": cands,
        "start_t": t[starts] if starts.size else np.array([]),
        "start_f": f[starts] if starts.size else np.array([]),
        "start_th": th_det[starts] if starts.size else np.array([]),
        "n_cycles": len(cands),
    }


def main() -> None:
    trial_names = list(TRIALS.keys())
    n = len(trial_names)

    fig, axes = plt.subplots(n, 1, figsize=(14.0, 2.25 * n), squeeze=False)
    axes = axes.ravel()

    for i, (name, ax) in enumerate(zip(trial_names, axes)):
        print(f"=== {name} ===")
        case = process_trial(name)

        force_t = case["force_t"]
        f_dyn = case["f_dyn"]
        kin_t = case["kin_t"]
        th = case["th"]
        start_t = case["start_t"]

        h_force, = ax.plot(force_t, f_dyn, lw=1.0, color="tab:blue", label="force (baseline-subtracted)")
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(*FORCE_YLIM_N)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        kin_t_brk, th_brk = break_tracking_gaps(kin_t, th)
        h_theta, = ax2.plot(kin_t_brk, th_brk, lw=0.95, color="tab:orange", label="theta (theta0-subtracted)")
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls=":", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(*THETA_YLIM_DEG)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        h_start_force = h_start_theta = None
        if start_t.size > 0:
            h_start_force = ax.scatter(
                start_t, case["start_f"], s=32, marker="o", facecolor="white",
                edgecolor="tab:red", linewidth=1.1, zorder=4, label="ramp start (force)",
            )
            h_start_theta = ax2.scatter(
                start_t, case["start_th"], s=26, marker="D", facecolor="white",
                edgecolor="tab:red", linewidth=0.9, zorder=4, label="ramp start theta (conditional snap)",
            )

        ax.set_title(
            f"{name}   ramp-start sync check "
            f"(lag {case['lag']:+.2f} s, {case['n_cycles']} cycles, theta0 {case['theta0']:+.2f} deg)",
            fontsize=10,
            loc="left",
        )

        if i == 0:
            handles = [h_force, h_theta]
            if h_start_force is not None:
                handles += [h_start_force, h_start_theta]
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, loc="upper left", fontsize=8)

        if i == n - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        "Lamina spreader ramp-start sync check (stacked) - intact levels/sides\n"
        "Force-foot starts; theta snap only from neutral low-force states; nonzero theta at zero force is retained.\n"
        "Filtered, baseline-subtracted force and theta0-subtracted theta; no per-cycle preload removal.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = Path(OUT_DIR) / "sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart.png"
    try:
        fig.savefig(out_png, dpi=160)
    except OSError:
        out_png = out_png.with_stem(f"{out_png.stem}_alt")
        fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
