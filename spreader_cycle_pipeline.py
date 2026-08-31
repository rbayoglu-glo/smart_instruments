"""Shared preload-free / theta0-corrected sync pipeline.

Reused across the force-tip-displacement and moment-theta chart scripts so
cycle boundaries, filtering, theta0 calibration, and ramp-start identification
stay identical across all downstream analyses.

Mirrors plot_sync_check_all_interventions_stacked_filtered_rampstart_preloadfree_theta0corr.py.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt

import analyze_interventions as ai
import lamina_spreader as ls


THETA_MEDIAN_KERNEL = 5
THETA_LOWPASS_CUTOFF_HZ = 1.2
FORCE_LOWPASS_CUTOFF_HZ = 1.5
BUTTER_ORDER = 2
THETA_DESPIKE_WINDOW = 15
THETA_DESPIKE_Z = 3.5
FORCE_DESPIKE_WINDOW = 11
FORCE_DESPIKE_Z = 3.5

SHORT_PEAK_GAP_MIN_S = 2.5
SHORT_PEAK_GAP_MAX_S = 10.0
SHORT_PEAK_GAP_FRAC = 0.35
MIN_RUN_SAMPLES = 30
MIN_RISE_SEC = 1.2
MIN_RISE_THETA_DEG = 0.2

# theta0 calibration: robust median over low-force samples, excluding
# contiguous low-force blocks that look like genuine non-neutral holds.
THETA0_FORCE_FRACTION = 0.05
THETA0_BLOCK_GAP_S = 1.0
THETA0_BLOCK_EXCLUDE_THETA_DEG = 1.0

# theta-zero snap: nudge force-based ramp start to nearest theta=0, unless the
# start is already clearly non-neutral (then leave it alone).
THETA_ZERO_SNAP_WINDOW_SEC = 1.5
THETA_ZERO_FORCE_FRACTION = 0.25
THETA_ZERO_SOURCE_MAX_ABS_DEG = 1.5


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


def dedupe_peak_candidates(cands: list[dict], t: np.ndarray) -> list[dict]:
    """Drop duplicate short-interval peaks; keep strongest candidate per cluster."""
    if len(cands) <= 1:
        return cands

    cands = sorted(cands, key=lambda c: c["pk"])
    peak_t = np.array([t[c["pk"]] for c in cands], dtype=float)
    diffs = np.diff(peak_t)
    if diffs.size == 0:
        return cands

    nominal = float(np.nanpercentile(diffs, 75))
    if not np.isfinite(nominal) or nominal <= 0:
        nominal = float(np.nanmedian(diffs))
    if not np.isfinite(nominal) or nominal <= 0:
        nominal = SHORT_PEAK_GAP_MAX_S

    min_gap_s = float(np.clip(SHORT_PEAK_GAP_FRAC * nominal, SHORT_PEAK_GAP_MIN_S, SHORT_PEAK_GAP_MAX_S))

    groups = [[cands[0]]]
    for c in cands[1:]:
        if t[c["pk"]] - t[groups[-1][-1]["pk"]] <= min_gap_s:
            groups[-1].append(c)
        else:
            groups.append([c])

    kept = []
    for g in groups:
        best = max(g, key=lambda c: (c["fpk"], c["dur"], c["pk"]))
        kept.append(best)
    return kept


def filter_pseudoramps(cands: list[dict], t: np.ndarray, th: np.ndarray) -> list[dict]:
    """Global cleanup: remove tiny pseudo-ramps consistently across all interventions."""
    out = []
    for c in cands:
        dt = float(t[c["pk"]] - t[c["i0"]])
        dtheta = float(th[c["pk"]] - th[c["i0"]])
        if c["dur"] < MIN_RUN_SAMPLES:
            continue
        if dt < MIN_RISE_SEC:
            continue
        if dtheta < MIN_RISE_THETA_DEG:
            continue
        out.append(c)
    return out


def compute_theta0_clean(t: np.ndarray, f: np.ndarray, th: np.ndarray, fmax: float) -> float:
    """Robust theta0: median over low-force samples, dropping non-neutral blocks.

    Mirrors ls.sync()'s theta0, but first splits the low-force selection into
    contiguous time blocks and excludes any block whose own median theta looks
    like a genuine non-neutral hold rather than a true unloaded/neutral dwell.
    """
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


def snap_start_to_theta_zero(
    t: np.ndarray,
    f: np.ndarray,
    th: np.ndarray,
    i0: int,
    pk: int,
) -> int:
    """Nudge force-based ramp start to the nearest theta~0 within the early ramp.

    Catches preload applied by pressing the tip into tissue at theta=0 (a
    supportive/tissue-contact force that isn't part of the actual distraction).
    Guardrail: skip snapping when the start is already clearly non-neutral, so
    genuine non-neutral starts (e.g. tool left bent between reps) are preserved.
    """
    if pk <= i0 + 2 or not np.isfinite(th[i0]):
        return int(i0)

    if abs(float(th[i0])) > THETA_ZERO_SOURCE_MAX_ABS_DEG:
        return int(i0)

    dt = float(np.nanmedian(np.diff(t))) if t.size >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    w = max(3, int(round(THETA_ZERO_SNAP_WINDOW_SEC / dt)))

    lo = max(0, int(i0) - w)
    hi = min(int(pk), int(i0) + w)
    if hi <= lo + 1:
        lo, hi = int(i0), int(pk)
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

    return int(idx[int(np.argmin(np.abs(th[idx])))])


def build_preloadfree_trace(f: np.ndarray, cands: list[dict]) -> np.ndarray:
    """Cycle-local preload-free force trace, referenced to each ramp start."""
    f_dyn = np.full_like(f, np.nan, dtype=float)
    for c in cands:
        i0 = int(c["i0"])
        e = int(c["e"])
        if i0 >= e:
            continue
        f0 = float(f[i0])
        f_dyn[i0:e] = f[i0:e] - f0
    return f_dyn


def load_case(file_name: str) -> dict:
    """Full pipeline for one intervention file: filtered, theta0-corrected,
    preload-free signals plus the list of identified cycles.

    Returns a dict with:
      lag, fs_hz, theta0, t, f (raw), th (raw, un-subtracted),
      f_filt (filtered, preload included), f_dyn (preload-free, filtered),
      th_filt (theta0-corrected, filtered), cycles (list of {s,e,i0,pk,fpk,dur}).
    """
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag0 = ls.find_lag(force, kin)
    lag, _ = ls.refine_lag(kin, force, lag0)
    d, _, fmax = ls.sync(kin, force, lag)

    t = d["t_sync"].to_numpy(dtype=float)
    f = d["force"].to_numpy(dtype=float)
    if "theta_sm" in d.columns:
        th_raw = d["theta_sm"].to_numpy(dtype=float)
    else:
        th_raw = d["theta_dist"].to_numpy(dtype=float)
        print("warning: theta_sm not found; falling back to theta_dist")

    theta0 = compute_theta0_clean(t, f, th_raw, fmax)
    th = th_raw - theta0

    if t.size >= 2:
        dt = float(np.nanmedian(np.diff(t)))
    else:
        dt = np.nan
    fs_hz = float(1.0 / dt) if np.isfinite(dt) and dt > 0 else np.nan

    f_dsp = despike_mad(f, FORCE_DESPIKE_WINDOW, FORCE_DESPIKE_Z)
    f_dsp[~np.isfinite(f)] = np.nan
    f_filt = butter_lowpass_filter(f_dsp, fs_hz, FORCE_LOWPASS_CUTOFF_HZ, BUTTER_ORDER)
    th_filt = filter_theta(th, fs_hz)

    fmax_filt = float(np.nanmax(f_filt)) if np.any(np.isfinite(f_filt)) else float(fmax)
    if not np.isfinite(fmax_filt) or fmax_filt <= 0:
        fmax_filt = float(fmax)

    runs = ls.segment_cycles(f_filt, ls.CYCLE_THRESH * fmax_filt)
    cands = []
    for s, e in runs:
        rb = ls.ramp_bounds(f_filt, s, e, fmax_filt)
        if rb is None:
            continue
        i0_force, _, pk = rb
        i0 = snap_start_to_theta_zero(t, f_filt, th_filt, i0_force, pk)
        cands.append(
            {
                "s": int(s),
                "e": int(e),
                "i0": int(i0),
                "pk": int(pk),
                "fpk": float(f_filt[pk]),
                "dur": int(e - s),
            }
        )

    cands = filter_pseudoramps(cands, t, th_filt)
    cands = dedupe_peak_candidates(cands, t)
    cands = sorted(cands, key=lambda c: c["pk"])

    f_dyn = build_preloadfree_trace(f_filt, cands)

    return {
        "lag": float(lag),
        "fs_hz": fs_hz,
        "theta0": float(theta0),
        "t": t,
        "f": f,
        "th": th_raw,
        "f_filt": f_filt,
        "f_dyn": f_dyn,
        "th_filt": th_filt,
        "cycles": cands,
    }
