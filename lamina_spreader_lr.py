#%%
"""Lamina spreader lab analysis - 2026-11-08 intact-only session (Lab_09-04_2026).

Same physical instrument as the previous lab (Excelsius Array #8 DRB,
uuid d80d3bb5-9174-4bf3-811e-e31423da384a - CAD geometry confirmed identical
via this session's patterns_*.json), so the pattern/CAD frame constants below
are copied verbatim from Data_analysis/lamina_spreader.py.

Unlike the previous lab, this tracker export never resolves a live tool pose
(poses.txt "kind 2" rows are always an identity placeholder) - every marker,
including the array's own 4 balls, is logged only as an anonymous stray 3D
point.  So the array pose has to be solved here each frame from the raw
strays (best 4-of-N subset + point-correspondence permutation against the
known rigid marker template), where the previous lab's tracker already
handed us that pose and only the 5th marker needed a geometric search.

Sync model
----------
The force DAQ's own embedded clock (Session Created/Armed/Scan Start/Stop,
written into extra columns of the CSV header rows) runs a fixed 1h17m0s
ahead of true EST.  This was confirmed by (a) DAQ Scan Start - 1:17:00 landing
within 3-34s of a tracking_data folder's start time (itself true EST, since
recordingMetadata.json's date_time is true UTC) for all 8 level/side trials
with no folder re-used, and (b) matching the "Lamina Spreader Data
Collection.xlsx" Timings sheet's Computer Clock/EST columns' 1:17:00 gap.
Tracking recordings are short (~12-14 s @ ~30 Hz) snippets, not the full
5-15 minute force recording, so only a short force window is analyzed here.
"""

from pathlib import Path
from datetime import datetime, timedelta
import json
import re

import numpy as np
import pandas as pd

BASE = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016\intraop\Lab_09-04_2026")
FORCE_DIR = BASE / "force_data"
TRACK_DIR = BASE / "tracking_data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

FORCE_COL = "Spreader (SRC2475-LN1-02) on Ch 01.01.01 Calibrated Values"
FORCE_SIGN = -1.0

# DAQ computer's own clock reads this far ahead of true EST (see module docstring).
DAQ_CLOCK_AHEAD_OF_EST = timedelta(hours=1, minutes=17, seconds=0)
LAB_DATE = "2026-11-08"

# level/side -> tracking_data folder. Resolved by matching each recording's
# duration to the force trial's own duration (Scans Recorded / Scan Rate),
# in chronological order; agrees with the clock-offset model everywhere
# except L1-L2 Left, where "2026-11-08_[14-23-00]" is a 13.8 s false start
# and "..._[14-24-04]" (started 64 s later) is the real 283 s recording.
# "2026-11-08_[13-50-13]" (no pattern library) and "..._[15-23-27]" (165 s,
# doesn't match any trial duration) are likewise discarded false starts.
TRIALS = {
    "L1-L2 Left":  "2026-11-08_[14-24-04]",
    "L1-L2 Right": "2026-11-08_[14-41-43]",
    "L2-L3 Left":  "2026-11-08_[14-55-39]",
    "L2-L3 Right": "2026-11-08_[15-11-40]",
    "L3-L4 Left":  "2026-11-08_[15-27-44]",
    "L3-L4 Right": "2026-11-08_[15-37-44]",
    "L4-L5 Left":  "2026-11-08_[15-50-03]",
    "L4-L5 Right": "2026-11-08_[16-06-03]",
}

# --------------------------------------------------------------------------- #
# Excelsius Array #8 geometry (copied from lamina_spreader.py; confirmed
# identical in this session's patterns_*.json, same instrument_uuid).
# --------------------------------------------------------------------------- #
PATTERN_MARKERS = np.array(
    [
        [52.6051, 34.0, 56.1315],
        [54.2956, -30.0, 52.5063],
        [94.8670, -28.0, -34.4993],
        [91.4860, 28.0, -27.2488],
    ]
)
CAD_MARKERS = np.array(
    [
        [354.182, 93.064, 34.000],
        [350.556, 94.754, -30.000],
        [263.551, 135.326, -28.000],
        [270.801, 131.945, 28.000],
    ]
)
CAD_FIFTH_CLOSED = np.array([360.650, -48.000, 0.000])
# Contact-region midpoint distance from pivot [mm], used for CAD geometry.
D_O = 131.87
# Moment arm to the spreader tip [mm], used for force-to-moment conversion.
D_TIP = 139.0
CAD_PIVOT = np.array([D_O, 0.0, 0.0])

CIRCLE_TOL_MM = 2.5


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform R, t with dst ~= src @ R.T + t."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cd - r @ cs


from itertools import combinations, permutations

_PERMS4 = list(permutations(range(4)))
_TEMPLATE_PAIR_IDX = list(combinations(range(4), 2))
_TEMPLATE_DISTS = np.sort(
    [np.linalg.norm(PATTERN_MARKERS[i] - PATTERN_MARKERS[j]) for i, j in _TEMPLATE_PAIR_IDX]
)


def fit_array_pose(strays: np.ndarray, template: np.ndarray = PATTERN_MARKERS,
                    max_resid_mm: float = 1.0, dist_tol_mm: float = 4.0):
    """Best rigid fit of `template` (4 pts) onto an N-point stray cloud.

    The room has other reflective clutter besides the array, so N can run
    well past 4-5.  A cheap pairwise-distance pre-filter (every 4-subset's
    sorted pairwise distances vs. the template's) rules out almost all
    subsets before the expensive Kabsch + 24-permutation scoring runs only on
    survivors.  Returns (R, t, resid, subset_idx) with world ~= template @
    R.T + t for the winning point order, or None if nothing fits within
    max_resid_mm.
    """
    n = len(strays)
    if n < 4:
        return None
    pdist = np.linalg.norm(strays[:, None, :] - strays[None, :, :], axis=-1)

    best = None
    for subset in combinations(range(n), 4):
        d = np.sort([pdist[i, j] for i, j in _TEMPLATE_PAIR_IDX_SUBSET(subset)])
        if np.max(np.abs(d - _TEMPLATE_DISTS)) > dist_tol_mm:
            continue
        pts = strays[list(subset)]
        for perm in _PERMS4:
            r, t = kabsch(template[list(perm)], pts)
            fit = template[list(perm)] @ r.T + t
            resid = float(np.sqrt(np.mean(np.sum((fit - pts) ** 2, axis=1))))
            if best is None or resid < best[0]:
                best = (resid, r, t, subset)
    if best is None or best[0] > max_resid_mm:
        return None
    resid, r, t, subset = best
    return r, t, resid, subset


def _TEMPLATE_PAIR_IDX_SUBSET(subset):
    """(i, j) index pairs within `subset`, matching _TEMPLATE_PAIR_IDX order."""
    s = list(subset)
    return [(s[i], s[j]) for i, j in _TEMPLATE_PAIR_IDX]


# --------------------------------------------------------------------------- #
# tracking: raw strays -> theta_deg
# --------------------------------------------------------------------------- #
def parse_strays(track_folder: Path) -> dict[int, np.ndarray]:
    """Read the raw NDI localizer log, keeping only the stray 3D point clouds.

    This is the per-camera raw export (tracking_data/<session>/ndi/<serial>/
    NDI_Localizer.txt); it carries far more of the visible reflective clutter
    per frame than pf/poses.txt (which is already filtered/rate-limited), so
    it is the more reliable source for finding the array + fifth marker.
    """
    ndi_dir = track_folder / "ndi"
    tool_dir = next(ndi_dir.iterdir())
    path = tool_dir / "NDI_Localizer.txt"
    strays: dict[int, np.ndarray] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("//"):
                continue
            f = [s.strip() for s in line.split(",")]
            if len(f) < 3 or f[1] != "0":
                continue
            ts = int(f[0])
            cnt = int(f[4])
            if cnt:
                strays[ts] = np.asarray(f[5:5 + 3 * cnt], float).reshape(cnt, 3)
    return strays


def load_recording_meta(track_folder: Path) -> dict:
    meta = json.loads((track_folder / "recordingMetadata.json").read_text())
    start_utc = datetime.fromisoformat(meta["start"]["date_time"])
    start_est = (start_utc - timedelta(hours=5)).replace(tzinfo=None)
    return {"start_ts_us": meta["start"]["timestamp"], "start_est": start_est}


# largest credible frame-to-frame travel of the fifth marker, per second
MAX_STEP_MM_PER_S = 60.0


def compute_theta_from_tracking(track_folder: Path, valid_only: bool = True) -> pd.DataFrame:
    """Per-frame fifth-marker angle theta_deg, plus true EST wall-clock time.

    With `valid_only=False` the frames that fail the circle or continuity gate
    are returned too, flagged, so a plot can show where tracking degraded
    instead of silently omitting it.
    """
    r_cad, t_cad = kabsch(PATTERN_MARKERS, CAD_MARKERS)
    v0 = CAD_FIFTH_CLOSED[:2] - CAD_PIVOT[:2]
    r0 = float(np.linalg.norm(v0))

    strays = parse_strays(track_folder)
    meta = load_recording_meta(track_folder)

    rows = []
    n_frames = len(strays)
    n_fit = n_resolved = 0
    last_cad, last_ts = None, None
    for ts, pts in sorted(strays.items()):
        if len(pts) < 4:
            continue
        n_fit += 1
        fit = fit_array_pose(pts)
        if fit is None:
            continue
        r_wd, t_wd, resid, subset = fit
        n_resolved += 1

        leftover = [i for i in range(len(pts)) if i not in subset]
        if not leftover:
            continue
        world = pts[leftover]
        local = (world - t_wd) @ r_wd  # world -> pattern frame
        cad = local @ r_cad.T + t_cad  # pattern -> CAD frame
        dr = np.linalg.norm(cad[:, :2] - CAD_PIVOT[:2], axis=1) - r0
        err = np.hypot(dr, cad[:, 2] - CAD_FIFTH_CLOSED[2])
        keep = err < CIRCLE_TOL_MM
        if keep.any():
            cand, cerr, circle_ok = cad[keep], err[keep], True
        else:
            cand, cerr, circle_ok = cad, err, False

        if last_cad is None:
            # bootstrap: no prior track to continue, take the best circle fit
            j = int(np.argmin(cerr))
            step_ok = True
        else:
            # always require continuity from the last accepted point (capping
            # the elapsed-time budget so a long dropout can't license an
            # arbitrarily large jump); reject (keep old last_cad) rather than
            # snap to an unrelated clutter point that also passes the gate
            step = np.linalg.norm(cand - last_cad, axis=1)
            j = int(np.argmin(step))
            budget = MAX_STEP_MM_PER_S * min((ts - last_ts) / 1e6, 1.0)
            step_ok = step[j] <= budget

        valid = circle_ok and step_ok
        chosen = cand[j]
        if valid:
            last_cad, last_ts = chosen, ts
        elif valid_only:
            continue

        v = chosen[:2] - CAD_PIVOT[:2]
        theta = np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(v0[1], v0[0]))
        t_true_est = meta["start_est"] + timedelta(
            seconds=(ts - meta["start_ts_us"]) / 1e6
        )
        rows.append({"ts_us": ts, "theta_deg": theta, "circle_err_mm": float(cerr[j]),
                      "fit_resid_mm": resid, "t_true_est": t_true_est,
                      "valid": valid, "circle_ok": circle_ok, "step_ok": step_ok})

    df = pd.DataFrame(rows)
    n_valid = int(df["valid"].sum()) if not df.empty else 0
    print(f"{track_folder.name}: {n_frames} frames with strays, "
          f"{n_fit} with >=4 pts, {n_resolved} array-pose fits, "
          f"{n_valid} passed circle gate")
    if valid_only and not df.empty:
        df = df.drop(columns=["valid", "circle_ok", "step_ok"])
    return df


# --------------------------------------------------------------------------- #
# force: CSV -> true-EST-anchored (t_s, force)
# --------------------------------------------------------------------------- #
def parse_force_anchor(path: Path) -> datetime:
    """True-EST wall clock for this file's Scan# 0 (Time == 0), from the
    embedded "Scan Start" annotation minus the fixed DAQ clock offset.
    """
    with path.open() as fh:
        for line in fh:
            f = [s.strip() for s in line.rstrip("\n").split(",")]
            if len(f) < 8 or f[6] != "Scan Start":
                continue
            scan_no, time_us = int(f[0]), int(f[1])
            daq_clock = datetime.strptime(f"{LAB_DATE} {f[7]}", "%Y-%m-%d %H:%M:%S")
            true_est_at_row = daq_clock - DAQ_CLOCK_AHEAD_OF_EST
            return true_est_at_row - timedelta(seconds=time_us / 1e6)
    raise ValueError(f"no 'Scan Start' annotation found in {path}")


def load_force(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Scan #", "Time", FORCE_COL])
    df["t_s"] = df["Time"] / 1e6
    df["force"] = FORCE_SIGN * (df[FORCE_COL] - df[FORCE_COL].iloc[:200].median())
    anchor_est = parse_force_anchor(path)
    df["t_true_est"] = [anchor_est + timedelta(seconds=t) for t in df["t_s"]]
    print(f"{path.name}: {len(df)} samples, {df['t_s'].iloc[-1]:.1f} s, "
          f"peak {df['force'].max():.2f} N, true-EST start {anchor_est}")
    return df[["t_s", "force", "t_true_est"]]


# --------------------------------------------------------------------------- #
# sync + stiffness
# --------------------------------------------------------------------------- #
def theta_deg_to_tip_mm(theta_deg):
    return D_O * np.radians(theta_deg)


def find_lag(force: pd.DataFrame, kin: pd.DataFrame, search_s: float = 30.0,
             fs: float = 20.0) -> float:
    """Cross-correlate force vs. theta over their overlapping true-EST window
    to refine the coarse (wall-clock) alignment by a few seconds.
    """
    t0 = kin["t_rel"].iloc[0] - search_s
    t1 = kin["t_rel"].iloc[-1] + search_s
    grid = np.arange(t0, t1, 1.0 / fs)

    def prep(t, y):
        s = pd.Series(y).interpolate(limit_direction="both")
        yi = np.interp(grid, t, s, left=np.nan, right=np.nan)
        yi = np.nan_to_num(yi - np.nanmedian(yi))
        return yi / (yi.std() + 1e-12)

    a = prep(force["t_rel"].to_numpy(), force["force"].to_numpy())
    b = prep(kin["t_rel"].to_numpy(), kin["theta_deg"].to_numpy())
    c = np.correlate(a, b, mode="full")
    lags = (np.arange(len(c)) - (len(b) - 1)) / fs
    keep = np.abs(lags) <= search_s
    lag = float(lags[keep][np.argmax(c[keep])])
    print(f"  fine lag refinement: {lag:+.2f} s (peak corr {c[keep].max() / len(grid):.3f})")
    return lag


def sync_trial(name: str) -> pd.DataFrame:
    """Load, coarse-align (true EST) and fine-align (cross-correlation) one
    level/side trial; return a shared-clock (t_rel, force, theta_deg,
    delta_tip_mm) frame restricted to the tracking recording's span.
    """
    track_folder = TRACK_DIR / TRIALS[name]
    kin = compute_theta_from_tracking(track_folder)
    force = load_force(FORCE_DIR / f"{name}.csv")

    t0 = min(kin["t_true_est"].min(), force["t_true_est"].min())
    kin["t_rel"] = (kin["t_true_est"] - t0).dt.total_seconds()
    force["t_rel"] = (force["t_true_est"] - t0).dt.total_seconds()

    lag = find_lag(force, kin)
    force = force.copy()
    # find_lag returns the lag where corr(force(t), theta(t-lag)) is maximal,
    # so we shift force by -lag to align force and theta on the same time base.
    force["t_rel"] = force["t_rel"] - lag

    d = kin[["t_rel", "theta_deg"]].copy()
    d["force"] = np.interp(d["t_rel"], force["t_rel"], force["force"],
                            left=np.nan, right=np.nan)
    d = d[d["force"].notna()].reset_index(drop=True)

    theta0 = d.loc[d["force"] < 0.15 * d["force"].max(), "theta_deg"].median()
    d["theta_dist"] = d["theta_deg"] - theta0
    d["delta_tip_mm"] = theta_deg_to_tip_mm(d["theta_dist"])
    return d


# --------------------------------------------------------------------------- #
# tracking dropouts
# --------------------------------------------------------------------------- #
TRACK_GAP_FRAMES = 3.0
TRACK_GAP_MIN_SHADE_S = 1.0


def tracking_gaps(t: np.ndarray, max_gap_frames: float = TRACK_GAP_FRAMES
                  ) -> list[tuple[float, float]]:
    """(start, end) of every span with no tracked sample.

    Dropped frames are missing rows rather than NaNs, so only the spacing
    between consecutive timestamps reveals them.
    """
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return []
    d = np.diff(t)
    dt = float(np.nanmedian(d))
    if not np.isfinite(dt) or dt <= 0:
        return []
    return [(float(t[i]), float(t[i + 1])) for i in np.flatnonzero(d > max_gap_frames * dt)]


def break_tracking_gaps(t: np.ndarray, y: np.ndarray,
                        max_gap_frames: float = TRACK_GAP_FRAMES
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN inside dropouts so a plot shows a hole instead of a straight
    line bridging it, which reads as a genuine flat measurement."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return t, y
    d = np.diff(t)
    dt = float(np.nanmedian(d))
    if not np.isfinite(dt) or dt <= 0:
        return t, y
    idx = np.flatnonzero(d > max_gap_frames * dt)
    if idx.size == 0:
        return t, y
    return np.insert(t, idx + 1, t[idx] + dt), np.insert(y, idx + 1, np.nan)


def shade_tracking_gaps(ax, t: np.ndarray, min_shade_s: float = TRACK_GAP_MIN_SHADE_S):
    """Grey out dropout spans long enough to matter; returns a legend handle."""
    handle = None
    for a, b in tracking_gaps(t):
        if b - a < min_shade_s:
            continue
        handle = ax.axvspan(a, b, color="0.85", lw=0, zorder=0, label="no theta tracking")
    return handle


# --------------------------------------------------------------------------- #
# cycle segmentation + linear-region stiffness (global regression method,
# no fixed force/angle window - same logic for every level/side).
# --------------------------------------------------------------------------- #
CYCLE_THRESH = 0.20  # fraction of trial-peak force that counts as "loaded"


def segment_cycles(force: np.ndarray, thresh: float, min_len: int = 10
                    ) -> list[tuple[int, int]]:
    """(start, end) index pairs of each loaded run."""
    loaded = np.nan_to_num(force) > thresh
    edge = np.diff(loaded.astype(int))
    starts = np.flatnonzero(edge == 1) + 1
    ends = np.flatnonzero(edge == -1) + 1
    if loaded[0]:
        starts = np.r_[0, starts]
    if loaded[-1]:
        ends = np.r_[ends, len(force)]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def ramp_bounds(f, s, e, fmax, max_back=60):
    """(start, end, peak) index of the distraction ramp within a loaded run."""
    pk = int(s + np.nanargmax(f[s:e]))
    lim = max(0, s - max_back)
    i0 = lim + int(np.nanargmin(f[lim:s + 1]))
    if pk - i0 < 10:
        return None
    return i0, pk + 1, pk


def linear_region(x, y, min_pts=12, r2_min=0.90):
    """Best-conditioned straight sub-window of a ramp (global regression
    method: scores every sub-window on shape, not on any fixed x/y window).
    """
    n = len(x)
    if n < min_pts:
        return None
    z = np.zeros(1)
    cx, cy = np.r_[z, np.cumsum(x)], np.r_[z, np.cumsum(y)]
    cxx, cyy = np.r_[z, np.cumsum(x * x)], np.r_[z, np.cumsum(y * y)]
    cxy = np.r_[z, np.cumsum(x * y)]

    i = np.arange(n)[:, None]
    j = np.arange(n + 1)[None, :]
    m = (j - i).astype(float)
    with np.errstate(all="ignore"):
        sx, sy = cx[j] - cx[i], cy[j] - cy[i]
        sxx = cxx[j] - cxx[i] - sx * sx / m
        syy = cyy[j] - cyy[i] - sy * sy / m
        sxy = cxy[j] - cxy[i] - sx * sy / m
        r2 = sxy * sxy / (sxx * syy)
        se = np.sqrt((syy - sxy * sxy / sxx) / ((m - 2) * sxx))
    ok = (m >= min_pts) & (sxy > 0) & np.isfinite(se) & (r2 >= r2_min)
    if not ok.any():
        return None
    a, b = np.unravel_index(np.argmin(np.where(ok, se, np.inf)), se.shape)
    return int(a), int(b)


def fit_cycles(d: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle stiffness (N/mm and N.m/deg) from each ramp's own best
    straight sub-window - the same global linear-region method for every
    level/side, no per-cycle or per-intervention tuning.
    """
    f = d["force"].to_numpy()
    tip = d["delta_tip_mm"].to_numpy()
    th = d["theta_dist"].to_numpy()
    fmax = float(np.nanmax(f))
    runs = segment_cycles(f, CYCLE_THRESH * fmax)

    rows = []
    for n, (s, e) in enumerate(runs, 1):
        rb = ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, pk = rb
        ramp = np.arange(i0, i1)
        reg = linear_region(tip[ramp], f[ramp])
        if reg is None:
            continue
        lin = ramp[reg[0]:reg[1]]
        if np.ptp(tip[lin]) < 0.05 or np.ptp(th[lin]) < 0.30:
            continue
        k_tip, b = np.polyfit(tip[lin], f[lin], 1)
        k_rot = k_tip * D_O ** 2 * np.pi / 180.0 / 1000.0
        rows.append({
            "cycle": n,
            "F_peak_N": round(float(f[pk]), 1),
            "theta_lo_deg": round(float(th[lin].min()), 3),
            "theta_hi_deg": round(float(th[lin].max()), 3),
            "n_pts": len(lin),
            "k_tip_N_per_mm": round(float(k_tip), 3),
            "k_rot_Nm_per_deg": round(float(k_rot), 4),
            "R2": round(float(np.corrcoef(tip[lin], f[lin])[0, 1] ** 2), 3),
        })
    return pd.DataFrame(rows)


def analyze_trial(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline for one level/side: sync, then per-cycle stiffness."""
    d = sync_trial(name)
    cycles = fit_cycles(d)
    print(f"  {name}: {len(cycles)} cycles fit, median k_rot = "
          f"{cycles['k_rot_Nm_per_deg'].median():.3f} N.m/deg"
          if len(cycles) else f"  {name}: 0 cycles fit")
    return d, cycles
