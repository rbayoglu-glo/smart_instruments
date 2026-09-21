#%%

"""Lamina spreader lab analysis.

Test: "2 - Intact Holding for longer".

Geometry
--------
The d80d3bb5 DRB is rigidly attached to the UPPER handle of the spreader, the
fifth stray marker is rigidly attached to the LOWER handle.  Both arms cross at
the pivot, so in the DRB (upper-arm) frame the fifth marker simply rotates about
the pivot, and that rotation IS the tip distraction angle theta.

The tracker reports the DRB pose in the "pattern" frame defined in
patterns_*.json.  The CAD table supplied for the spreader uses a different frame
whose origin sits at the centre of the distal tips.  Both describe the same four
markers, so a rigid transform between the frames is recovered with Kabsch.

    delta_tip = d_o * theta_rad
    k_tip     = F / delta_tip
    k_rot     = k_tip * d_o^2 * pi/180      [N.mm/deg if F in N, d_o in mm]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
BASE = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab")
TEST_NAME = "2 - Intact Holding for longer"
FORCE_CSV = BASE / "Force_Data" / f"{TEST_NAME}.csv"
POSES_TXT = BASE / "Robot_Camera_Data" / "2 - Intact Holding Longer" / "pf" / "poses.txt"
OUT_DIR = Path(__file__).parent / "output"

FORCE_COL = "Spreader (SRC2475-LN1-02) on Ch 01.02.02 Calibrated Values"
SPREADER_UUID = "d80d3bb5-9174-4bf3-811e-e31423da384a"

# Contact-region midpoint distance from pivot [mm], used for CAD geometry.
D_O = 131.87
# Moment arm to the spreader tip [mm], used for force-to-moment conversion.
D_TIP = 139.0

# DRB markers in the tracker pattern frame (patterns_0042_*.json, "Excelsius Array #8")
PATTERN_MARKERS = np.array(
    [
        [52.6051, 34.0, 56.1315],
        [54.2956, -30.0, 52.5063],
        [94.8670, -28.0, -34.4993],
        [91.4860, 28.0, -27.2488],
    ]
)

# the same four markers in the CAD frame (origin = centre of the distal tips),
# ordered to match PATTERN_MARKERS: top-right, top-left, bottom-left, bottom-right
CAD_MARKERS = np.array(
    [
        [354.182, 93.064, 34.000],
        [350.556, 94.754, -30.000],
        [263.551, 135.326, -28.000],
        [270.801, 131.945, 28.000],
    ]
)

# fifth (lower-handle) marker in the CAD frame, spreader closed -> theta = 0
CAD_FIFTH_CLOSED = np.array([360.650, -48.000, 0.000])

# pivot in the CAD frame: on the instrument axis, d_o distal-ward from the tips
CAD_PIVOT = np.array([D_O, 0.0, 0.0])

# the load cell reads distraction as compression
FORCE_SIGN = -1.0

# a candidate stray is accepted as the fifth marker only if it sits this close to
# the circle it must sweep about the pivot
CIRCLE_TOL_MM = 6.0
# largest credible frame-to-frame travel of the fifth marker (30 Hz capture)
MAX_STEP_MM = 15.0

# a cycle is "loaded" above this fraction of the test peak force
CYCLE_THRESH = 0.20
# the linear region is the best-conditioned straight run of the ramp
LINEAR_R2 = 0.90
LINEAR_MIN_PTS = 12
# fixed tissue-angle window for stiffness, measured from tip engagement
THETA_WINDOW = (0.5, 1.25)
# tip engagement reference for cycle-local angle zero
ENGAGE_FORCE_N = 12.0
# force band used only to score candidate sync lags
LAG_BAND = (0.40, 0.90)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def force_to_moment_nm(force_n):
    """Convert distraction force [N] to tip moment [N.m]."""
    return force_n * D_TIP / 1000.0


def moment_to_force_n(moment_nm):
    """Convert tip moment [N.m] to distraction force [N]."""
    return moment_nm * 1000.0 / D_TIP


def theta_deg_to_tip_mm(theta_deg):
    """Convert distraction angle [deg] to tip displacement [mm] using D_TIP."""
    return D_TIP * np.radians(theta_deg)


def tip_mm_to_theta_deg(delta_tip_mm):
    """Convert tip displacement [mm] to distraction angle [deg] using D_TIP."""
    return np.degrees(delta_tip_mm / D_TIP)


def k_rot_to_k_linear_factor() -> float:
    """Scale factor to convert k_rot [N.m/deg] to k_linear [N/mm]."""
    return 1000.0 / (D_O**2 * np.pi / 180.0)


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform R, t with dst ~= src @ R.T + t."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cd - r @ cs


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions in (w, x, y, z) -> (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def parse_poses(path: Path, uuid: str) -> pd.DataFrame:
    """Read a pf/poses.txt log into per-frame DRB pose + stray marker cloud."""
    strays: dict[int, np.ndarray] = {}
    tool: dict[int, np.ndarray] = {}

    with path.open() as fh:
        for line in fh:
            if line.startswith("//"):
                continue
            f = [s.strip() for s in line.split(",")]
            if len(f) < 3:
                continue
            ts, kind = int(f[0]), int(f[1])

            if kind == 0:  # stray 3D points
                n = int(f[4])
                if n:
                    strays[ts] = np.asarray(f[5 : 5 + 3 * n], float).reshape(n, 3)
            else:  # tool poses
                n = int(f[2])
                for k in range(n):
                    blk = f[3 + 9 * k : 12 + 9 * k]
                    if blk[0] == uuid:
                        tool[ts] = np.asarray(blk[2:9], float)
                        break

    ts = np.array(sorted(set(tool) & set(strays)))
    pose = np.stack([tool[t] for t in ts])
    return pd.DataFrame(
        {
            "ts_us": ts,
            "pos": list(pose[:, :3]),
            "quat": list(pose[:, 3:]),
            "strays": [strays[t] for t in ts],
        }
    )


# --------------------------------------------------------------------------- #
# kinematics
# --------------------------------------------------------------------------- #
def compute_theta(poses: pd.DataFrame) -> pd.DataFrame:
    """Fifth-marker position in the CAD frame and the resulting distraction angle."""
    r_cad, t_cad = kabsch(PATTERN_MARKERS, CAD_MARKERS)
    resid = np.linalg.norm(PATTERN_MARKERS @ r_cad.T + t_cad - CAD_MARKERS, axis=1)
    print(f"pattern->CAD frame fit residual: max {resid.max():.4f} mm")

    r_wd = quat_to_mat(np.stack(poses["quat"].to_numpy()))
    p_wd = np.stack(poses["pos"].to_numpy())

    v0 = CAD_FIFTH_CLOSED[:2] - CAD_PIVOT[:2]
    r0 = float(np.linalg.norm(v0))

    out = np.full((len(poses), 3), np.nan)
    n_cand = 0
    last, last_i = None, -1000
    for i, world in enumerate(poses["strays"]):
        local = (world - p_wd[i]) @ r_wd[i]  # camera -> pattern frame
        cad = local @ r_cad.T + t_cad  # pattern -> CAD frame
        n_cand += len(cad)
        # the fifth marker must lie on the circle it sweeps about the pivot
        dr = np.linalg.norm(cad[:, :2] - CAD_PIVOT[:2], axis=1) - r0
        err = np.hypot(dr, cad[:, 2] - CAD_FIFTH_CLOSED[2])
        keep = err < CIRCLE_TOL_MM
        if not keep.any():
            continue
        cand, cerr = cad[keep], err[keep]

        if last is not None and i - last_i <= 15:  # continue the existing track
            step = np.linalg.norm(cand - last, axis=1)
            j = int(np.argmin(step))
            if step[j] > MAX_STEP_MM * (i - last_i):
                continue
        else:  # re-acquire after a dropout
            j = int(np.argmin(cerr))
        out[i], last, last_i = cand[j], cand[j], i

    v = out[:, :2] - CAD_PIVOT[:2]
    theta = np.degrees(np.arctan2(v[:, 1], v[:, 0]) - np.arctan2(v0[1], v0[0]))

    res = poses[["ts_us"]].copy()
    res["x_cad"], res["y_cad"], res["z_cad"] = out.T
    res["radius"] = np.linalg.norm(v, axis=1)
    res["theta_deg"] = theta
    res["t_s"] = (res["ts_us"] - res["ts_us"].iloc[0]) / 1e6

    # reject single-frame jumps that survive the geometric gate
    med = res["theta_deg"].rolling(9, center=True, min_periods=1).median()
    res.loc[(res["theta_deg"] - med).abs() > 0.5, "theta_deg"] = np.nan
    # light low-pass; the ramps last ~1-2 s so this keeps their shape intact
    res["theta_sm"] = (
        res["theta_deg"]
        .rolling(5, center=True, min_periods=2).median()
        .rolling(5, center=True, min_periods=2).mean()
    )

    good = res["theta_deg"].notna()
    print(f"frames: {len(res)}, mean strays/frame {n_cand / len(res):.2f}, "
          f"fifth marker accepted in {good.sum()} ({100 * good.mean():.1f}%)")
    print(
        f"pivot->marker radius: {res.loc[good, 'radius'].mean():.2f} +/- "
        f"{res.loc[good, 'radius'].std():.2f} mm (CAD closed = {r0:.2f} mm)"
    )
    print(f"fifth marker out-of-plane z: {res.loc[good, 'z_cad'].abs().max():.2f} mm")
    ctr = res.loc[good, ["x_cad", "y_cad", "z_cad"]].to_numpy().mean(0)
    print(
        f"fifth marker mean position: ({ctr[0]:.1f}, {ctr[1]:.1f}, {ctr[2]:.1f}) mm  "
        f"| CAD table ({CAD_FIFTH_CLOSED[0]:.1f}, {CAD_FIFTH_CLOSED[1]:.1f}, "
        f"{CAD_FIFTH_CLOSED[2]:.1f}) mm"
    )
    print(
        f"theta range: {res.loc[good, 'theta_deg'].min():.2f} .. "
        f"{res.loc[good, 'theta_deg'].max():.2f} deg"
    )

    # planarity: is the swing really confined to the spreader plane (CAD x-y)?
    u = out[good.to_numpy()] - CAD_PIVOT
    uref = u[0]
    ang3d = np.degrees(
        np.arccos(
            np.clip(
                u @ uref / (np.linalg.norm(u, axis=1) * np.linalg.norm(uref)), -1, 1
            )
        )
    )
    angxy = np.abs(
        res.loc[good, "theta_deg"].to_numpy() - res.loc[good, "theta_deg"].iloc[0]
    )
    dev = np.abs(ang3d - angxy)
    zsd = res.loc[good, "z_cad"].std()
    print(
        f"planarity: out-of-plane z {res.loc[good, 'z_cad'].mean():+.2f} +/- {zsd:.2f} mm "
        f"over {np.ptp(res.loc[good, 'theta_deg']) * np.pi / 180 * r0:.1f} mm in-plane "
        f"travel -> plane tilt {np.degrees(np.arctan(zsd / r0)):.2f} deg; "
        f"|3D swing - in-plane swing| median {np.median(dev):.3f}, "
        f"p95 {np.percentile(dev, 95):.3f} deg"
    )
    return res


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


def per_cycle_lag(d, runs, fmax, pad=15):
    """Residual force-vs-theta shift inside each ramp, in seconds.

    A vertical leg in the F-theta curve looks like tip seating but is also what a
    leftover clock offset produces, so this checks the two apart: a real offset
    is the same in every cycle, seating is not.
    """
    f, th = d["force"].to_numpy(), d["theta_dist"].to_numpy()
    dt = float(np.median(np.diff(d["t_sync"].to_numpy())))
    out = []
    for s, e in runs:
        rb = ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, _, pk = rb
        w = slice(max(0, i0 - pad), min(len(f), pk + pad))
        a, b = f[w], th[w]
        if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
            continue
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        c = np.correlate(a, b, "full")
        out.append((np.argmax(c) - (len(b) - 1)) * dt)
    return np.array(out)


def ramp_bounds(f, s, e, fmax, max_back=60):
    """(start, end, peak) of the distraction manoeuvre.

    Peak force marks the spreader pause: while the operator is still advancing
    the ratchet the force rises, and the moment they stop it decays through
    relaxation.  The start is the foot of that rise.  Both anchors use force,
    which is clean; theta is too noisy between cycles, where the spreader is
    being handled, to trigger on reliably.
    """
    pk = int(s + np.nanargmax(f[s:e]))
    lim = max(0, s - max_back)
    i0 = lim + int(np.nanargmin(f[lim:s + 1]))
    if pk - i0 < 10:
        return None
    return i0, pk + 1, pk


def load_force(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["t_s"] = df["Time"] / 1e6
    df["force"] = FORCE_SIGN * (df[FORCE_COL] - df[FORCE_COL].iloc[:200].median())
    print(f"force: {len(df)} samples, {df['t_s'].iloc[-1]:.1f} s, "
          f"peak {df['force'].max():.2f} N")
    return df[["t_s", "force"]]


def find_lag(force: pd.DataFrame, kin: pd.DataFrame, fs: float = 20.0) -> float:
    """Cross-correlate the two plateau-shaped signals to recover the clock offset."""
    t0 = min(force["t_s"].iloc[0], kin["t_s"].iloc[0])
    t1 = max(force["t_s"].iloc[-1], kin["t_s"].iloc[-1])
    grid = np.arange(t0, t1, 1.0 / fs)

    def prep(t, y):
        s = pd.Series(y).interpolate(limit_direction="both")
        y = np.interp(grid, t, s, left=np.nan, right=np.nan)
        y = np.nan_to_num(y - np.nanmedian(y))
        return y / (y.std() + 1e-12)

    a = prep(force["t_s"].to_numpy(), force["force"].to_numpy())
    b = prep(kin["t_s"].to_numpy(), kin["theta_deg"].to_numpy())
    c = np.correlate(a, b, mode="full")
    lags = (np.arange(len(c)) - (len(b) - 1)) / fs
    lag = float(lags[np.argmax(c)])
    print(f"estimated force->mocap lag: {lag:+.3f} s "
          f"(peak corr {c.max() / len(grid):.3f})")
    return lag


# --------------------------------------------------------------------------- #
def linear_region(x, y, min_pts=LINEAR_MIN_PTS, r2_min=LINEAR_R2):
    """Best-conditioned straight sub-window of a ramp, as slice bounds into x/y.

    Selecting on force level alone cannot tell an elastic rise from a creep
    plateau, since both can sit inside the same force band.  This scores every
    sub-window on shape instead, minimising the standard error of the fitted
    slope subject to a linearity floor.  That trades width against scatter on
    its own: plateaus and the curved toe blow up the residual, while windows
    that are merely short lose on Sxx.
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


def sync(kin: pd.DataFrame, force: pd.DataFrame, lag: float):
    """Interpolate the load cell onto the mocap clock at `lag`."""
    k = kin.copy()
    k["t_sync"] = k["t_s"] + lag
    k["force"] = np.interp(
        k["t_sync"], force["t_s"], force["force"], left=np.nan, right=np.nan
    )
    fmax = float(force["force"].max())
    theta0 = k.loc[k["force"] < 0.15 * fmax, "theta_sm"].median()
    k["theta_dist"] = k["theta_sm"] - theta0
    k["delta_tip"] = theta_deg_to_tip_mm(k["theta_dist"])
    k["moment"] = force_to_moment_nm(k["force"])  # N.m about the pivot
    d = k[k["force"].notna() & k["theta_dist"].notna()].reset_index(drop=True)
    return d, theta0, fmax


def lag_score(d: pd.DataFrame, cycles) -> float:
    """Mean linearity of the mid-force band, used only to tune the sync lag."""
    f, tip = d["force"].to_numpy(), d["delta_tip"].to_numpy()
    r2 = []
    for s, e in cycles:
        pk = int(s + np.nanargmax(f[s:e]))
        seg = np.arange(s, pk + 1)
        band = seg[(f[seg] > LAG_BAND[0] * f[pk]) & (f[seg] < LAG_BAND[1] * f[pk])]
        if len(band) >= 8 and np.ptp(tip[band]) > 0.1:
            r2.append(np.corrcoef(tip[band], f[band])[0, 1] ** 2)
    return float(np.mean(r2)) if len(r2) >= 5 else -np.inf


def fit_cycles(d: pd.DataFrame, runs, fmax: float):
    """Per cycle: isolate the distraction manoeuvre and fit its linear region.

    Two fits are reported.  `k_rot` uses the fixed tissue-angle window, which is
    comparable across cycles and interventions.  `auto_lo/auto_hi` report where
    the data itself stays straight, as a check on that window.
    """
    f = d["force"].to_numpy()
    tip = d["delta_tip"].to_numpy()
    th = d["theta_dist"].to_numpy()
    t = d["t_sync"].to_numpy()
    k_linear_factor = k_rot_to_k_linear_factor()

    rows, segs = [], []
    for n, (s, e) in enumerate(runs, 1):
        rb = ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, pk = rb
        ramp = np.arange(i0, i1)

        f_eng = min(ENGAGE_FORCE_N, 0.60 * f[pk])
        hit = ramp[f[ramp] >= f_eng]
        if len(hit) == 0:
            continue
        iref = int(hit[0])
        th_ref = th[iref]
        thr = th - th_ref

        win = ramp[(thr[ramp] >= THETA_WINDOW[0]) & (thr[ramp] <= THETA_WINDOW[1])]
        if len(win) < 5 or np.ptp(tip[win]) < 0.05:
            continue
        k_tip, b = np.polyfit(tip[win], f[win], 1)

        reg = linear_region(tip[ramp], f[ramp])
        lin = ramp[reg[0]:reg[1]] if reg else np.array([], int)
        rows.append(
            {
                "cycle": n,
                "F_peak_N": round(f[pk], 1),
                "F_start_N": round(f[i0], 1),
                "F_engage_N": round(f_eng, 1),
                "theta_max_deg": round(np.nanmax(thr[ramp]), 2),
                "ramp_s": round(t[i1 - 1] - t[i0], 1),
                "k_rot_Nm_deg": round(k_tip / k_linear_factor, 3),
                "R2": round(np.corrcoef(tip[win], f[win])[0, 1] ** 2, 3),
                "n": len(win),
                "auto_lo": round(thr[lin].min(), 2) if len(lin) else np.nan,
                "auto_hi": round(thr[lin].max(), 2) if len(lin) else np.nan,
                "auto_k_rot": (
                      round(np.polyfit(tip[lin], f[lin], 1)[0] / k_linear_factor, 3)
                      if len(lin)
                      else np.nan
                ),
            }
        )
        segs.append({"ramp": ramp, "win": win, "lin": lin, "pk": pk,
                 "th_ref": th_ref, "k_tip": k_tip, "b": b, "cycle": n,
                 "f_eng": f_eng})
    return pd.DataFrame(rows), segs


def plot_ramp_grid(d, segs, tab, out: Path) -> None:
    """Force vs angle for each distraction manoeuvre, on a shared scale."""
    f, th, tip = (d["force"].to_numpy(), d["theta_dist"].to_numpy(),
                  d["delta_tip"].to_numpy())
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True, sharey=True)
    for a, sg, (_, row) in zip(axes.ravel(), segs, tab.iterrows()):
        x = th[sg["ramp"]] - sg["th_ref"]
        a.axvspan(*THETA_WINDOW, color="tab:green", alpha=0.12)
        a.axvline(0.0, color="0.6", lw=0.8, ls=":")
        a.plot(x, f[sg["ramp"]], "-o", ms=2.2, lw=0.9, color="0.55",
               label="ramp to spreader pause")
        a.plot(th[sg["win"]] - sg["th_ref"], f[sg["win"]], "o", ms=3.5,
               color="tab:green", label=f"{THETA_WINDOW[0]}-{THETA_WINDOW[1]}$\\degree$")
        if len(sg["lin"]):
            a.plot(th[sg["lin"]] - sg["th_ref"], f[sg["lin"]], lw=2.0,
                   color="tab:red", alpha=0.7, label="auto straight run")
        xs = np.linspace(*THETA_WINDOW, 10)
        a.plot(xs, sg["k_tip"] * theta_deg_to_tip_mm(xs + sg["th_ref"]) + sg["b"], "k--",
               lw=1.0)
        a.set_title(f"cycle {sg['cycle']}  |  $k_{{rot}}$="
                    f"{row['k_rot_Nm_deg']:.2f} N\u00b7m/deg, "
                    f"$F_{{pk}}$={row['F_peak_N']:.0f} N", fontsize=9)
        a.grid(alpha=0.3)
    for a in axes[-1]:
        a.set_xlabel(r"distraction angle from tip engagement $\theta$ [deg]")
    for a in axes[:, 0]:
        a.set_ylabel("distraction force [N]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{TEST_NAME}  -  loading ramp of each cycle "
                 f"(shaded = {THETA_WINDOW[0]}-{THETA_WINDOW[1]} deg from tip engagement)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def _panels(d):
    """(x, y, xlabel, ylabel, title) for the three views of the same ramp fit."""
    f = d["force"].to_numpy()
    return [
        (d["delta_tip"].to_numpy(), f, r"tip displacement $\delta_{tip}$ [mm]",
         "distraction force [N]", "force vs tip displacement"),
        (d["theta_dist"].to_numpy(), f, r"distraction angle $\theta$ [deg]",
         "distraction force [N]", "force vs angle"),
        (d["theta_dist"].to_numpy(), d["moment"].to_numpy(),
         r"distraction angle $\theta$ [deg]",
         r"pivot moment $M = F\,d_o$ [N$\cdot$m]", "moment vs angle"),
    ]


def _fit_line(p, lin, k_tip, b, tip):
    """Map the force-displacement fit of one cycle into panel `p`."""
    xs = np.linspace(tip[lin].min(), tip[lin].max(), 10)
    yf = k_tip * xs + b
    return (xs if p == 0 else tip_mm_to_theta_deg(xs),
            force_to_moment_nm(yf) if p == 2 else yf)


def plot_cycles(d, segs, tab, out: Path) -> None:
    """All loading ramps overlaid, in the three views."""
    tip = d["delta_tip"].to_numpy()
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(segs)))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for p, (a, (x, y, xl, yl, ti)) in enumerate(zip(ax, _panels(d))):
        for c, sg, (_, row) in zip(colors, segs, tab.iterrows()):
            a.plot(x[sg["ramp"]], y[sg["ramp"]], lw=0.9, color=c, alpha=0.30)
            a.plot(x[sg["win"]], y[sg["win"]], lw=2.2, color=c,
                   label=f"{sg['cycle']}: {row['k_rot_Nm_deg']:.2f}")
            a.plot(*_fit_line(p, sg["win"], sg["k_tip"], sg["b"], tip), "k--", lw=0.9)
        a.set(xlabel=xl, ylabel=yl, title=ti)
        a.grid(alpha=0.3)
    ax[2].legend(title=r"cycle: $k_{rot}$ [N$\cdot$m/deg]", fontsize=7,
                 title_fontsize=7, loc="upper left", ncol=2)
    fig.suptitle(f"{TEST_NAME}  -  {len(segs)} loading cycles "
                 f"(bold = linear region used for the fit)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_cycle_grid(d, segs, tab, out_dir: Path) -> None:
    """One 3x3 grid per view, so each cycle can be inspected on its own."""
    tip = d["delta_tip"].to_numpy()
    names = ["force_vs_tip", "force_vs_theta", "moment_vs_theta"]

    for p, (x, y, xl, yl, ti) in enumerate(_panels(d)):
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
        for a, sg, (_, row) in zip(axes.ravel(), segs, tab.iterrows()):
            a.plot(x[sg["ramp"]], y[sg["ramp"]], lw=1.0, color="0.6", label="ramp")
            a.plot(x[sg["win"]], y[sg["win"]], lw=2.4, color="tab:blue",
                   label="linear region")
            a.plot(*_fit_line(p, sg["win"], sg["k_tip"], sg["b"], tip), "k--",
                   lw=1.0, label="fit")
            a.set_title(
                f"cycle {sg['cycle']}  |  $k_{{rot}}$="
                f"{row['k_rot_Nm_deg']:.2f} N\u00b7m/deg, $R^2$={row['R2']:.2f}",
                fontsize=9,
            )
            a.grid(alpha=0.3)
        for a in axes[-1]:
            a.set_xlabel(xl)
        for a in axes[:, 0]:
            a.set_ylabel(yl)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{TEST_NAME}  -  {ti}, per loading cycle")
        fig.tight_layout()
        path = out_dir / f"cycles_{names[p]}.png"
        fig.savefig(path, dpi=150)
        print(f"saved {path}")


def refine_lag(kin, force, lag0: float, span: float = 3.0, step: float = 1 / 30):
    """Sweep the lag and keep the value that makes the ramps most linear.

    The coarse cross-correlation only resolves the plateaus; the ramps last a
    couple of seconds, so a few tenths of a second of residual offset dominates
    the fitted slope.
    """
    best, scores = (lag0, -np.inf), []
    for lag in np.arange(lag0 - span, lag0 + span + step, step):
        d, _, fmax = sync(kin, force, float(lag))
        s = lag_score(d, segment_cycles(d["force"].to_numpy(), CYCLE_THRESH * fmax))
        scores.append((lag, s))
        if s > best[1]:
            best = (float(lag), s)
    print(f"refined lag: {best[0]:+.3f} s (mean band R2 {best[1]:.3f}, "
          f"was {lag0:+.3f} s)")
    return best[0], np.array(scores)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    kin = compute_theta(parse_poses(POSES_TXT, SPREADER_UUID))
    force = load_force(FORCE_CSV)
    lag, scores = refine_lag(kin, force, find_lag(force, kin))

    d, theta0, fmax = sync(kin, force, lag)
    runs = segment_cycles(d["force"].to_numpy(), CYCLE_THRESH * fmax)
    tab, segs = fit_cycles(d, runs, fmax)
    pcl = per_cycle_lag(d, runs, fmax)
    print(f"per-cycle residual F-theta shift [s]: {np.round(pcl, 2)}  "
          f"median {np.median(pcl):+.2f}")
    print(f"unloaded theta baseline (vs CAD table pose): {theta0:+.2f} deg")
    print(f"\n{len(segs)} cycles; k_rot fit over theta = "
          f"{THETA_WINDOW[0]}-{THETA_WINDOW[1]} deg from tip engagement at "
          f"{ENGAGE_FORCE_N:.1f} N (capped at 60% of each cycle peak); "
          f"auto_* = where the data itself stays straight")
    print(tab.to_string(index=False))
    k_rot = tab["k_rot_Nm_deg"]
    print(f"\nk_rot = {k_rot.mean():.3f} +/- {k_rot.std():.3f} N.m/deg "
          f"(median {k_rot.median():.3f}, n={len(k_rot)})")
    print(f"k_rot vs peak force:   r = "
          f"{np.corrcoef(tab['F_peak_N'], k_rot)[0, 1]:.2f}")
    print(f"auto straight run spans theta "
          f"{tab['auto_lo'].median():.2f} - {tab['auto_hi'].median():.2f} deg "
          f"(median across cycles)")

    plot_ramp_grid(d, segs, tab, OUT_DIR / "cycles_ramp_force_vs_theta.png")
    plot_cycles(d, segs, tab, OUT_DIR / "cycles_force_moment.png")
    plot_cycle_grid(d, segs, tab, OUT_DIR)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].plot(force["t_s"], force["force"], lw=0.8)
    for sg in segs:
        ax[0].axvspan(d["t_sync"][sg["ramp"][0]], d["t_sync"][sg["ramp"][-1]],
                      color="tab:red", alpha=0.25)
    ax[0].set(xlabel="time [s]", ylabel="distraction force [N]",
              title="load cell (ramps shaded)")

    ax[1].plot(d["t_sync"], th, lw=0.8, color="tab:orange")
    ax[1].set(xlabel="time [s]", ylabel=r"$\theta$ [deg]", title="distraction angle")

    ax[2].plot(th, f, lw=0.4, color="0.8", zorder=1)
    for sg in segs:
        ax[2].plot(th[sg["win"]], f[sg["win"]], lw=2, zorder=2)
        i0 = int(sg["ramp"][0])
        ax[2].scatter(
            [th[i0]],
            [f[i0]],
            s=36,
            marker="o",
            facecolor="white",
            edgecolor="tab:red",
            linewidth=1.2,
            zorder=4,
        )
    ax[2].set(xlabel=r"$\theta$ [deg]", ylabel="distraction force [N]",
              title=f"force vs angle  ($k_{{rot}}$ = {k_rot.mean():.2f} "
                    f"$\\pm$ {k_rot.std():.2f} N\u00b7m/deg)")

    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(TEST_NAME)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "force_vs_theta.png", dpi=150)
    print(f"saved {OUT_DIR / 'force_vs_theta.png'}")

    # sync check: the two channels overlaid, plus the lag sensitivity sweep
    fig2, bx = plt.subplots(3, 1, figsize=(14, 10))
    for a, (lo, hi) in zip(bx, [(0, 340), (25, 40)]):
        a.plot(force["t_s"], force["force"], lw=0.8, color="tab:blue")
        start_idx = np.array([int(sg["ramp"][0]) for sg in segs], dtype=int)
        t0 = d["t_sync"].to_numpy()[start_idx]
        f0 = f[start_idx]
        th0 = th[start_idx]
        a.scatter(
            t0,
            f0,
            s=34,
            marker="o",
            facecolor="white",
            edgecolor="tab:red",
            linewidth=1.1,
            zorder=4,
            label="ramp start",
        )
        a.set(xlabel="time [s]", ylabel="distraction force [N]", xlim=(lo, hi))
        a2 = a.twinx()
        a2.plot(d["t_sync"], th, lw=0.9, color="tab:orange")
        a2.scatter(
            t0,
            th0,
            s=28,
            marker="D",
            facecolor="white",
            edgecolor="tab:red",
            linewidth=0.9,
            zorder=4,
        )
        a2.set_ylabel(r"$\theta$ [deg]", color="tab:orange")
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    bx[0].set_title(f"{TEST_NAME}  -  sync check (lag {lag:+.2f} s)")
    bx[2].plot(scores[:, 0], scores[:, 1], lw=1)
    bx[2].axvline(lag, color="tab:red", ls="--")
    bx[2].set(xlabel="lag [s]", ylabel="mean ramp $R^2$", title="lag sensitivity")
    bx[2].grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "sync_check.png", dpi=150)
    print(f"saved {OUT_DIR / 'sync_check.png'}")
    plt.show()


if __name__ == "__main__":
    main()
