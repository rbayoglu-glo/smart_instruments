#%%
"""Stacked true-raw force/theta sync view for all level/side trials.

SH-pose-only variant:
- Force: DAQ channel with direction correction only (-1 * force), no baseline/preload subtraction
- Theta: raw theta_deg from fifth-marker selection using only type-1 SH array poses
- Time alignment: wall-clock alignment + lag refinement from force/theta cross-correlation

Unlike the default intact solver, this script does not fit array pose from strays.
It only uses timestamps where SH tool pose is available for the target array UUID.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lamina_spreader_lr import (
    CAD_FIFTH_CLOSED,
    CAD_MARKERS,
    CAD_PIVOT,
    CIRCLE_TOL_MM,
    FORCE_COL,
    FORCE_DIR,
    FORCE_SIGN,
    MAX_STEP_MM_PER_S,
    OUT_DIR,
    PATTERN_MARKERS,
    TRACK_DIR,
    TRIALS,
    break_tracking_gaps,
    find_lag,
    kabsch,
    load_recording_meta,
    parse_force_anchor,
    tracking_gaps,
)


FORCE_YLIM_N = (-25.0, 150.0)
THETA_YLIM_DEG = (-4.0, 7.0)
ARRAY_UUID = "d80d3bb5-9174-4bf3-811e-e31423da384a"


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w,x,y,z) to right-handed 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _parse_ndi_localizer(track_folder: Path) -> tuple[dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Return type-0 strays and type-1 SH pose (for ARRAY_UUID), keyed by timestamp."""
    ndi_root = track_folder / "ndi"
    tool_dir = next(ndi_root.iterdir())
    path = tool_dir / "NDI_Localizer.txt"

    strays: dict[int, np.ndarray] = {}
    sh_pose: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    with path.open(errors="ignore") as fh:
        for line in fh:
            if line.startswith("//"):
                continue
            f = [s.strip() for s in line.split(",")]
            if len(f) < 3:
                continue
            ts = int(f[0])
            kind = f[1]

            if kind == "0":
                if len(f) < 6:
                    continue
                cnt = int(f[4])
                if cnt:
                    strays[ts] = np.asarray(f[5:5 + 3 * cnt], float).reshape(cnt, 3)
                else:
                    strays[ts] = np.empty((0, 3), float)

            elif kind == "1":
                n_tools = int(f[2])
                base = 3
                stride = 9
                for i in range(n_tools):
                    j = base + i * stride
                    if j + 8 >= len(f):
                        break
                    uuid = f[j]
                    if uuid != ARRAY_UUID:
                        continue

                    x = float(f[j + 2])
                    y = float(f[j + 3])
                    z = float(f[j + 4])
                    qw = float(f[j + 5])
                    qx = float(f[j + 6])
                    qy = float(f[j + 7])
                    qz = float(f[j + 8])

                    r_wd = _quat_to_rot(qw, qx, qy, qz)
                    t_wd = np.array([x, y, z], dtype=float)
                    sh_pose[ts] = (r_wd, t_wd)
                    break

    return strays, sh_pose


def compute_theta_from_sh_pose(track_folder: Path, valid_only: bool = False) -> pd.DataFrame:
    """Per-frame fifth-marker theta using SH tool pose only (no array fitting)."""
    r_cad, t_cad = kabsch(PATTERN_MARKERS, CAD_MARKERS)
    v0 = CAD_FIFTH_CLOSED[:2] - CAD_PIVOT[:2]
    r0 = float(np.linalg.norm(v0))

    strays, sh_pose = _parse_ndi_localizer(track_folder)
    meta = load_recording_meta(track_folder)

    rows = []
    n_frames = len(strays)
    n_with_sh = 0

    last_cad = None
    last_ts = None

    for ts in sorted(strays.keys()):
        pts = strays[ts]
        pose = sh_pose.get(ts)
        if pose is None:
            continue
        n_with_sh += 1
        if len(pts) == 0:
            continue

        r_wd, t_wd = pose

        # world -> pattern frame from SH pose directly (no fitted array pose)
        local = (pts - t_wd) @ r_wd
        cad = local @ r_cad.T + t_cad

        dr = np.linalg.norm(cad[:, :2] - CAD_PIVOT[:2], axis=1) - r0
        err = np.hypot(dr, cad[:, 2] - CAD_FIFTH_CLOSED[2])
        keep = err < CIRCLE_TOL_MM
        if keep.any():
            cand, cerr, circle_ok = cad[keep], err[keep], True
        else:
            cand, cerr, circle_ok = cad, err, False

        if last_cad is None:
            j = int(np.argmin(cerr))
            step_ok = True
        else:
            step = np.linalg.norm(cand - last_cad, axis=1)
            j = int(np.argmin(step))
            budget = MAX_STEP_MM_PER_S * min((ts - last_ts) / 1e6, 1.0)
            step_ok = step[j] <= budget

        valid = circle_ok and step_ok
        chosen = cand[j]
        if valid:
            last_cad = chosen
            last_ts = ts
        elif valid_only:
            continue

        v = chosen[:2] - CAD_PIVOT[:2]
        theta = np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(v0[1], v0[0]))
        t_true_est = meta["start_est"] + pd.to_timedelta((ts - meta["start_ts_us"]) / 1e6, unit="s")

        rows.append(
            {
                "ts_us": ts,
                "theta_deg": float(theta),
                "marker_cad_x_mm": float(chosen[0]),
                "marker_cad_y_mm": float(chosen[1]),
                "marker_cad_z_mm": float(chosen[2]),
                "circle_err_mm": float(cerr[j]),
                "fit_resid_mm": np.nan,
                "t_true_est": t_true_est,
                "valid": valid,
                "circle_ok": circle_ok,
                "step_ok": step_ok,
            }
        )

    df = pd.DataFrame(rows)
    n_valid = int(df["valid"].sum()) if not df.empty else 0
    print(
        f"{track_folder.name}: {n_frames} frames with strays, "
        f"{n_with_sh} with SH pose, {n_valid} passed circle+step gates"
    )

    if valid_only and not df.empty:
        df = df.drop(columns=["valid", "circle_ok", "step_ok"])
    return df


def load_force_true_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Time", FORCE_COL])
    df["t_s"] = df["Time"] / 1e6
    df["force_raw"] = pd.to_numeric(df[FORCE_COL], errors="coerce")
    df["force_dir"] = FORCE_SIGN * df["force_raw"]

    anchor_est = parse_force_anchor(path)
    df["t_true_est"] = pd.to_datetime(anchor_est) + pd.to_timedelta(df["t_s"], unit="s")
    return df[["t_s", "force_raw", "force_dir", "t_true_est"]]


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

        theta = kin["theta_deg"].to_numpy()
        valid = kin["valid"].to_numpy()

        kin_for_lag = pd.DataFrame({"t_rel": kin_t[valid], "theta_deg": theta[valid]})
        force_for_lag = pd.DataFrame({"t_rel": force_t, "force": force["force_dir"].to_numpy()})
        lag = find_lag(force_for_lag, kin_for_lag)
        force_t_sync = force_t - lag
        theta_valid = np.where(valid, theta, np.nan)
        theta_rej = np.where(valid, np.nan, theta)

        h_force, = ax.plot(
            force_t_sync,
            force["force_dir"].to_numpy(),
            lw=1.0,
            color="tab:blue",
            label="force (-1 x raw), lag-synced",
        )
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(*FORCE_YLIM_N)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        t_ok, th_ok = break_tracking_gaps(kin_t, theta_valid)
        t_bad, th_bad = break_tracking_gaps(kin_t, theta_rej)
        h_theta, = ax2.plot(t_ok, th_ok, lw=0.95, color="tab:orange", label="theta valid (SH pose)")
        h_bad, = ax2.plot(t_bad, th_bad, lw=0.95, color="tab:red", label="theta rejected (SH pose)")
        ax2.axhline(0.0, color="tab:orange", lw=0.9, ls=":", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(*THETA_YLIM_DEG)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        n_lost = int(np.count_nonzero(~valid))
        untracked = sum(b - a for a, b in tracking_gaps(kin_t))
        ax.set_title(
            f"{name}   SH-pose-only raw theta + force-direction corrected sync check "
            f"(lag {lag:+.2f} s, {n_lost} rejected frames, {untracked:.0f} s with no SH frames)",
            fontsize=10,
        )

        if i == 0:
            handles = [h_force, h_theta, h_bad]
            ax.legend(handles, [h.get_label() for h in handles], loc="upper right", fontsize=8)

        if i == n - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        "Lamina spreader raw-theta sync check (stacked) - SH pose only\n"
        "Force direction corrected (-1 x raw) + lag refinement; theta shown as raw theta_deg (no theta0 subtraction).",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = Path(OUT_DIR) / "sync_check_all_levels_sides_stacked_true_raw_shpose_only.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
