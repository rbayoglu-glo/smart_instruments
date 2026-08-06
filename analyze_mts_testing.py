#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Optional legacy preview (disabled by default to keep runtime focused on theta/force analysis).
ENABLE_LEGACY_EXCEL_PREVIEW = 'Yes'
if ENABLE_LEGACY_EXCEL_PREVIEW:
    file_path = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Research_Team_Data\L231147_L5-S1\6DOF\Book4.xlsx"
    df = pd.read_excel(file_path)
    print(df.head())

#%
# Plot stiffness data: Mx (x-axis) vs Angle (y-axis) for flexion
fig, ax = plt.subplots(figsize=(8, 6))

# ax.plot(df['Mx (Nm)'], df['Angle (deg)'], marker='o', markersize=3, linewidth=1.5)
ax.plot(df.iloc[612:-1, 5], df.iloc[612:-1, 1], marker='o', markersize=3, linewidth=1.5)
ax.set_xlabel('Mx (Nm)', fontsize=12)
ax.set_ylabel('Angle (°)', fontsize=12)
ax.set_title('Flexion Stiffness - L231147 L5-S1', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.axhline(0, color='k', linewidth=0.8)
ax.axvline(0, color='k', linewidth=0.8)

plt.tight_layout()
plt.show()

# #%
# # Lateral Bending stiffness: My vs Angle
# file_path_lb = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Research_Team_Data\L231147_L5-S1\6DOF\Book5.xlsx"
# df_lb = pd.read_excel(file_path_lb)

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df_lb.iloc[395:-1, 6], df_lb.iloc[395:-1, 1], marker='o', markersize=3, linewidth=1.5)
# ax.set_xlabel('My (Nm)', fontsize=12)
# ax.set_ylabel('Angle (°)', fontsize=12)
# ax.set_title('Lateral Bending Stiffness - L231147 L5-S1', fontsize=14)
# ax.grid(True, linestyle='--', alpha=0.6)
# ax.axhline(0, color='k', linewidth=0.8)
# ax.axvline(0, color='k', linewidth=0.8)

# plt.tight_layout()
# plt.show()

# #%
# # Axial Rotation stiffness: Mz vs Angle
# file_path_ar = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Research_Team_Data\L231147_L5-S1\6DOF\Book6.xlsx"
# df_ar = pd.read_excel(file_path_ar)

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df_ar.iloc[174:-1, 7], df_ar.iloc[174:-1, 1], marker='o', markersize=3, linewidth=1.5)
# ax.set_xlabel('Mz (Nm)', fontsize=12)
# ax.set_ylabel('Angle (°)', fontsize=12)
# ax.set_title('Axial Rotation Stiffness - L231147 L5-S1', fontsize=14)
# ax.grid(True, linestyle='--', alpha=0.6)
# ax.axhline(0, color='k', linewidth=0.8)
# ax.axvline(0, color='k', linewidth=0.8)

# plt.tight_layout()
# plt.show()

# #%

# # fpath_spr = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Force_Data\1 - Intact.csv"
# fpath_spr = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Force_Data\2 - Intact Holding for longer.csv"
# # fpath_spr = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Force_Data\3 - PUBF Left.csv"
# # fpath_spr = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Force_Data\4 - FUF Left.csv"
# # fpath_spr = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Force_Data\5 - FBF.csv"
# # fpath_spr = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Force_Data\6 - Posterior Release.csv"
# # fpath_spr = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Force_Data\7 - SPO.csv"

# df_spr    = pd.read_csv(fpath_spr)

# # Time column is in microseconds (10,000 us step = 0.01 s at 100 Hz).
# time_us = pd.to_numeric(df_spr['Time'], errors='coerce')
# time_s = (time_us - time_us.iloc[0]) / 1000000

# #%

# fig, ax = plt.subplots(figsize=(8, 6))
# # ax.plot(time_s[2500:3490], df_spr.iloc[2500:3490, 2], marker='o', markersize=3, linewidth=1.5)
# # ax.plot(time_s[0:3490], df_spr.iloc[0:3490, 2], marker='o', markersize=3, linewidth=1.5)
# ax.plot(time_s, df_spr.iloc[:, 2], marker='o', markersize=3, linewidth=1.5)
# ax.set_xlabel('Time (s)', fontsize=12)
# ax.set_ylabel('Force (N)', fontsize=12)
# # ax.set_title('Lamina Spreader Force - Intact', fontsize=14)
# ax.grid(True, linestyle='--', alpha=0.6)
# ax.axhline(0, color='k', linewidth=0.8)
# ax.axvline(0, color='k', linewidth=0.8)

# plt.tight_layout()
# plt.show()

#%%
#%
# ---------------------------------------------------------------------------
# Verify DRB + fifth-marker tracking, then compute theta and delta_tip
# ---------------------------------------------------------------------------
# Use the 4-marker DRB on the upper handle + the fifth stray marker on the
# other handle. Theta is computed in the spreader plane from the signed
# relative angle (bounded to [-180, 180]) instead of unwrap-based accumulation.
import json

BASE = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab"
# Trial configuration.
# Requirement: when analyzing intact, use only "2 - Intact Holding Longer" for both
# calibration and test analysis (no cross-trial calibration from "1 - Intact").
TRIAL_PROFILES = {
    "intact": {
        "calibration_folder": r"Robot_Camera_Data\2 - Intact Holding Longer",
        "camera_folder": r"Robot_Camera_Data\2 - Intact Holding Longer",
        "force_csv": r"Force_Data\2 - Intact Holding for longer.csv",
        "intervention_file_name": "2 - Intact Holding Longer",
    },
    "pubf_left": {
        "calibration_folder": r"Robot_Camera_Data\3 - PUBF Left",
        "camera_folder": r"Robot_Camera_Data\3 - PUBF Left",
        "force_csv": r"Force_Data\3 - PUBF Left.csv",
        "intervention_file_name": "3 - PUBF Left",
    },
}
ACTIVE_TRIAL = os.getenv("ANALYSIS_TRIAL", "intact").strip().lower()
if ACTIVE_TRIAL not in TRIAL_PROFILES:
    raise ValueError(f"unknown ANALYSIS_TRIAL='{ACTIVE_TRIAL}'. Valid options: {sorted(TRIAL_PROFILES)}")

CALIBRATION_FOLDER = TRIAL_PROFILES[ACTIVE_TRIAL]["calibration_folder"]
CAMERA_FOLDER = TRIAL_PROFILES[ACTIVE_TRIAL]["camera_folder"]
FORCE_CSV = TRIAL_PROFILES[ACTIVE_TRIAL]["force_csv"]
INTERVENTION_FILE_NAME = TRIAL_PROFILES[ACTIVE_TRIAL]["intervention_file_name"]
HANDLE_DRB_UUID = "d80d3bb5-9174-4bf3-811e-e31423da384a"    # Excelsius Array #8 on upper handle
REFERENCE_DRB_UUID = "e6d6d324-1d7c-4f3b-a3a7-77dba6223f70"  # Spine DRB Array near load cell (used to cull known markers)
do_mm = 139   # pivot-to-tip lever arm (mm)
TOOL_MARKER_EXCLUSION_MM = 8.0
MAX_FIFTH_MARKER_JUMP_MM = 20.0
THETA_MAX_DEG = 40.0
TARGET_THETA_TYPICAL_DEG = 5.0
USE_TYPICAL_TARGET = False
TYPICAL_TARGET_WEIGHT = 0.2
MIN_THETA_STD_DEG = 2.0
THETA_STD_FLOOR_WEIGHT = 4.0
RADIUS_MEAN_ERR_WEIGHT = 0.20
RADIUS_STD_WEIGHT = 0.10
SEED_BUDGET = 600
COARSE_TS_STRIDE = 5
REFINE_TOP_K = 80
REFINE_TOP_K_CAP = 120
ENABLE_EXPANDED_CAP_SEARCH = True
FALLBACK_SEED_BUDGET = 2000
MIN_CAL_SPREAD_MM = 3.0
MIN_PLANARITY = 0.90
ENABLE_PLOTS = False
SAVE_PLOTS = True
PLOT_OUTPUT_DIR = os.path.join(BASE, "Data_analysis", "plot_outputs")

TOOL_NAME_BY_UUID = {
    "P9-17851": "Camera Reference",
    "e6d6d324-1d7c-4f3b-a3a7-77dba6223f70": "Spine DRB Array",
    "d80d3bb5-9174-4bf3-811e-e31423da384a": "Excelsius Array #8",
}


def tool_motion_audit(camera_folder):
    """Print translation/rotation excursions for all tracked tool UUIDs.
    This is a quick sanity check that the chosen UUID is physically plausible
    for spreader/load-cell-side motion in this trial."""
    poses_path = fr"{BASE}\{camera_folder}\pf\poses.txt"
    per_tool = {}

    with open(poses_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("//"):
                continue
            tok = [t.strip() for t in line.split(",")]
            if len(tok) < 3 or tok[1] != "2":
                continue
            count, idx = int(tok[2]), 3
            for _ in range(count):
                uuid = tok[idx]
                x, y, z = map(float, tok[idx + 2:idx + 5])
                qw, qi, qj, qk = map(float, tok[idx + 5:idx + 9])
                per_tool.setdefault(uuid, {"pos": [], "quat": []})
                per_tool[uuid]["pos"].append((x, y, z))
                per_tool[uuid]["quat"].append((qw, qi, qj, qk))
                idx += 9

    def quat_mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def quat_conj(q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    print(f"\nTool motion audit: {poses_path}")
    for uuid, values in sorted(per_tool.items(), key=lambda kv: -len(kv[1]["pos"])):
        p = np.array(values["pos"])
        q = np.array(values["quat"])
        p0 = p[0]
        q0 = q[0] / np.linalg.norm(q[0])
        trans = np.linalg.norm(p - p0, axis=1)
        rot = []
        for qi in q:
            qi = qi / np.linalg.norm(qi)
            dq = quat_mul(qi, quat_conj(q0))
            ang_deg = np.degrees(2 * np.arccos(np.clip(abs(dq[0]), -1.0, 1.0)))
            rot.append(ang_deg)
        rot = np.array(rot)
        name = TOOL_NAME_BY_UUID.get(uuid, "Unknown")
        print(
            f"  {uuid} ({name}): n={len(p)} "
            f"trans_max={np.max(trans):.2f} mm, rot_max={np.max(rot):.2f} deg"
        )


def save_or_show(fig, plot_stem):
    """Save figure to disk and optionally show interactively."""
    if SAVE_PLOTS:
        os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(PLOT_OUTPUT_DIR, f"{ACTIVE_TRIAL}_{plot_stem}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"saved plot: {out_path}")
    if ENABLE_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def load_pattern_markers(camera_folder, tool_uuids):
    """Load local marker coordinates for selected tool UUIDs from pattern JSON."""
    pattern_dir = fr"{BASE}\{camera_folder}\patterns"
    pattern_files = sorted([f for f in os.listdir(pattern_dir) if f.lower().endswith(".json")])
    if not pattern_files:
        raise RuntimeError(f"no pattern files in {pattern_dir}")
    pattern_path = os.path.join(pattern_dir, pattern_files[-1])
    with open(pattern_path, encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    markers = {}
    for tool in data.get("tools", []):
        uuid = tool.get("instrument_uuid")
        if uuid in tool_uuids:
            markers[uuid] = np.array([m["position"] for m in tool.get("markers", [])], dtype=float)
    missing = [u for u in tool_uuids if u not in markers]
    if missing:
        raise RuntimeError(f"missing tool markers in {pattern_path}: {missing}")
    return pattern_path, markers


def load_pf_streams(poses_path, tool_uuids):
    """Parse pf/poses.txt: type-0 points + selected type-2 tool poses."""
    tools = {u: {} for u in tool_uuids}
    strays = {}
    with open(poses_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("//"):
                continue
            tok = line.strip().split(",")
            if len(tok) < 3:
                continue
            ts, typ = int(tok[0]), tok[1].strip()
            if typ == "0":
                count = int(tok[4])
                pts = [tuple(map(float, tok[5 + 3 * i:8 + 3 * i])) for i in range(count)]
                strays[ts] = np.array(pts) if pts else np.zeros((0, 3))
            elif typ == "2":
                count, idx = int(tok[2]), 3
                for _ in range(count):
                    uuid = tok[idx].strip()
                    if uuid in tools:
                        x, y, z = map(float, tok[idx + 2:idx + 5])
                        qw, qi, qj, qk = map(float, tok[idx + 5:idx + 9])
                        tools[uuid][ts] = (x, y, z, qw, qi, qj, qk)
                    idx += 9
    return strays, tools


def quat_to_rot(quat):
    """Quaternion (w,i,j,k) to rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def build_fifth_candidates_local(camera_folder, handle_uuid, helper_uuid):
    """Build per-frame candidate stray markers in handle DRB local coordinates."""
    pattern_path, pattern_markers = load_pattern_markers(camera_folder, (handle_uuid, helper_uuid))
    poses_path = fr"{BASE}\{camera_folder}\pf\poses.txt"
    strays, tools = load_pf_streams(poses_path, (handle_uuid, helper_uuid))
    common_ts = sorted(set(strays) & set(tools[handle_uuid]))
    if not common_ts:
        raise RuntimeError(f"no common type-0/type-2 timestamps in {poses_path}")

    candidates_local = {}
    for ts in common_ts:
        hp = np.array(tools[handle_uuid][ts][0:3])
        hq = np.array(tools[handle_uuid][ts][3:7])
        Rh = quat_to_rot(hq / np.linalg.norm(hq))

        pts = strays[ts]
        if pts.shape[0] == 0:
            candidates_local[ts] = np.zeros((0, 3))
            continue

        keep = np.ones(len(pts), dtype=bool)
        for uuid in (handle_uuid, helper_uuid):
            if ts not in tools[uuid] or uuid not in pattern_markers:
                continue
            tp = np.array(tools[uuid][ts][0:3])
            tq = np.array(tools[uuid][ts][3:7])
            Rt = quat_to_rot(tq / np.linalg.norm(tq))
            tool_markers_world = (Rt @ pattern_markers[uuid].T).T + tp
            for mw in tool_markers_world:
                d = np.linalg.norm(pts - mw, axis=1)
                j = np.argmin(d)
                if d[j] < TOOL_MARKER_EXCLUSION_MM:
                    keep[j] = False

        pw = pts[keep]
        if pw.shape[0]:
            candidates_local[ts] = (Rh.T @ (pw - hp).T).T
        else:
            candidates_local[ts] = np.zeros((0, 3))

    debug = {
        "poses_path": poses_path,
        "pattern_path": pattern_path,
        "n_common_timestamps": len(common_ts),
    }
    return np.array(common_ts), candidates_local, debug


def track_local_candidate(common_ts, candidates_local, seed_ts, seed_idx):
    """Track one local candidate through time by nearest-neighbor continuity."""
    ts_list = list(common_ts)
    seed_i = ts_list.index(seed_ts)
    tracked = {}

    estimate = candidates_local[seed_ts][seed_idx].copy()
    for ts in ts_list[seed_i:]:
        pts = candidates_local[ts]
        if pts.shape[0]:
            d = np.linalg.norm(pts - estimate, axis=1)
            j = np.argmin(d)
            if d[j] < MAX_FIFTH_MARKER_JUMP_MM:
                estimate = pts[j]
        tracked[ts] = estimate.copy()

    estimate = candidates_local[seed_ts][seed_idx].copy()
    for ts in ts_list[:seed_i][::-1]:
        pts = candidates_local[ts]
        if pts.shape[0]:
            d = np.linalg.norm(pts - estimate, axis=1)
            j = np.argmin(d)
            if d[j] < MAX_FIFTH_MARKER_JUMP_MM:
                estimate = pts[j]
        tracked[ts] = estimate.copy()

    return np.array([tracked[ts] for ts in ts_list])


def fit_circle_in_plane(points_3d):
    """Fit best plane + circle for 3D points, returning plane/circle parameters."""
    origin = points_3d.mean(axis=0)
    _, S, Vt = np.linalg.svd(points_3d - origin, full_matrices=False)
    basis = Vt[:2]
    planarity = (S[0] ** 2 + S[1] ** 2) / np.sum(S ** 2)

    p2 = (points_3d - origin) @ basis.T
    x, y = p2[:, 0], p2[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    a, b, c = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    radius = np.sqrt(c + a ** 2 + b ** 2)
    resid = np.sqrt((x - a) ** 2 + (y - b) ** 2) - radius
    resid_rms = np.sqrt(np.mean(resid ** 2))
    return origin, basis, a, b, radius, resid_rms, planarity


def choose_fifth_marker_tracks(cal_folder, test_folder, handle_uuid, helper_uuid):
    """Select a consistent fifth-marker trajectory in both recordings."""
    cal_ts, cal_candidates, dbg_cal = build_fifth_candidates_local(cal_folder, handle_uuid, helper_uuid)
    test_ts, test_candidates, dbg_test = build_fifth_candidates_local(test_folder, handle_uuid, helper_uuid)
    same_recording = os.path.normcase(cal_folder) == os.path.normcase(test_folder)

    def _downsample_ts(ts_arr, stride):
        if stride <= 1 or len(ts_arr) <= 2:
            return ts_arr
        ds = ts_arr[::stride]
        if ds[-1] != ts_arr[-1]:
            ds = np.append(ds, ts_arr[-1])
        return ds

    def _evaluate_seed(seed_ts, seed_j, cal_ts_use, test_ts_use):
        local_cal = track_local_candidate(cal_ts_use, cal_candidates, seed_ts, seed_j)
        p05 = np.percentile(local_cal, 5, axis=0)
        p95 = np.percentile(local_cal, 95, axis=0)
        spread = np.linalg.norm(p95 - p05)
        _, _, _, _, _, _, planarity_cal = fit_circle_in_plane(local_cal)
        if spread < MIN_CAL_SPREAD_MM or planarity_cal < MIN_PLANARITY:
            return None

        if same_recording:
            # Same recording for calibration/test: keep one marker identity track.
            local_test = local_cal.copy()
            test_seed_dist = 0.0
            test_seed_ts = seed_ts
            test_seed_j = seed_j
        else:
            ref_local = np.median(local_cal, axis=0)
            test_seed = None
            for ts in test_ts_use:
                pts = test_candidates[ts]
                if pts.shape[0] == 0:
                    continue
                d = np.linalg.norm(pts - ref_local, axis=1)
                j = np.argmin(d)
                candidate = (d[j], ts, j)
                if test_seed is None or candidate[0] < test_seed[0]:
                    test_seed = candidate
            if test_seed is None:
                return None

            local_test = track_local_candidate(test_ts_use, test_candidates, test_seed[1], test_seed[2])
            test_seed_dist = test_seed[0]
            test_seed_ts = test_seed[1]
            test_seed_j = test_seed[2]
        origin, basis, a, b, radius, resid_rms, planarity = fit_circle_in_plane(local_cal)
        p2_test = (local_test - origin) @ basis.T
        tx, ty = p2_test[:, 0], p2_test[:, 1]
        theta = np.degrees(np.arctan2(ty - b, tx - a))
        theta_ref = np.degrees(np.arctan2(np.mean(np.sin(np.radians(theta))), np.mean(np.cos(np.radians(theta)))))
        theta_bounded = ((theta - theta_ref + 180) % 360) - 180
        theta_span = np.percentile(theta_bounded, 95) - np.percentile(theta_bounded, 5)
        theta_max_abs = np.max(np.abs(theta_bounded))
        theta_typical_abs = np.percentile(np.abs(theta_bounded), 95)
        theta_std = np.std(theta_bounded)
        dist_to_pivot_test = np.sqrt((tx - a) ** 2 + (ty - b) ** 2)
        radius_mean_err = abs(np.mean(dist_to_pivot_test) - radius)
        radius_std_test = np.std(dist_to_pivot_test)

        # Prefer physically plausible trajectories, avoid flat theta tracks, and penalize
        # geometric mismatch between test trajectory and calibrated pivot circle.
        cap_violation = max(0.0, theta_max_abs - THETA_MAX_DEG)
        theta_std_floor_penalty = max(0.0, MIN_THETA_STD_DEG - theta_std)
        typical_target_term = (
            TYPICAL_TARGET_WEIGHT * abs(theta_typical_abs - TARGET_THETA_TYPICAL_DEG)
            if USE_TYPICAL_TARGET else 0.0
        )
        objective = (
            5.0 * cap_violation
            + typical_target_term
            + 0.2 * resid_rms
            + 10.0 * (1.0 - planarity)
            + THETA_STD_FLOOR_WEIGHT * theta_std_floor_penalty
            + RADIUS_MEAN_ERR_WEIGHT * radius_mean_err
            + RADIUS_STD_WEIGHT * radius_std_test
        )
        return {
            "objective": objective,
            "seed_ts": seed_ts,
            "seed_j": seed_j,
            "test_seed_dist": test_seed_dist,
            "test_seed_ts": test_seed_ts,
            "test_seed_j": test_seed_j,
            "local_cal": local_cal,
            "local_test": local_test,
            "origin": origin,
            "basis": basis,
            "a": a,
            "b": b,
            "radius": radius,
            "resid_rms": resid_rms,
            "planarity": planarity,
            "theta_bounded": theta_bounded,
            "theta_span": theta_span,
            "theta_max_abs": theta_max_abs,
            "theta_typical_abs": theta_typical_abs,
            "theta_std": theta_std,
            "cap_violation": cap_violation,
            "radius_mean_err": radius_mean_err,
            "radius_std_test": radius_std_test,
        }

    cal_ts_coarse = _downsample_ts(cal_ts, COARSE_TS_STRIDE)
    test_ts_coarse = _downsample_ts(test_ts, COARSE_TS_STRIDE)

    def _run_search_pass(seed_budget, label):
        seed_list = []
        for ts in cal_ts_coarse:
            for j in range(len(cal_candidates[ts])):
                seed_list.append((ts, j))
                if len(seed_list) >= seed_budget:
                    break
            if len(seed_list) >= seed_budget:
                break
        if not seed_list:
            return {
                "seed_list": [],
                "refine_seeds": [],
                "refined_results": [],
            }

        coarse_scores = []
        for i, (seed_ts, seed_j) in enumerate(seed_list, start=1):
            result = _evaluate_seed(seed_ts, seed_j, cal_ts_coarse, test_ts_coarse)
            if result is not None:
                coarse_scores.append((result["objective"], result["theta_max_abs"], seed_ts, seed_j))
            if i % 100 == 0:
                print(f"seed search coarse progress ({label}): {i}/{len(seed_list)}")

        if not coarse_scores:
            return {
                "seed_list": seed_list,
                "refine_seeds": [],
                "refined_results": [],
            }

        best_by_objective = sorted(coarse_scores, key=lambda x: x[0])[:REFINE_TOP_K]
        best_by_cap = sorted(coarse_scores, key=lambda x: x[1])[:REFINE_TOP_K_CAP]
        refine_seed_set = {(ts, j) for _, _, ts, j in best_by_objective}
        refine_seed_set.update((ts, j) for _, _, ts, j in best_by_cap)
        refine_seeds = list(refine_seed_set)

        refined_results = []
        for seed_ts, seed_j in refine_seeds:
            result = _evaluate_seed(seed_ts, seed_j, cal_ts, test_ts)
            if result is None:
                continue
            refined_results.append(result)

        return {
            "seed_list": seed_list,
            "refine_seeds": refine_seeds,
            "refined_results": refined_results,
        }

    primary_pass = _run_search_pass(SEED_BUDGET, "primary")
    if not primary_pass["seed_list"]:
        raise RuntimeError("no candidate fifth-marker seeds found in calibration recording")
    if not primary_pass["refined_results"]:
        raise RuntimeError("no viable fifth-marker candidate after coarse screening")

    combined_results = list(primary_pass["refined_results"])
    total_seeds_evaluated = len(primary_pass["seed_list"])
    total_refine_seeds = len(primary_pass["refine_seeds"])

    if ENABLE_EXPANDED_CAP_SEARCH:
        primary_cap = [r for r in primary_pass["refined_results"] if r["theta_max_abs"] <= THETA_MAX_DEG]
        if not primary_cap and FALLBACK_SEED_BUDGET > SEED_BUDGET:
            print(
                "no cap-compliant candidate in primary pass; running expanded cap search "
                f"(budget={FALLBACK_SEED_BUDGET})"
            )
            fallback_pass = _run_search_pass(FALLBACK_SEED_BUDGET, "fallback")
            if fallback_pass["refined_results"]:
                combined_results.extend(fallback_pass["refined_results"])
                total_seeds_evaluated += len(fallback_pass["seed_list"])
                total_refine_seeds += len(fallback_pass["refine_seeds"])

    refined_results = combined_results
    if not refined_results:
        raise RuntimeError("could not identify a consistent fifth-marker trajectory")

    cap_compliant = [r for r in refined_results if r["theta_max_abs"] <= THETA_MAX_DEG]
    if cap_compliant:
        best = min(cap_compliant, key=lambda r: r["objective"])
        cap_compliant_count = len(cap_compliant)
    else:
        best = min(refined_results, key=lambda r: (r["cap_violation"], r["objective"]))
        cap_compliant_count = 0


    debug = {
        "cal": dbg_cal,
        "test": dbg_test,
        "n_seeds_evaluated": total_seeds_evaluated,
        "n_seeds_refined": total_refine_seeds,
        "n_cap_compliant_refined": cap_compliant_count,
        "coarse_stride": COARSE_TS_STRIDE,
        "selected_seed_ts": best["seed_ts"],
        "selected_seed_index": best["seed_j"],
        "selected_test_seed_ts": best["test_seed_ts"],
        "selected_test_seed_index": best["test_seed_j"],
        "selected_test_seed_dist_mm": best["test_seed_dist"],
        "selected_theta_span_deg": best["theta_span"],
        "selected_theta_max_abs_deg": best["theta_max_abs"],
        "selected_theta_typical_abs_deg": best["theta_typical_abs"],
        "selected_theta_std_deg": best["theta_std"],
        "selected_theta_cap_violation_deg": best["cap_violation"],
        "selected_radius_mean_err_mm": best["radius_mean_err"],
        "selected_radius_std_test_mm": best["radius_std_test"],
    }
    return cal_ts, best["local_cal"], test_ts, best["local_test"], best["theta_bounded"], best, debug


tool_motion_audit(CAMERA_FOLDER)

print(f"analysis trial: {ACTIVE_TRIAL}")
print(f"handle DRB UUID:   {HANDLE_DRB_UUID} ({TOOL_NAME_BY_UUID.get(HANDLE_DRB_UUID, 'Unknown')})")
print(f"helper DRB UUID:   {REFERENCE_DRB_UUID} ({TOOL_NAME_BY_UUID.get(REFERENCE_DRB_UUID, 'Unknown')})")
print(f"calibration (pivot-fit) recording: {BASE}\\{CALIBRATION_FOLDER}\\pf\\poses.txt")
print(f"test recording: {BASE}\\{CAMERA_FOLDER}\\pf\\poses.txt")
force_path = fr"{BASE}\{FORCE_CSV}"
print(f"force file: {force_path}")

# --- calibrate the pivot (plane + circle) from the wide-sweep recording ---
cal_ts, ml_cal, cam_ts, ml_test, theta_deg, track_fit, track_debug = choose_fifth_marker_tracks(
    CALIBRATION_FOLDER,
    CAMERA_FOLDER,
    HANDLE_DRB_UUID,
    REFERENCE_DRB_UUID,
)

plane_origin = track_fit["origin"]
plane_basis = track_fit["basis"]
a, b = track_fit["a"], track_fit["b"]
pivot_radius_mm = track_fit["radius"]
planarity = track_fit["planarity"]
fit_resid_rms = track_fit["resid_rms"]

print(f"calibration candidate source: {track_debug['cal']['poses_path']}")
print(f"test candidate source: {track_debug['test']['poses_path']}")
print(f"pattern source (cal): {track_debug['cal']['pattern_path']}")
print(f"pattern source (test): {track_debug['test']['pattern_path']}")
print(f"seed search evaluated: {track_debug['n_seeds_evaluated']} candidates")
print(f"seed search refined: {track_debug['n_seeds_refined']} candidates (coarse stride={track_debug['coarse_stride']})")
print(f"cap-compliant refined candidates: {track_debug['n_cap_compliant_refined']}")
print(f"selected calibration seed: ts={track_debug['selected_seed_ts']}, idx={track_debug['selected_seed_index']}")
print(f"selected test seed: ts={track_debug['selected_test_seed_ts']}, idx={track_debug['selected_test_seed_index']}, "
      f"dist={track_debug['selected_test_seed_dist_mm']:.2f} mm")
print(f"selected theta span (p95-p5): {track_debug['selected_theta_span_deg']:.2f} deg")
print(f"selected theta |95th|: {track_debug['selected_theta_typical_abs_deg']:.2f} deg")
print(f"selected theta std: {track_debug['selected_theta_std_deg']:.2f} deg")
print(f"selected theta |max|: {track_debug['selected_theta_max_abs_deg']:.2f} deg "
    f"(cap violation: {track_debug['selected_theta_cap_violation_deg']:.2f} deg)")
print(
    f"selected pivot mismatch on test: mean-radius error={track_debug['selected_radius_mean_err_mm']:.2f} mm, "
    f"radius std={track_debug['selected_radius_std_test_mm']:.2f} mm"
)
if track_debug["selected_theta_std_deg"] < MIN_THETA_STD_DEG:
    print(
        f"WARNING: theta variability is low (std={track_debug['selected_theta_std_deg']:.2f} deg < "
        f"{MIN_THETA_STD_DEG:.2f} deg)."
    )
print(f"variance explained by fitted plane: {100 * planarity:.2f}%")

p2 = (ml_cal - plane_origin) @ plane_basis.T
px, py = p2[:, 0], p2[:, 1]
resid = np.sqrt((px - a) ** 2 + (py - b) ** 2) - pivot_radius_mm
print(f"calibrated pivot: radius = {pivot_radius_mm:.2f} mm, fit residual RMS = {fit_resid_rms:.2f} mm")

# --- apply the calibrated pivot to the test recording ---
p2_test = (ml_test - plane_origin) @ plane_basis.T
tx, ty = p2_test[:, 0], p2_test[:, 1]
dist_to_pivot = np.sqrt((tx - a) ** 2 + (ty - b) ** 2)
print(f"test recording distance to calibrated pivot: {dist_to_pivot.mean():.2f} +/- {dist_to_pivot.std():.2f} mm "
      f"(should be close to {pivot_radius_mm:.2f} mm if the same hinge geometry applies)")

do_m = do_mm / 1000
delta_tip_mm = do_mm * np.radians(theta_deg)
theta_valid = theta_deg[np.isfinite(theta_deg)]
print(f"theta range (5th to 95th percentile): {np.percentile(theta_valid, 5):.2f} to {np.percentile(theta_valid, 95):.2f} deg")
print(f"theta full min/max: {np.min(theta_valid):.2f} to {np.max(theta_valid):.2f} deg")

# force channel (calibrated, N) at 100 Hz
df_spr = pd.read_csv(force_path)
time_us = pd.to_numeric(df_spr["Time"], errors="coerce")
time_s = (time_us - time_us.iloc[0]) / 1e6
force_N = df_spr.iloc[:, 2].to_numpy()

# wall-clock sync: Force DAQ (Intervention Tracker "Scan Start", Eastern local time)
# vs NDI camera (recordingMetadata.json "start", UTC)
tracker = pd.read_excel(fr"{BASE}\Force_Data\Intervention Tracker.xlsx", sheet_name="8-20")
scan_start = tracker.loc[tracker["File Name"] == INTERVENTION_FILE_NAME, "Scan Start"].iloc[0]
force_epoch = pd.Timestamp(scan_start).tz_localize("America/New_York").tz_convert("UTC").timestamp() + time_s.to_numpy()

meta = json.load(open(fr"{BASE}\{CAMERA_FOLDER}\recordingMetadata.json"))
cam_epoch = pd.Timestamp(meta["start"]["date_time"]).timestamp() + (cam_ts - meta["start"]["timestamp"]) / 1e6

in_overlap = (force_epoch >= cam_epoch.min()) & (force_epoch <= cam_epoch.max())
valid_cam = np.isfinite(theta_deg) & np.isfinite(delta_tip_mm)
if np.sum(valid_cam) < 2:
    raise RuntimeError("not enough valid theta samples after fifth-marker filtering")
theta_on_force = np.interp(force_epoch[in_overlap], cam_epoch[valid_cam], theta_deg[valid_cam])
delta_tip_on_force = np.interp(force_epoch[in_overlap], cam_epoch[valid_cam], delta_tip_mm[valid_cam])

sync_df = pd.DataFrame({
    "time_s": time_s.to_numpy()[in_overlap] - time_s.to_numpy()[in_overlap][0],
    "force_N": force_N[in_overlap],
    "theta_deg": theta_on_force,
    "delta_tip_mm": delta_tip_on_force,
})

#%
# Validation plot: marker trajectory (both recordings) in the calibrated
# pivot plane, with the fitted circle overlaid.
if ENABLE_PLOTS:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(px, py, ".", markersize=2, alpha=0.4, label="1 - Intact (calibration)")
    ax.plot(tx, ty, ".", markersize=2, alpha=0.4, label="2 - Holding Longer (test)")
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(a + pivot_radius_mm * np.cos(circle), b + pivot_radius_mm * np.sin(circle), "k--", linewidth=1, label="fitted pivot circle")
    ax.plot(a, b, "k+", markersize=12, label="pivot")
    ax.set_xlabel("plane axis 1 (mm)")
    ax.set_ylabel("plane axis 2 (mm)")
    ax.set_title("Fifth-marker trajectory in the handle DRB local frame")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_or_show(fig, "pivot_plane_trajectory")

#%
# Inspect Force, theta and delta_tip together to identify a clean loading/unloading cycle.
if ENABLE_PLOTS or SAVE_PLOTS:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sync_df["time_s"], sync_df["force_N"], color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force (N)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(sync_df["time_s"], sync_df["theta_deg"], color="tab:red")
    ax2.set_ylabel("theta (deg)", color="tab:red")
    ax1.set_title("Lamina Spreader - Force & Pivot Angle (Intact, Holding Longer)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_or_show(fig, "force_and_theta_vs_time")

# Requested diagnostic plot: Force vs Theta.
if ENABLE_PLOTS or SAVE_PLOTS:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sync_df["theta_deg"], sync_df["force_N"], marker="o", markersize=2, linewidth=0.5)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Force (N)")
    ax.set_title(f"Force vs Theta ({ACTIVE_TRIAL})")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_or_show(fig, "force_vs_theta")

#%
# Pick the sample range of a clean loading cycle from the plot above (default:
# full overlap window), then fit k_tip = slope(Force vs delta_tip) over it.
fit_mask = sync_df["time_s"] < sync_df["time_s"].max()   # <-- adjust after inspecting the plot above
fit_df = sync_df[fit_mask]

if ENABLE_PLOTS or SAVE_PLOTS:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fit_df["delta_tip_mm"], fit_df["force_N"], marker="o", markersize=3, linewidth=0.5)
    ax.set_xlabel("delta_tip (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Lamina Spreader Tip Stiffness")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_or_show(fig, "force_vs_tip_displacement")

k_tip_N_per_mm, intercept = np.polyfit(fit_df["delta_tip_mm"], fit_df["force_N"], 1)
k_tip_N_per_m = k_tip_N_per_mm * 1000
k_rot_Nm_per_deg = k_tip_N_per_m * do_m ** 2 * 0.01745  # 0.01745 = pi/180 (rad -> deg)

print(f"k_tip = {k_tip_N_per_mm:.4f} N/mm ({k_tip_N_per_m:.2f} N/m)")
print(f"do    = {do_mm:.1f} mm")
print(f"k_rot = {k_rot_Nm_per_deg:.4f} Nm/deg")

