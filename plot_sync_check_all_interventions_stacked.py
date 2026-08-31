from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

START_LOOKBACK = 60
SHORT_PEAK_GAP_MIN_S = 2.5
SHORT_PEAK_GAP_MAX_S = 10.0
SHORT_PEAK_GAP_FRAC = 0.35
MIN_RUN_SAMPLES = 30
MIN_RISE_SEC = 1.2
MIN_RISE_THETA_DEG = 0.2
FORCE_YLIM_N = (-25.0, 125.0)
THETA_YLIM_DEG = (-4.0, 7.0)

# Global transition-flag criteria (applied across all interventions together).
TRANSITION_FORCE_Q = 35.0
TRANSITION_THETA_Q = 75.0
TRANSITION_MIN_BLOCK = 2
TRANSITION_EDGE_FORCE_RELAX = 1.08
SHOW_IDENTIFICATION_OVERLAYS = False


def _clean_intervention(name: str) -> str:
    txt = str(name).strip().replace("_", " ")
    txt = re.sub(r"^\d+\s*-\s*", "", txt)
    txt = " ".join(txt.split())
    low = txt.lower()
    if "intact" in low and "holding" in low:
        return "Intact"
    if low == "intact":
        return "Intact"
    if "pubf" in low or low.startswith("puf"):
        return "PUF"
    if low.startswith("fuf"):
        return "FUF"
    return txt


def _display_case_name(file_name: str, intervention: str) -> str:
    m = re.match(r"^(\d+)\s*-\s*", str(file_name).strip())
    if m:
        return f"{m.group(1)} - {_clean_intervention(intervention)}"
    return _clean_intervention(intervention)


def require_flag_block(cyc: pd.DataFrame, min_block: int = TRANSITION_MIN_BLOCK) -> pd.Series:
    """Keep transition flags only when they appear in contiguous cycle blocks."""
    keep = pd.Series(False, index=cyc.index)
    for file_name, g in cyc.groupby("file_name", sort=False):
        gg = g.sort_values("cycle")
        raw = gg["transition_flag_raw"].to_numpy(bool)
        cyc_no = gg["cycle"].to_numpy(int)
        idx = gg.index.to_numpy()

        block_start = None
        for i in range(len(gg)):
            if not raw[i]:
                if block_start is not None:
                    block_end = i - 1
                    block_len = block_end - block_start + 1
                    if block_len >= min_block:
                        keep.loc[idx[block_start:block_end + 1]] = True
                    block_start = None
                continue

            if block_start is None:
                block_start = i
                continue

            if cyc_no[i] != cyc_no[i - 1] + 1:
                block_end = i - 1
                block_len = block_end - block_start + 1
                if block_len >= min_block:
                    keep.loc[idx[block_start:block_end + 1]] = True
                block_start = i

        if block_start is not None:
            block_end = len(gg) - 1
            block_len = block_end - block_start + 1
            if block_len >= min_block:
                keep.loc[idx[block_start:block_end + 1]] = True

    return keep


def expand_flag_edges(
    cyc: pd.DataFrame,
    base_flag: pd.Series,
    f_cut: float,
    th_cut: float,
    force_relax: float = TRANSITION_EDGE_FORCE_RELAX,
) -> pd.Series:
    """Expand accepted transition blocks by one adjacent near-threshold cycle."""
    out = base_flag.copy()
    for _, g in cyc.groupby("file_name", sort=False):
        gg = g.sort_values("cycle")
        idx = gg.index.to_numpy()
        cyc_no = gg["cycle"].to_numpy(int)
        fpk = gg["F_peak_N"].to_numpy(float)
        th0 = gg["theta_start_deg"].to_numpy(float)
        base = out.loc[idx].to_numpy(bool)

        add = np.zeros_like(base)
        for i in range(len(gg)):
            if base[i]:
                continue

            left_neighbor = i > 0 and cyc_no[i] == cyc_no[i - 1] + 1 and base[i - 1]
            right_neighbor = i < len(gg) - 1 and cyc_no[i + 1] == cyc_no[i] + 1 and base[i + 1]
            if not (left_neighbor or right_neighbor):
                continue

            if th0[i] >= th_cut and fpk[i] <= f_cut * force_relax:
                add[i] = True

        out.loc[idx] = base | add

    return out


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


def load_synced_case(file_name: str) -> dict:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag0 = ls.find_lag(force, kin)
    lag, _ = ls.refine_lag(kin, force, lag0)
    # Step 1 is intended to show raw synchronized signals (no theta smoothing,
    # no theta0 subtraction), so we sync force onto the mocap clock directly.
    k = kin.copy()
    k["t_sync"] = k["t_s"] + lag
    k["force"] = np.interp(
        k["t_sync"].to_numpy(float),
        force["t_s"].to_numpy(float),
        force["force"].to_numpy(float),
        left=np.nan,
        right=np.nan,
    )
    d = k[k["force"].notna() & k["theta_deg"].notna()].reset_index(drop=True)
    fmax = float(np.nanmax(force["force"].to_numpy(float)))

    t = d["t_sync"].to_numpy(float)
    f = d["force"].to_numpy(float)
    th = d["theta_deg"].to_numpy(float)

    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    cands = []
    for s, e in runs:
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, _, pk = rb
        cands.append(
            {
                "s": int(s),
                "e": int(e),
                "i0": int(i0),
                "pk": int(pk),
                "fpk": float(f[pk]),
                "dur": int(e - s),
            }
        )

    cands = dedupe_peak_candidates(cands, t)
    cands = filter_pseudoramps(cands, t, th)
    cands = sorted(cands, key=lambda c: c["pk"])

    cycles = []
    for n, c in enumerate(cands, 1):
        i0 = int(c["i0"])
        pk = int(c["pk"])
        cycles.append(
            {
                "cycle": int(n),
                "i0": i0,
                "pk": pk,
                "t_start_s": float(t[i0]),
                "t_peak_s": float(t[pk]),
                "rise_s": float(t[pk] - t[i0]),
                "F_start_N": float(f[i0]),
                "F_peak_N": float(f[pk]),
                "theta_start_deg": float(th[i0]),
                "theta_peak_deg": float(th[pk]),
                "dtheta_deg": float(th[pk] - th[i0]),
                "run_samples": int(c["dur"]),
            }
        )

    starts = np.array([c["i0"] for c in cycles], dtype=int)

    return {
        "lag": float(lag),
        "t": t,
        "f": f,
        "th": th,
        "start_idx": starts,
        "cycles": cycles,
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

    # Build all case data first so transition flags can be computed globally.
    case_data = []
    rows = []
    for file_name, intervention in cases:
        print(f"=== {file_name} | {intervention} ===")
        case = load_synced_case(file_name)
        case_data.append((file_name, intervention, case))
        for c in case["cycles"]:
            rows.append(
                {
                    "file_name": file_name,
                    "intervention": intervention,
                    **c,
                }
            )

    cyc = pd.DataFrame(rows)
    if cyc.empty:
        raise RuntimeError("No valid cycles found for transition analysis.")

    f_cut = float(np.nanpercentile(cyc["F_peak_N"].to_numpy(float), TRANSITION_FORCE_Q))
    th_cut = float(np.nanpercentile(cyc["theta_start_deg"].to_numpy(float), TRANSITION_THETA_Q))
    cyc["transition_flag_raw"] = (cyc["F_peak_N"] <= f_cut) & (cyc["theta_start_deg"] >= th_cut)
    block_flag = require_flag_block(cyc)
    cyc["transition_flag"] = expand_flag_edges(cyc, block_flag, f_cut, th_cut)

    print("\nGlobal transition flag thresholds")
    print(f"  F_peak <= Q{TRANSITION_FORCE_Q:.0f}: {f_cut:.2f} N")
    print(f"  theta_start >= Q{TRANSITION_THETA_Q:.0f}: {th_cut:.2f} deg")

    flagged_raw = cyc[cyc["transition_flag_raw"]].copy()
    flagged = cyc[cyc["transition_flag"]].copy()
    print(f"  raw flagged cycles: {len(flagged_raw)}/{len(cyc)}")
    print(f"  block+edge flagged cycles: {len(flagged)}/{len(cyc)}")
    if SHOW_IDENTIFICATION_OVERLAYS and not flagged.empty:
        print(
            flagged[
                [
                    "file_name",
                    "intervention",
                    "cycle",
                    "t_start_s",
                    "F_start_N",
                    "F_peak_N",
                    "theta_start_deg",
                    "dtheta_deg",
                ]
            ].to_string(index=False)
        )
    if not SHOW_IDENTIFICATION_OVERLAYS:
        print("  Note: transition overlays are hidden in this figure, but flags are still saved to CSV.")

    out_csv = OUT_DIR / "sync_check_transition_flags.csv"
    cyc.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    flag_map = {
        (str(r.file_name), int(r.cycle)): bool(r.transition_flag)
        for r in cyc.itertuples(index=False)
    }

    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(14.0, 2.25 * n), squeeze=False)
    axes = axes.ravel()

    for i, ((file_name, intervention, case), ax) in enumerate(zip(case_data, axes)):
        t = case["t"]
        f = case["f"]
        th = case["th"]
        idx = case["start_idx"]
        cyc_meta = case["cycles"]

        h_force, = ax.plot(t, f, lw=0.9, color="tab:blue", label="distraction force")
        ax.set_ylabel("force [N]", color="tab:blue")
        ax.set_ylim(*FORCE_YLIM_N)
        ax.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax.twinx()
        h_theta, = ax2.plot(t, th, lw=0.9, color="tab:orange", label="theta")
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(*THETA_YLIM_DEG)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        h_start_force = None
        h_start_theta = None
        h_flag_force = None
        h_flag_theta = None
        if SHOW_IDENTIFICATION_OVERLAYS and idx.size > 0:
            is_flag = np.array(
                [flag_map.get((file_name, int(c["cycle"])), False) for c in cyc_meta],
                dtype=bool,
            )
            idx_ok = idx[~is_flag]
            idx_flag = idx[is_flag]

            h_start_force = ax.scatter(
                t[idx_ok],
                f[idx_ok],
                s=32,
                marker="o",
                facecolor="white",
                edgecolor="tab:red",
                linewidth=1.1,
                zorder=4,
                label="ramp start (force)",
            )
            h_start_theta = ax2.scatter(
                t[idx_ok],
                th[idx_ok],
                s=26,
                marker="D",
                facecolor="white",
                edgecolor="tab:red",
                linewidth=0.9,
                zorder=4,
                label="ramp start (theta)",
            )
            if idx_flag.size > 0:
                h_flag_force = ax.scatter(
                    t[idx_flag],
                    f[idx_flag],
                    s=54,
                    marker="o",
                    facecolor="tab:red",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                    label="transition start (force)",
                )
                h_flag_theta = ax2.scatter(
                    t[idx_flag],
                    th[idx_flag],
                    s=46,
                    marker="D",
                    facecolor="tab:red",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                    label="transition start (theta)",
                )

        case_disp = _display_case_name(file_name, intervention)
        ax.set_title(f"{case_disp}   sync check (lag {case['lag']:+.2f} s)", fontsize=10)
        ax.grid(alpha=0.3)

        if i == 0:
            handles = [h_force, h_theta]
            if h_start_force is not None:
                handles.append(h_start_force)
            if h_start_theta is not None:
                handles.append(h_start_theta)
            if h_flag_force is not None:
                handles.append(h_flag_force)
            if h_flag_theta is not None:
                handles.append(h_flag_theta)
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, loc="upper right", fontsize=8)

        if i == n - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        (
            "Lamina spreader sync check (top panel style) - all interventions\n"
            f"transition flag: F_peak<=Q{TRANSITION_FORCE_Q:.0f} and theta_start>=Q{TRANSITION_THETA_Q:.0f}"
            if SHOW_IDENTIFICATION_OVERLAYS
            else "Lamina spreader sync check (top panel style) - all interventions\n"
            "raw force/theta/time only (identification overlays disabled)"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / "sync_check_all_interventions_stacked.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
