from pathlib import Path

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
THETA_ZERO_SNAP_WINDOW_SEC = 1.5
THETA_ZERO_FORCE_FRACTION = 0.25
THETA_ZERO_MAX_ABS_DEG = 0.35
THETA_ZERO_SOURCE_MAX_ABS_DEG = 1.0
NON_NEUTRAL_WINDOW_SEC = 1.5
NON_NEUTRAL_THETA_DEG = 2.0
NON_NEUTRAL_MIN_FRACTION = 0.25
NON_NEUTRAL_MEDIAN_MIN_DEG = 0.5
NON_NEUTRAL_FORCE_START_DEG = 2.0

# Global transition-flag criteria (applied across all interventions together).
TRANSITION_FORCE_Q = 35.0
TRANSITION_THETA_Q = 75.0
TRANSITION_MIN_BLOCK = 2
TRANSITION_EDGE_FORCE_RELAX = 1.08
FORCE_REF_MOMENTS_NM = (0.5, 4.0)
FORCE_REF_VALUES_N = tuple(float(ls.moment_to_force_n(m)) for m in FORCE_REF_MOMENTS_NM)
SECANT_CYCLES_CSV = OUT_DIR / "spreader_secant_cycles_preloadcorr_rampref_fzero.csv"


def require_flag_block(cyc: pd.DataFrame, min_block: int = TRANSITION_MIN_BLOCK) -> pd.Series:
    """Keep transition flags only when they appear in contiguous cycle blocks."""
    keep = pd.Series(False, index=cyc.index)
    for _, g in cyc.groupby("file_name", sort=False):
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
        fpk = gg["F_peak_raw_N"].to_numpy(float)
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


def snap_start_to_theta_zero(
    t: np.ndarray,
    f: np.ndarray,
    th: np.ndarray,
    s: int,
    i0: int,
    pk: int,
) -> int:
    """Nudge force-based ramp start to nearby theta~0 within the early ramp.

    Guardrail: if the force-based start is already far from neutral, keep it.
    This avoids overwriting clearly non-neutral starts in late/transition cycles.
    """
    if pk <= i0 + 2:
        return int(i0)

    if not np.isfinite(th[i0]) or abs(float(th[i0])) > THETA_ZERO_SOURCE_MAX_ABS_DEG:
        return int(i0)

    dt = float(np.nanmedian(np.diff(t))) if t.size >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    w = max(3, int(round(THETA_ZERO_SNAP_WINDOW_SEC / dt)))

    lo = max(0, int(i0) - w)
    hi = min(int(pk), int(i0) + w)
    if hi <= lo + 1:
        lo = int(i0)
        hi = int(pk)
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

    j = int(idx[int(np.argmin(np.abs(th[idx])))] )
    if abs(float(th[j])) <= THETA_ZERO_MAX_ABS_DEG:
        return j
    return int(i0)


def detect_non_neutral_start(
    t: np.ndarray,
    th: np.ndarray,
    i0_force: int,
) -> tuple[float, float, bool, bool, bool]:
    """Detect non-neutral starts from sustained pre-ramp theta before snapping.

    Returns:
      (pre-ramp median theta,
       pre-ramp fraction above theta threshold,
       force-start theta criterion,
       pre-ramp window criterion,
       combined non-neutral flag)
    """
    if i0_force <= 0:
        return np.nan, np.nan, False, False, False

    force_start_flag = bool(np.isfinite(th[i0_force]) and float(th[i0_force]) >= NON_NEUTRAL_FORCE_START_DEG)

    dt = float(np.nanmedian(np.diff(t))) if t.size >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    w = max(3, int(round(NON_NEUTRAL_WINDOW_SEC / dt)))
    lo = max(0, int(i0_force) - w)
    hi = int(i0_force)
    if hi <= lo:
        return np.nan, np.nan, force_start_flag, False, force_start_flag

    vals = th[lo:hi + 1]
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan, np.nan, force_start_flag, False, force_start_flag

    med = float(np.nanmedian(vals))
    frac = float(np.mean(vals >= NON_NEUTRAL_THETA_DEG))
    window_flag = bool(med >= NON_NEUTRAL_MEDIAN_MIN_DEG and frac >= NON_NEUTRAL_MIN_FRACTION)
    flag = bool(force_start_flag or window_flag)
    return med, frac, force_start_flag, window_flag, flag


def build_preloadfree_trace(f: np.ndarray, cycles: list[dict]) -> np.ndarray:
    """Cycle-local preload-free force trace, referenced to each ramp start."""
    f_dyn = np.full_like(f, np.nan, dtype=float)
    for c in cycles:
        i0 = int(c["i0"])
        e = int(c["e"])
        if i0 >= e:
            continue
        f0 = float(f[i0])
        f_dyn[i0:e] = f[i0:e] - f0
    return f_dyn


def load_synced_case(file_name: str) -> dict:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag0 = ls.find_lag(force, kin)
    lag, _ = ls.refine_lag(kin, force, lag0)
    d, _, fmax = ls.sync(kin, force, lag)

    t = d["t_sync"].to_numpy(float)
    f = d["force"].to_numpy(float)
    th = d["theta_dist"].to_numpy(float)

    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    cands = []
    for s, e in runs:
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0_force, _, pk = rb
        i0 = snap_start_to_theta_zero(t, f, th, s, i0_force, pk)
        cands.append(
            {
                "s": int(s),
                "e": int(e),
                "i0_force": int(i0_force),
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
        i0_force = int(c["i0_force"])
        i0 = int(c["i0"])
        pk = int(c["pk"])
        f0 = float(f[i0])
        th_pre_med, th_pre_frac, nn_force_flag, nn_window_flag, nn_flag = detect_non_neutral_start(
            t, th, i0_force
        )
        cycles.append(
            {
                "cycle": int(n),
                "s": int(c["s"]),
                "e": int(c["e"]),
                "i0_force": i0_force,
                "i0": i0,
                "pk": pk,
                "t_start_s": float(t[i0]),
                "t_peak_s": float(t[pk]),
                "rise_s": float(t[pk] - t[i0]),
                "F_start_raw_N": f0,
                "F_peak_raw_N": float(f[pk]),
                "F_peak_dyn_N": float(f[pk] - f0),
                "theta_force_start_deg": float(th[i0_force]),
                "theta_start_deg": float(th[i0]),
                "theta_peak_deg": float(th[pk]),
                "dtheta_deg": float(th[pk] - th[i0]),
                "theta_preramp_med_deg": th_pre_med,
                "theta_preramp_open_frac": th_pre_frac,
                "non_neutral_force_start_flag": bool(nn_force_flag),
                "non_neutral_preramp_window_flag": bool(nn_window_flag),
                "non_neutral_start_window_flag": bool(nn_flag),
                "run_samples": int(c["dur"]),
            }
        )

    starts = np.array([c["i0"] for c in cycles], dtype=int)
    f_dyn = build_preloadfree_trace(f, cycles)

    return {
        "lag": float(lag),
        "t": t,
        "f": f,
        "f_dyn": f_dyn,
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


def load_secant_metrics(
    cyc_plot: pd.DataFrame | None = None,
) -> tuple[dict[str, tuple[int, int]], dict[tuple[str, int], tuple[float, float]], str]:
    """Load secant counts and per-cycle theta@moment markers, excluding transition cycles when possible."""
    if not SECANT_CYCLES_CSV.exists():
        return {}, {}, "N/A"

    try:
        sec = pd.read_csv(SECANT_CYCLES_CSV)
    except Exception:
        return {}, {}, "N/A"

    req = {"intervention", "valid_secant"}
    if not req.issubset(set(sec.columns)):
        return {}, {}, "N/A"

    v = sec["valid_secant"]
    if v.dtype == bool:
        valid = v
    else:
        valid = v.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    sec = sec.assign(valid_secant_bool=valid)

    counts: dict[str, tuple[int, int]] = {}
    theta_marks: dict[tuple[str, int], tuple[float, float]] = {}

    # Preferred path: count only plotted cycles and exclude transition-flagged ones.
    match_cols = {"file_name", "intervention", "t_start_s", "t_peak_s"}
    if (
        cyc_plot is not None
        and match_cols.issubset(set(cyc_plot.columns))
        and match_cols.issubset(set(sec.columns))
        and "transition_flag" in cyc_plot.columns
    ):
        cp = cyc_plot[
            ["file_name", "intervention", "cycle", "t_start_s", "t_peak_s", "transition_flag"]
        ].copy()
        cp["t_start_key"] = pd.to_numeric(cp["t_start_s"], errors="coerce").round(3)
        cp["t_peak_key"] = pd.to_numeric(cp["t_peak_s"], errors="coerce").round(3)

        sm = sec[
            [
                "file_name",
                "intervention",
                "t_start_s",
                "t_peak_s",
                "valid_secant_bool",
                "theta_ref_deg",
                "theta_low_deg",
                "theta_high_deg",
            ]
        ].copy()
        sm["t_start_key"] = pd.to_numeric(sm["t_start_s"], errors="coerce").round(3)
        sm["t_peak_key"] = pd.to_numeric(sm["t_peak_s"], errors="coerce").round(3)
        sm = (
            sm.groupby(["file_name", "intervention", "t_start_key", "t_peak_key"], as_index=False)
            .agg(
                valid_secant_bool=("valid_secant_bool", "max"),
                theta_ref_deg=("theta_ref_deg", "median"),
                theta_low_deg=("theta_low_deg", "median"),
                theta_high_deg=("theta_high_deg", "median"),
            )
        )

        m = cp.merge(
            sm,
            on=["file_name", "intervention", "t_start_key", "t_peak_key"],
            how="left",
        )
        m["transition_flag"] = m["transition_flag"].fillna(False).astype(bool)
        m["valid_secant_bool"] = m["valid_secant_bool"].fillna(False).astype(bool)
        m["theta_abs_low_deg"] = pd.to_numeric(m["theta_ref_deg"], errors="coerce") + pd.to_numeric(
            m["theta_low_deg"], errors="coerce"
        )
        m["theta_abs_high_deg"] = pd.to_numeric(m["theta_ref_deg"], errors="coerce") + pd.to_numeric(
            m["theta_high_deg"], errors="coerce"
        )

        keep = ~m["transition_flag"]
        g = (
            m.loc[keep]
            .groupby("intervention", as_index=False)
            .agg(
                n_cycles=("valid_secant_bool", "count"),
                n_valid=("valid_secant_bool", "sum"),
            )
        )
        counts = {
            str(r.intervention): (int(r.n_valid), int(r.n_cycles))
            for r in g.itertuples(index=False)
        }

        m_valid = m.loc[keep & m["valid_secant_bool"]].copy()
        for r in m_valid.itertuples(index=False):
            lo = float(r.theta_abs_low_deg) if np.isfinite(r.theta_abs_low_deg) else np.nan
            hi = float(r.theta_abs_high_deg) if np.isfinite(r.theta_abs_high_deg) else np.nan
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            theta_marks[(str(r.file_name), int(r.cycle))] = (lo, hi)

    # Fallback: count from all secant rows if cycle-level matching is unavailable.
    if not counts:
        g = (
            sec.groupby("intervention", as_index=False)
            .agg(
                n_cycles=("valid_secant_bool", "count"),
                n_valid=("valid_secant_bool", "sum"),
            )
        )
        counts = {
            str(r.intervention): (int(r.n_valid), int(r.n_cycles))
            for r in g.itertuples(index=False)
        }

    if {"M_low_Nm", "M_high_Nm"}.issubset(set(sec.columns)):
        m_low = float(pd.to_numeric(sec["M_low_Nm"], errors="coerce").dropna().iloc[0])
        m_high = float(pd.to_numeric(sec["M_high_Nm"], errors="coerce").dropna().iloc[0])
        wtxt = f"[{m_low:.1f}, {m_high:.1f}]"
    else:
        wtxt = "N/A"

    return counts, theta_marks, wtxt


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

    f_cut = float(np.nanpercentile(cyc["F_peak_raw_N"].to_numpy(float), TRANSITION_FORCE_Q))
    th_cut = float(np.nanpercentile(cyc["theta_start_deg"].to_numpy(float), TRANSITION_THETA_Q))
    cyc["transition_flag_raw"] = (cyc["F_peak_raw_N"] <= f_cut) & (cyc["theta_start_deg"] >= th_cut)
    block_flag = require_flag_block(cyc)
    cyc["transition_flag"] = expand_flag_edges(cyc, block_flag, f_cut, th_cut)

    print("\nGlobal transition flag thresholds")
    print(f"  F_peak_raw <= Q{TRANSITION_FORCE_Q:.0f}: {f_cut:.2f} N")
    print(f"  theta_start >= Q{TRANSITION_THETA_Q:.0f}: {th_cut:.2f} deg")

    flagged_raw = cyc[cyc["transition_flag_raw"]].copy()
    flagged = cyc[cyc["transition_flag"]].copy()
    nn = cyc["non_neutral_start_window_flag"].fillna(False).astype(bool)
    print(f"  raw flagged cycles: {len(flagged_raw)}/{len(cyc)}")
    print(f"  block+edge flagged cycles: {len(flagged)}/{len(cyc)}")
    print(f"  non-neutral start (window) cycles: {int(nn.sum())}/{len(cyc)}")

    out_csv = OUT_DIR / "sync_check_transition_flags_preloadfree_plot.csv"
    cyc.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    secant_valid_map, secant_theta_map, secant_window_txt = load_secant_metrics(cyc)

    flag_map = {
        (str(r.file_name), int(r.cycle)): bool(r.transition_flag)
        for r in cyc.itertuples(index=False)
    }

    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(14.0, 2.25 * n), squeeze=False)
    axes = axes.ravel()

    for i, ((file_name, intervention, case), ax) in enumerate(zip(case_data, axes)):
        t = case["t"]
        f_dyn = case["f_dyn"]
        th = case["th"]
        idx = case["start_idx"]

        h_force, = ax.plot(t, f_dyn, lw=0.9, color="tab:blue", label="distraction force (no preload)")
        ax.set_ylabel("dyn force [N]", color="tab:blue")
        ax.set_ylim(-25.0, 125.0)
        ax.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax.twinx()
        h_theta, = ax2.plot(t, th, lw=0.9, color="tab:orange", label="theta")
        ax2.axhline(0.0, color="tab:orange", lw=1.0, ls=":", alpha=0.8)
        ax2.set_ylabel("theta [deg]", color="tab:orange")
        ax2.set_ylim(-4.0, 10.0)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        h_start_force = None
        h_start_theta = None
        if idx.size > 0:
            h_start_force = ax.scatter(
                t[idx],
                f_dyn[idx],
                s=32,
                marker="o",
                facecolor="white",
                edgecolor="tab:red",
                linewidth=1.1,
                zorder=4,
                label="ramp start (dyn force)",
            )
            h_start_theta = ax2.scatter(
                t[idx],
                th[idx],
                s=26,
                marker="D",
                facecolor="white",
                edgecolor="tab:red",
                linewidth=0.9,
                zorder=4,
                label="ramp start (theta raw)",
            )

        valid_txt = ""
        if intervention in secant_valid_map:
            nv, nt = secant_valid_map[intervention]
            valid_txt = f" | valid secant (no-transition) {nv}/{nt}"

        ax.set_title(
            f"{file_name} | {intervention}   sync check preload-free (lag {case['lag']:+.2f} s){valid_txt}",
            fontsize=10,
        )
        ax.grid(alpha=0.3)

        if i == 0:
            handles = [h_force, h_theta]
            if h_start_force is not None:
                handles.append(h_start_force)
            if h_start_theta is not None:
                handles.append(h_start_theta)
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, loc="upper right", fontsize=8)

        if i == n - 1:
            ax.set_xlabel("time [s]")

    fig.suptitle(
        "Lamina spreader sync check (stacked) - preload-free distraction force\n"
        "Only force, theta, and ramp starts are shown; theta is displayed in raw values.",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / "sync_check_all_interventions_stacked_preloadfree.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
