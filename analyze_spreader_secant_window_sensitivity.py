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

THETA_REF_WINDOW = 10
FORCE_ZERO_BAND_N = 0.5
FORCE_ZERO_MAX_SAMPLES = 20
FORCE_ZERO_MIN_SAMPLES = 3
SECANT_MIN_DTHETA_DEG = 0.3

# Keep sensitivity study inside the production 1-4 Nm band.
LOW_GRID_NM = [float(x) for x in np.arange(1.0, 3.5 + 1e-9, 0.5)]
HIGH_GRID_NM = [float(x) for x in np.arange(1.5, 4.0 + 1e-9, 0.5)]
BASELINE_WINDOW = (1.0, 4.0)


def secant_on_rising_branch(theta_rel_deg: np.ndarray, moment_nm: np.ndarray, m_low: float, m_high: float) -> float:
    if theta_rel_deg.size < 2 or moment_nm.size < 2:
        return np.nan

    m_all = np.asarray(moment_nm, dtype=float)
    a_all = np.asarray(theta_rel_deg, dtype=float)
    keep = np.isfinite(m_all) & np.isfinite(a_all)
    m_all = m_all[keep]
    a_all = a_all[keep]
    if m_all.size < 2:
        return np.nan

    pk = int(np.nanargmax(m_all))
    if pk < 1:
        return np.nan

    below = np.flatnonzero(m_all[:pk + 1] <= (m_low + 1e-9))
    if below.size == 0:
        return np.nan
    i_start = int(below[-1])

    m_seg = m_all[i_start:pk + 1]
    a_seg = a_all[i_start:pk + 1]
    if m_seg.size < 2 or float(np.nanmax(m_seg)) < m_high:
        return np.nan

    hull = m_seg >= np.maximum.accumulate(m_seg) - 1e-9
    m = m_seg[hull]
    a = a_seg[hull]
    if m.size < 2:
        return np.nan

    order = np.argsort(m)
    m = m[order]
    a = a[order]

    m_u, first = np.unique(m, return_index=True)
    a_u = a[first]
    if m_u.size < 2:
        return np.nan

    if m_low < float(np.min(m_u)) or m_high > float(np.max(m_u)):
        return np.nan

    th_low = float(np.interp(m_low, m_u, a_u))
    th_high = float(np.interp(m_high, m_u, a_u))
    dtheta = th_high - th_low
    if not np.isfinite(dtheta) or dtheta < SECANT_MIN_DTHETA_DEG:
        return np.nan

    return float((m_high - m_low) / dtheta)


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


def build_cycle_curves(file_name: str, intervention: str) -> list[dict]:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, _, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()

    out = []
    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    for n, (s, e) in enumerate(runs, 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue

        i0, i1, pk = rb
        ramp = np.arange(i0, i1)
        if ramp.size < 3:
            continue

        f0 = float(f[i0])
        f_dyn = f[ramp] - f0
        if float(np.nanmax(f_dyn)) <= 0:
            continue

        early_n = min(FORCE_ZERO_MAX_SAMPLES, len(ramp))
        early = np.arange(early_n)
        zero_idx = early[np.abs(f_dyn[early]) <= FORCE_ZERO_BAND_N]

        if zero_idx.size >= FORCE_ZERO_MIN_SAMPLES:
            th_ref = float(np.nanmedian(th[ramp[zero_idx]]))
        else:
            iref0 = max(0, i0 - THETA_REF_WINDOW + 1)
            th_ref = float(np.nanmedian(th[iref0:i0 + 1]))

        if not np.isfinite(th_ref):
            th_ref = float(th[i0])

        theta_rel = th[ramp] - th_ref
        moment_rel = ls.force_to_moment_nm(f_dyn)

        out.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": int(n),
                "M_peak_dyn_Nm": float(np.nanmax(moment_rel)),
                "theta_rel_deg": theta_rel,
                "moment_rel_nm": moment_rel,
            }
        )

    return out


def summarize_window(cycles: list[dict], m_low: float, m_high: float) -> pd.DataFrame:
    rows = []
    for c in cycles:
        k = secant_on_rising_branch(c["theta_rel_deg"], c["moment_rel_nm"], m_low, m_high)
        rows.append(
            {
                "file_name": c["file_name"],
                "intervention": c["intervention"],
                "cycle": c["cycle"],
                "M_low_Nm": m_low,
                "M_high_Nm": m_high,
                "k_sec_Nm_per_deg": k,
                "valid_secant": bool(np.isfinite(k)),
            }
        )

    cyc = pd.DataFrame(rows)
    summary = (
        cyc.groupby(["file_name", "intervention"], as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            n_valid=("valid_secant", "sum"),
            k_sec_med_Nm_per_deg=("k_sec_Nm_per_deg", "median"),
            k_sec_q25_Nm_per_deg=("k_sec_Nm_per_deg", lambda s: float(np.nanpercentile(s, 25))),
            k_sec_q75_Nm_per_deg=("k_sec_Nm_per_deg", lambda s: float(np.nanpercentile(s, 75))),
        )
    )
    summary["rank_k_sec"] = summary["k_sec_med_Nm_per_deg"].rank(ascending=False, method="dense")
    summary["M_low_Nm"] = m_low
    summary["M_high_Nm"] = m_high
    return summary


def rank_corr_vs_baseline(df: pd.DataFrame, base_map: dict[str, float]) -> float:
    vals = []
    base = []
    for _, r in df.iterrows():
        iv = str(r["intervention"])
        if iv not in base_map:
            continue
        if not np.isfinite(r["rank_k_sec"]):
            continue
        vals.append(float(r["rank_k_sec"]))
        base.append(float(base_map[iv]))
    if len(vals) < 3:
        return np.nan
    return float(np.corrcoef(base, vals)[0, 1])


def window_label(m_low: float, m_high: float) -> str:
    return f"{m_low:.1f}-{m_high:.1f}"


def plot_rank_heatmap(rank_table: pd.DataFrame, out_png: Path) -> None:
    windows = rank_table["window"].drop_duplicates().tolist()
    interventions = rank_table["intervention"].drop_duplicates().tolist()

    mat = np.full((len(interventions), len(windows)), np.nan, dtype=float)
    for i, iv in enumerate(interventions):
        sub = rank_table[rank_table["intervention"] == iv]
        wmap = {str(r.window): float(r.rank_k_sec) for r in sub.itertuples(index=False)}
        for j, w in enumerate(windows):
            mat[i, j] = wmap.get(w, np.nan)

    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(windows)), 5.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("rank (1 = stiffest)")

    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels(windows, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(interventions)))
    ax.set_yticklabels(interventions)
    ax.set_xlabel("secant moment window [Nm]")
    ax.set_ylabel("intervention")
    ax.set_title("Spreader secant rank sensitivity to moment window")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    cases = ordered_cases()
    if not cases:
        raise RuntimeError("No intervention cases resolved from tracker.")

    print("Building cycle curves once, then sweeping windows...")
    all_cycles = []
    for file_name, intervention in cases:
        print(f"=== {file_name} | {intervention} ===")
        all_cycles.extend(build_cycle_curves(file_name, intervention))

    if not all_cycles:
        raise RuntimeError("No spreader cycles available for sensitivity analysis.")

    windows = []
    for lo in LOW_GRID_NM:
        for hi in HIGH_GRID_NM:
            if hi <= lo:
                continue
            windows.append((float(lo), float(hi)))

    summary_rows = []
    all_window_rows = []

    base_map = None
    for lo, hi in windows:
        s = summarize_window(all_cycles, lo, hi).copy()
        s["window"] = window_label(lo, hi)
        all_window_rows.append(s)

        if (lo, hi) == BASELINE_WINDOW:
            base_map = {
                str(r.intervention): float(r.rank_k_sec)
                for r in s.itertuples(index=False)
                if np.isfinite(r.rank_k_sec)
            }

    rank_df = pd.concat(all_window_rows, ignore_index=True)

    if base_map is None:
        raise RuntimeError(f"Baseline window {BASELINE_WINDOW} not found in sweep.")

    for (lo, hi), g in rank_df.groupby(["M_low_Nm", "M_high_Nm"], as_index=False):
        rho = rank_corr_vs_baseline(g, base_map)
        top = g.sort_values("rank_k_sec").iloc[0]["intervention"]
        summary_rows.append(
            {
                "M_low_Nm": float(lo),
                "M_high_Nm": float(hi),
                "window": window_label(lo, hi),
                "rank_corr_vs_baseline_1to4": rho,
                "top_ranked_intervention": str(top),
                "mean_valid_cycles": float(np.nanmean(g["n_valid"] / g["n_cycles"])),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["M_low_Nm", "M_high_Nm"]).reset_index(drop=True)

    out_rank = OUT_DIR / "spreader_secant_window_sensitivity_rankings.csv"
    out_sum = OUT_DIR / "spreader_secant_window_sensitivity_summary.csv"
    out_png = OUT_DIR / "spreader_secant_window_sensitivity_rank_heatmap.png"

    rank_df.to_csv(out_rank, index=False)
    summary_df.to_csv(out_sum, index=False)
    plot_rank_heatmap(rank_df[["window", "intervention", "rank_k_sec"]], out_png)

    print(f"saved {out_rank}")
    print(f"saved {out_sum}")
    print(f"saved {out_png}")

    print("\nTop-ranked intervention by window:")
    print(summary_df[["window", "top_ranked_intervention", "rank_corr_vs_baseline_1to4", "mean_valid_cycles"]].to_string(index=False))


if __name__ == "__main__":
    main()
