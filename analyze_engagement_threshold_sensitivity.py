#%%
"""Sensitivity of spreader theta-at-moment results to engagement threshold."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"

# Keep the same case set as analyze_spreader_theta_at_moment.py (no baseline Intact).
CASES = [
    ("2 - Intact Holding Longer", "Intact Holding"),
    ("3 - PUBF Left", "PUBF Left"),
    ("4 - FUF Left", "FUF Left"),
    ("5 - FBF", "FBF"),
    ("6 - Posterior Release", "Posterior Release"),
    ("7 - SPO", "SPO"),
]

SURGICAL_ORDER = [
    "Intact Holding",
    "PUBF Left",
    "FUF Left",
    "FBF",
    "Posterior Release",
    "SPO",
]

TARGET_MOMENTS_NM = (2.0, 3.0, 4.0, 5.0)
HEADLINE_MOMENT_NM = 5.0
THRESHOLDS_N = (8.0, 10.0, 12.0, 14.0, 16.0)


def angle_at_moment(theta: np.ndarray, moment: np.ndarray, target: float) -> float:
    """Angle where the rising branch first reaches `target` moment."""
    if theta.size < 2 or target > np.nanmax(moment) or target < np.nanmin(moment):
        return np.nan
    hull = moment >= np.maximum.accumulate(moment) - 1e-9
    m, a = moment[hull], theta[hull]
    order = np.argsort(m)
    m, a = m[order], a[order]
    m_u, first = np.unique(m, return_index=True)
    if m_u.size < 2:
        return np.nan
    return float(np.interp(target, m_u, a[first]))


def prepare_case(file_name: str) -> tuple[pd.DataFrame, float, list[tuple[int, int]]]:
    """Load and synchronize one case once; reused for all thresholds."""
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, _, fmax = ls.sync(kin, force, lag)
    runs = ls.segment_cycles(d["force"].to_numpy(), ls.CYCLE_THRESH * fmax)
    return d, float(fmax), runs


def case_cycle_table(
    d: pd.DataFrame,
    fmax: float,
    runs: list[tuple[int, int]],
    file_name: str,
    label: str,
    threshold_n: float,
) -> pd.DataFrame:
    """Per-cycle theta@moment metrics for one threshold."""
    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()
    mom = d["moment"].to_numpy()

    rows = []
    for n, (s, e) in enumerate(runs, 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, pk = rb
        ramp = np.arange(i0, i1)

        f_eng = min(threshold_n, 0.60 * f[pk])
        hit = ramp[f[ramp] >= f_eng]
        if len(hit) == 0:
            continue
        iref = int(hit[0])

        row = {
            "threshold_N": threshold_n,
            "file_name": file_name,
            "intervention": label,
            "cycle": n,
            "F_peak_N": float(f[pk]),
            "M_peak_Nm": float(mom[pk]),
            "F_engage_N": float(f_eng),
        }
        for tgt in TARGET_MOMENTS_NM:
            row[f"th_eng@{tgt}"] = angle_at_moment(th[ramp] - th[iref], mom[ramp], tgt)
        rows.append(row)

    return pd.DataFrame(rows)


def rank_corr(sub: pd.DataFrame, col: str) -> float:
    """Correlation between ranked interventions and intended surgical order."""
    r = sub[["intervention", col]].dropna().sort_values(col).reset_index(drop=True)
    ordered = [x for x in r["intervention"] if x in SURGICAL_ORDER]
    if len(ordered) <= 2:
        return np.nan
    x = [SURGICAL_ORDER.index(k) for k in ordered]
    y = list(range(len(ordered)))
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    prepared: dict[str, tuple[pd.DataFrame, float, list[tuple[int, int]]]] = {}
    for file_name, label in CASES:
        print(f"\n=== prepare {file_name} | {label} ===")
        prepared[file_name] = prepare_case(file_name)

    frames = []
    for thr in THRESHOLDS_N:
        print(f"\n--- threshold {thr:.1f} N ---")
        for file_name, label in CASES:
            d, fmax, runs = prepared[file_name]
            frames.append(case_cycle_table(d, fmax, runs, file_name, label, thr))

    cyc = pd.concat(frames, ignore_index=True)
    cyc_path = OUT_DIR / "engagement_threshold_sensitivity_cycles.csv"
    cyc.to_csv(cyc_path, index=False)

    agg = {"cycle": "count", "F_peak_N": "max", "M_peak_Nm": "median", "F_engage_N": "median"}
    for tgt in TARGET_MOMENTS_NM:
        agg[f"th_eng@{tgt}"] = "median"

    summ = (
        cyc.groupby(["threshold_N", "file_name", "intervention"], as_index=False)
        .agg(agg)
        .rename(columns={"cycle": "n_cycles"})
    )

    # coverage per threshold and moment
    for tgt in TARGET_MOMENTS_NM:
        col = f"th_eng@{tgt}"
        cov = (
            cyc.assign(ok=cyc[col].notna())
            .groupby(["threshold_N", "file_name"], as_index=False)["ok"]
            .sum()
            .rename(columns={"ok": f"n_reach@{tgt}"})
        )
        summ = summ.merge(cov, on=["threshold_N", "file_name"], how="left")

    # ranking summaries by threshold
    rank_rows = []
    for thr in THRESHOLDS_N:
        s = summ[summ["threshold_N"] == thr]
        row = {"threshold_N": thr}
        for tgt in TARGET_MOMENTS_NM:
            row[f"rank_corr@{tgt}"] = rank_corr(s, f"th_eng@{tgt}")
        head_col = f"th_eng@{HEADLINE_MOMENT_NM}"
        rr = s[["intervention", head_col]].dropna().sort_values(head_col).reset_index(drop=True)
        row["headline_order"] = " > ".join(rr["intervention"].tolist())
        rank_rows.append(row)

    rank_df = pd.DataFrame(rank_rows)

    summ_path = OUT_DIR / "engagement_threshold_sensitivity_summary.csv"
    rank_path = OUT_DIR / "engagement_threshold_sensitivity_rankcorr.csv"
    summ.to_csv(summ_path, index=False)
    rank_df.to_csv(rank_path, index=False)

    # Plot 1: theta@5Nm vs threshold for each intervention
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    head_col = f"th_eng@{HEADLINE_MOMENT_NM}"
    for intervention, g in summ.groupby("intervention"):
        gg = g.sort_values("threshold_N")
        ax1.plot(gg["threshold_N"], gg[head_col], "-o", lw=1.8, ms=5, label=intervention)
    ax1.set(
        xlabel="engagement threshold [N]",
        ylabel=rf"median $\theta_{{eng}}$ at {HEADLINE_MOMENT_NM:.1f} N·m [deg]",
        title="Threshold sensitivity: theta@5 N·m",
    )
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)
    fig1.tight_layout()
    fig1_path = OUT_DIR / "engagement_threshold_sensitivity_theta5.png"
    fig1.savefig(fig1_path, dpi=160)

    # Plot 2: rank-order correlation vs threshold at each target moment
    fig2, ax2 = plt.subplots(figsize=(9, 5.0))
    for tgt in TARGET_MOMENTS_NM:
        col = f"rank_corr@{tgt}"
        ax2.plot(rank_df["threshold_N"], rank_df[col], "-o", lw=1.8, ms=5, label=f"M={tgt:.1f} N·m")
    ax2.set(
        xlabel="engagement threshold [N]",
        ylabel="rank correlation vs surgical order",
        title="Threshold sensitivity: ranking stability",
        ylim=(-1.05, 1.05),
    )
    ax2.axhline(0.0, color="0.5", ls="--", lw=1)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2_path = OUT_DIR / "engagement_threshold_sensitivity_rankcorr.png"
    fig2.savefig(fig2_path, dpi=160)

    print("\nSaved:")
    print(cyc_path)
    print(summ_path)
    print(rank_path)
    print(fig1_path)
    print(fig2_path)

    print("\nRank-correlation table:")
    print(rank_df.to_string(index=False))


if __name__ == "__main__":
    main()
