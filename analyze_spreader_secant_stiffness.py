#%%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"

# Positive-branch secant window for spreader loading ramps.
SECANT_M_LOW_NM = 1.0
SECANT_M_HIGH_NM = 4.0
SECANT_MIN_DTHETA_DEG = 0.3

# Automatic exclusion gates for noisy/non-physical cycles.
AUTO_GATE_M_LOW_NM = 1.0
AUTO_GATE_M_HIGH_NM = 4.0
AUTO_GATE_RADIUS_STD_MAX_MM = 0.20
AUTO_GATE_Z_STD_MAX_MM = 1.00
AUTO_GATE_MIN_UP_STEPS = 5
AUTO_GATE_OPPOSITE_RATIO_MAX = 0.35
AUTO_GATE_DM_EPS_NM = 1e-6
AUTO_GATE_DTH_EPS_DEG = 1e-6

REMOVE_CYCLE_PRELOAD = True
THETA_REFERENCE = "ramp_start_force_zero"
THETA_REF_WINDOW = 10
FORCE_ZERO_BAND_N = 0.5
FORCE_ZERO_MAX_SAMPLES = 20
FORCE_ZERO_MIN_SAMPLES = 3

# Baseline Intact excluded, Intact Holding retained.
CASE_ORDER = [
    "2 - Intact Holding Longer",
    "3 - PUBF Left",
    "4 - FUF Left",
    "5 - FBF",
    "6 - Posterior Release",
    "7 - SPO",
]


def secant_on_rising_branch(theta_rel_deg: np.ndarray, moment_nm: np.ndarray,
                            m_low: float, m_high: float) -> tuple[float, float, float, float, str]:
    """Compute secant stiffness k = dM/dtheta on rising branch only.

    Returns k_sec [Nm/deg], theta_low [deg], theta_high [deg], dtheta [deg], reason.
    """
    if theta_rel_deg.size < 2 or moment_nm.size < 2:
        return np.nan, np.nan, np.nan, np.nan, "too_few_points"

    m_all = np.asarray(moment_nm, dtype=float)
    a_all = np.asarray(theta_rel_deg, dtype=float)
    keep = np.isfinite(m_all) & np.isfinite(a_all)
    m_all = m_all[keep]
    a_all = a_all[keep]
    if m_all.size < 2:
        return np.nan, np.nan, np.nan, np.nan, "all_nonfinite"

    # Anchor to the final climb that reaches peak moment so m_low is not picked on
    # the early low-moment slack/take-up plateau.
    pk = int(np.nanargmax(m_all))
    if pk < 1:
        return np.nan, np.nan, np.nan, np.nan, "peak_too_early"

    below = np.flatnonzero(m_all[:pk + 1] <= (m_low + 1e-9))
    if below.size == 0:
        return np.nan, np.nan, np.nan, np.nan, "no_mlow_crossing_before_peak"
    i_start = int(below[-1])

    m_seg = m_all[i_start:pk + 1]
    a_seg = a_all[i_start:pk + 1]
    if m_seg.size < 2 or float(np.nanmax(m_seg)) < m_high:
        return np.nan, np.nan, np.nan, np.nan, "mhigh_not_reached_on_final_climb"

    # Keep rising hull so M(theta) is single-valued on the final loading segment.
    hull = m_seg >= np.maximum.accumulate(m_seg) - 1e-9
    m = m_seg[hull]
    a = a_seg[hull]
    if m.size < 2:
        return np.nan, np.nan, np.nan, np.nan, "hull_too_short"

    order = np.argsort(m)
    m = m[order]
    a = a[order]

    m_u, first = np.unique(m, return_index=True)
    a_u = a[first]
    if m_u.size < 2:
        return np.nan, np.nan, np.nan, np.nan, "unique_m_too_short"

    if m_low < float(np.min(m_u)) or m_high > float(np.max(m_u)):
        return np.nan, np.nan, np.nan, np.nan, "window_outside_final_climb"

    th_low = float(np.interp(m_low, m_u, a_u))
    th_high = float(np.interp(m_high, m_u, a_u))
    dtheta = th_high - th_low
    if not np.isfinite(dtheta) or dtheta < SECANT_MIN_DTHETA_DEG:
        return np.nan, th_low, th_high, dtheta, "dtheta_too_small"

    k_sec = float((m_high - m_low) / dtheta)
    return k_sec, th_low, th_high, dtheta, "ok"


def cycle_secant_table(file_name: str, intervention: str) -> pd.DataFrame:
    kin = ls.compute_theta(ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID))
    force = ls.load_force(ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, theta0, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()
    mom = d["moment"].to_numpy()
    t = d["t_sync"].to_numpy()
    rad = d["radius"].to_numpy(float) if "radius" in d.columns else np.full_like(f, np.nan, dtype=float)
    z = d["z_cad"].to_numpy(float) if "z_cad" in d.columns else np.full_like(f, np.nan, dtype=float)

    rows = []
    curves = []
    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    for n, (s, e) in enumerate(runs, 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, pk = rb
        ramp = np.arange(i0, i1)

        # Preload at ramp start is removed so secant uses distraction-only moment.
        f0 = float(f[i0])
        f_ramp_dyn = f[ramp] - f0
        f_peak_dyn = float(f[pk] - f0)
        if f_peak_dyn <= 0:
            continue

        early_n = min(FORCE_ZERO_MAX_SAMPLES, len(ramp))
        early = np.arange(early_n)
        zero_idx = early[np.abs(f_ramp_dyn[early]) <= FORCE_ZERO_BAND_N]

        if zero_idx.size >= FORCE_ZERO_MIN_SAMPLES:
            th_ref = float(np.nanmedian(th[ramp[zero_idx]]))
            theta_ref_window_n = int(zero_idx.size)
            theta_ref_mode = "force_zero_band"
        else:
            iref0 = max(0, i0 - THETA_REF_WINDOW + 1)
            th_ref = float(np.nanmedian(th[iref0:i0 + 1]))
            theta_ref_window_n = int(i0 - iref0 + 1)
            theta_ref_mode = "preramp_median_fallback"

        if not np.isfinite(th_ref):
            th_ref = float(th[i0])
            theta_ref_window_n = 1
            theta_ref_mode = "sample_fallback"

        # With preload removed, reference angle to ramp start (not engagement).
        theta_rel = th[ramp] - th_ref
        m_rel = ls.force_to_moment_nm(f_ramp_dyn)
        k_sec, th_lo, th_hi, dtheta_sec, sec_reason = secant_on_rising_branch(
            theta_rel, m_rel, SECANT_M_LOW_NM, SECANT_M_HIGH_NM
        )

        # Automatic exclusion gates in the 1-4 Nm loading band.
        band = (m_rel >= AUTO_GATE_M_LOW_NM) & (m_rel <= AUTO_GATE_M_HIGH_NM)
        if int(np.count_nonzero(band)) >= 4:
            th_band = theta_rel[band]
            m_band = m_rel[band]
            rad_band = rad[ramp][band]
            z_band = z[ramp][band]
        else:
            th_band = theta_rel
            m_band = m_rel
            rad_band = rad[ramp]
            z_band = z[ramp]

        dm = np.diff(m_band)
        dth = np.diff(th_band)
        up_m = dm > AUTO_GATE_DM_EPS_NM
        down_th = dth < -AUTO_GATE_DTH_EPS_DEG
        n_up = int(np.count_nonzero(up_m))
        n_opp = int(np.count_nonzero(up_m & down_th))
        opp_ratio = float(n_opp / n_up) if n_up > 0 else np.nan
        corr_band = float(np.corrcoef(th_band, m_band)[0, 1]) if th_band.size >= 3 else np.nan

        radius_std = float(np.nanstd(rad_band)) if np.any(np.isfinite(rad_band)) else np.nan
        z_std = float(np.nanstd(z_band)) if np.any(np.isfinite(z_band)) else np.nan

        pass_geom = (
            np.isfinite(radius_std)
            and np.isfinite(z_std)
            and radius_std <= AUTO_GATE_RADIUS_STD_MAX_MM
            and z_std <= AUTO_GATE_Z_STD_MAX_MM
        )
        pass_mono = (
            np.isfinite(opp_ratio)
            and n_up >= AUTO_GATE_MIN_UP_STEPS
            and opp_ratio <= AUTO_GATE_OPPOSITE_RATIO_MAX
        )
        pass_auto = bool(pass_geom and pass_mono)

        fail_parts = []
        if not pass_geom:
            fail_parts.append("geometry_gate")
        if not pass_mono:
            fail_parts.append("monotonicity_gate")
        auto_reason = "ok" if pass_auto else "+".join(fail_parts)

        rows.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "cycle": n,
                "t_start_s": float(t[i0]),
                "t_peak_s": float(t[pk]),
                "lag_s": float(lag),
                "theta0_deg": float(theta0),
                "F_preload_start_N": f0,
                "theta_ref_deg": th_ref,
                "theta_ref_window_n": theta_ref_window_n,
                "theta_ref_mode": theta_ref_mode,
                "F_peak_raw_N": float(f[pk]),
                "M_peak_raw_Nm": float(mom[pk]),
                "F_peak_dyn_N": f_peak_dyn,
                "M_peak_dyn_Nm": float(ls.force_to_moment_nm(f_peak_dyn)),
                "M_low_Nm": float(SECANT_M_LOW_NM),
                "M_high_Nm": float(SECANT_M_HIGH_NM),
                "theta_low_deg": float(th_lo) if np.isfinite(th_lo) else np.nan,
                "theta_high_deg": float(th_hi) if np.isfinite(th_hi) else np.nan,
                "dtheta_secant_deg": float(dtheta_sec) if np.isfinite(dtheta_sec) else np.nan,
                "k_sec_Nm_per_deg": float(k_sec) if np.isfinite(k_sec) else np.nan,
                "secant_reason": sec_reason,
                "valid_secant": bool(np.isfinite(k_sec)),
                "gate_band_n": int(np.count_nonzero(band)),
                "gate_n_up_m": n_up,
                "gate_n_opposite": n_opp,
                "gate_opposite_ratio": opp_ratio,
                "gate_corr_theta_m": corr_band,
                "gate_radius_std_mm": radius_std,
                "gate_z_std_mm": z_std,
                "pass_geometry_gate": bool(pass_geom),
                "pass_monotonicity_gate": bool(pass_mono),
                "pass_auto_exclusion": pass_auto,
                "auto_exclusion_reason": auto_reason,
            }
        )

        curves.append(
            {
                "cycle": n,
                "theta_rel_deg": theta_rel,
                "moment_nm": m_rel,
                "theta_low_deg": float(th_lo) if np.isfinite(th_lo) else np.nan,
                "theta_high_deg": float(th_hi) if np.isfinite(th_hi) else np.nan,
                "dtheta_secant_deg": float(dtheta_sec) if np.isfinite(dtheta_sec) else np.nan,
                "k_sec_Nm_per_deg": float(k_sec) if np.isfinite(k_sec) else np.nan,
                "secant_reason": sec_reason,
            }
        )

    plot_moment_theta_cycles(file_name, intervention, curves)

    return pd.DataFrame(rows)


def rank_corr(summary: pd.DataFrame) -> float:
    """Correlation between ranking order and surgical file order."""
    r = summary.dropna(subset=["k_sec_med_Nm_per_deg"]).sort_values(
        "k_sec_med_Nm_per_deg", ascending=False
    )
    names = [x for x in r["file_name"] if x in CASE_ORDER]
    if len(names) <= 2:
        return np.nan
    x = [CASE_ORDER.index(n) for n in names]
    y = list(range(len(names)))
    return float(np.corrcoef(x, y)[0, 1])


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def plot_moment_theta_cycles(file_name: str, intervention: str, curves: list[dict]) -> None:
    """Save one figure per intervention with one subplot per cycle (M vs theta)."""
    if not curves:
        return

    n = len(curves)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.1 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, curves):
        theta = c["theta_rel_deg"]
        moment = c["moment_nm"]
        ax.plot(theta, moment, lw=1.6, color="tab:blue")

        if np.isfinite(c["theta_low_deg"]) and np.isfinite(c["theta_high_deg"]):
            ax.plot(
                [c["theta_low_deg"], c["theta_high_deg"]],
                [SECANT_M_LOW_NM, SECANT_M_HIGH_NM],
                color="tab:orange",
                lw=2.0,
            )
            ax.scatter(
                [c["theta_low_deg"], c["theta_high_deg"]],
                [SECANT_M_LOW_NM, SECANT_M_HIGH_NM],
                color="tab:orange",
                s=18,
                zorder=3,
            )

        title = (
            f"cycle {c['cycle']}  |  k_sec={c['k_sec_Nm_per_deg']:.2f}"
            if np.isfinite(c["k_sec_Nm_per_deg"])
            else f"cycle {c['cycle']}  |  k_sec=N/A"
        )
        ax.set_title(title, fontsize=9)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.axvline(0.0, color="0.7", lw=0.8)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$\theta_{ramp}$ [deg]")
    for ax in axes[::ncols]:
        ax.set_ylabel(r"preload-corrected pivot moment $\Delta M$ [N$\cdot$m]")

    fig.suptitle(
        f"{file_name} ({intervention}) - preload-corrected moment vs theta by cycle\n"
        f"theta referenced to ramp start, secant window [{SECANT_M_LOW_NM:.1f}, {SECANT_M_HIGH_NM:.1f}] Nm"
    )
    fig.tight_layout()
    out = OUT_DIR / f"spreader_moment_theta_cycles_preloadcorr_{sanitize(file_name)}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def plot_summary(summary: pd.DataFrame, out: Path) -> None:
    df = summary.sort_values("k_sec_med_Nm_per_deg", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    y = df["k_sec_med_Nm_per_deg"].to_numpy(float)
    ylo = y - df["k_sec_q25_Nm_per_deg"].to_numpy(float)
    yhi = df["k_sec_q75_Nm_per_deg"].to_numpy(float) - y

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.bar(x, y, color="tab:blue", alpha=0.85)
    ax.errorbar(x, y, yerr=[ylo, yhi], fmt="none", ecolor="black", capsize=4, lw=1)

    for i, (b, n_valid, n_total) in enumerate(
        zip(bars, df["n_valid"].to_numpy(int), df["n_cycles"].to_numpy(int))
    ):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{n_valid}/{n_total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["intervention"].to_list(), rotation=20, ha="right")
    ax.set_ylabel("secant stiffness k_sec [Nm/deg]")
    ax.set_title(
        f"Spreader secant stiffness ranking (preload-corrected, ramp-start-referenced)\n"
        f"M in [{SECANT_M_LOW_NM:.1f}, {SECANT_M_HIGH_NM:.1f}] Nm; labels = valid/total cycles"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    tracker = ai.tracker_cases()
    tracker = tracker[tracker["File Name"].isin(CASE_ORDER)].copy()

    frames = []
    print("Computing spreader secant stiffness per cycle...")
    if REMOVE_CYCLE_PRELOAD:
        print("Using preload-corrected cycle moment: ΔM = (F - F_start_cycle) * d_o")
    print("Using theta reference: ramp start (theta_ramp), no engagement reference")
    print(
        f"Theta ramp reference: median theta where preload-free force is within "
        f"±{FORCE_ZERO_BAND_N:.1f} N in first {FORCE_ZERO_MAX_SAMPLES} ramp samples "
        f"(fallback: pre-ramp median {THETA_REF_WINDOW} samples)"
    )
    print(f"Secant validity: dtheta >= {SECANT_MIN_DTHETA_DEG:.2f} deg")
    print(
        "Auto-exclusion gates: "
        f"geometry (radius_std<={AUTO_GATE_RADIUS_STD_MAX_MM:.2f} mm, "
        f"z_std<={AUTO_GATE_Z_STD_MAX_MM:.2f} mm) + "
        f"monotonicity in [{AUTO_GATE_M_LOW_NM:.1f}, {AUTO_GATE_M_HIGH_NM:.1f}] Nm "
        f"(n_up>={AUTO_GATE_MIN_UP_STEPS}, opposite_ratio<={AUTO_GATE_OPPOSITE_RATIO_MAX:.2f})"
    )
    for _, r in tracker.iterrows():
        file_name = str(r["File Name"]).strip()
        intervention = str(r["Intervention"]).strip()
        print(f"\n=== {file_name} | {intervention} ===")
        frames.append(cycle_secant_table(file_name, intervention))

    cyc = pd.concat(frames, ignore_index=True)

    summary = (
        cyc.groupby(["file_name", "intervention"], as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            n_valid=("valid_secant", "sum"),
            k_sec_med_Nm_per_deg=("k_sec_Nm_per_deg", "median"),
            k_sec_mean_Nm_per_deg=("k_sec_Nm_per_deg", "mean"),
            k_sec_q25_Nm_per_deg=("k_sec_Nm_per_deg", lambda s: float(np.nanpercentile(s, 25))),
            k_sec_q75_Nm_per_deg=("k_sec_Nm_per_deg", lambda s: float(np.nanpercentile(s, 75))),
            M_peak_dyn_med_Nm=("M_peak_dyn_Nm", "median"),
            F_peak_dyn_max_N=("F_peak_dyn_N", "max"),
            preload_med_N=("F_preload_start_N", "median"),
        )
    )
    summary["rank_k_sec"] = summary["k_sec_med_Nm_per_deg"].rank(ascending=False, method="dense")
    summary = summary.sort_values("rank_k_sec").reset_index(drop=True)

    valid_auto = cyc["valid_secant"] & cyc["pass_auto_exclusion"]
    cyc["k_sec_auto_Nm_per_deg"] = np.where(valid_auto, cyc["k_sec_Nm_per_deg"], np.nan)
    summary_auto = (
        cyc.groupby(["file_name", "intervention"], as_index=False)
        .agg(
            n_cycles=("cycle", "count"),
            n_valid_secant=("valid_secant", "sum"),
            n_pass_auto=("pass_auto_exclusion", "sum"),
            n_valid_auto=("k_sec_auto_Nm_per_deg", lambda s: int(np.isfinite(s).sum())),
            k_sec_med_auto_Nm_per_deg=("k_sec_auto_Nm_per_deg", "median"),
            k_sec_q25_auto_Nm_per_deg=("k_sec_auto_Nm_per_deg", lambda s: float(np.nanpercentile(s, 25))),
            k_sec_q75_auto_Nm_per_deg=("k_sec_auto_Nm_per_deg", lambda s: float(np.nanpercentile(s, 75))),
            gate_fail_geom=("pass_geometry_gate", lambda s: int((~s).sum())),
            gate_fail_mono=("pass_monotonicity_gate", lambda s: int((~s).sum())),
        )
    )
    summary_auto["rank_k_sec_auto"] = summary_auto["k_sec_med_auto_Nm_per_deg"].rank(
        ascending=False, method="dense"
    )
    summary_auto = summary_auto.sort_values("rank_k_sec_auto").reset_index(drop=True)

    rho = rank_corr(summary)
    print("\n" + "=" * 72)
    print("Secant stiffness ranking (higher k_sec = stiffer)")
    print(f"Window: [{SECANT_M_LOW_NM:.1f}, {SECANT_M_HIGH_NM:.1f}] Nm")
    print("Moment basis: preload-corrected per cycle (distraction-only)")
    print(f"Rank correlation vs surgical file order: {rho:+.2f}")
    print("=" * 72)
    print(
        summary[
            [
                "file_name",
                "intervention",
                "n_valid",
                "n_cycles",
                "preload_med_N",
                "k_sec_med_Nm_per_deg",
                "k_sec_q25_Nm_per_deg",
                "k_sec_q75_Nm_per_deg",
                "rank_k_sec",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("Secant stiffness ranking after automatic exclusion gates")
    print(
        f"Gate band: [{AUTO_GATE_M_LOW_NM:.1f}, {AUTO_GATE_M_HIGH_NM:.1f}] Nm | "
        "require pass_geometry_gate AND pass_monotonicity_gate"
    )
    print("=" * 72)
    print(
        summary_auto[
            [
                "file_name",
                "intervention",
                "n_valid_auto",
                "n_cycles",
                "gate_fail_geom",
                "gate_fail_mono",
                "k_sec_med_auto_Nm_per_deg",
                "k_sec_q25_auto_Nm_per_deg",
                "k_sec_q75_auto_Nm_per_deg",
                "rank_k_sec_auto",
            ]
        ].to_string(index=False)
    )

    suffix = "_preloadcorr_rampref_fzero" if REMOVE_CYCLE_PRELOAD and THETA_REFERENCE == "ramp_start_force_zero" else ("_preloadcorr_rampref" if REMOVE_CYCLE_PRELOAD and THETA_REFERENCE == "ramp_start" else ("_preloadcorr" if REMOVE_CYCLE_PRELOAD else ""))
    cyc_path = OUT_DIR / f"spreader_secant_cycles{suffix}.csv"
    sum_path = OUT_DIR / f"spreader_secant_summary{suffix}.csv"
    sum_auto_path = OUT_DIR / f"spreader_secant_summary_autogated{suffix}.csv"
    fig_path = OUT_DIR / f"spreader_secant_ranking{suffix}.png"

    cyc.to_csv(cyc_path, index=False)
    summary.to_csv(sum_path, index=False)
    summary_auto.to_csv(sum_auto_path, index=False)
    plot_summary(summary, fig_path)

    print(f"\nSaved: {cyc_path}")
    print(f"Saved: {sum_path}")
    print(f"Saved: {sum_auto_path}")


if __name__ == "__main__":
    main()
