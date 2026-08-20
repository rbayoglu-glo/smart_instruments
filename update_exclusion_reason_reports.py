from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_interventions as ai
import analyze_spreader_secant_stiffness as ass
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"
CYCLE_CSV = OUT_DIR / "spreader_secant_cycles_preloadcorr_rampref_fzero.csv"
EXAMPLES_DIR = OUT_DIR / "exclusion_examples"
REASON_CSV = OUT_DIR / "exclusion_reason_counts_by_intervention_1to4.csv"
EXCLUDED_CYCLES_CSV = OUT_DIR / "excluded_cycles_with_reasons_1to4.csv"
ALL_EXCLUDED_DIR = EXAMPLES_DIR / "all_excluded_cycles_1to4"
ALL_EXCLUDED_INDEX_CSV = ALL_EXCLUDED_DIR / "index_all_excluded_cycles_1to4.csv"
MAX_EXAMPLES_PER_REASON = 9


def normalize_exclusion_reason(row: pd.Series) -> str:
    valid_secant = bool(row["valid_secant"])
    pass_auto = bool(row["pass_auto_exclusion"])
    pass_geom = bool(row["pass_geometry_gate"])
    pass_mono = bool(row["pass_monotonicity_gate"])

    if valid_secant and pass_auto:
        return "included"

    if not valid_secant:
        return str(row["secant_reason"])

    if (not pass_geom) and (not pass_mono):
        return "geometry_gate+monotonicity_gate"
    if not pass_geom:
        return "geometry_gate"
    if not pass_mono:
        return "monotonicity_gate"
    return "auto_excluded_unknown"


def exclusion_marks(row: pd.Series) -> str:
    """Attach all failed checks as compact, human-readable reason marks."""
    marks = []
    if not bool(row["valid_secant"]):
        marks.append(f"secant:{row['secant_reason']}")
    if not bool(row["pass_geometry_gate"]):
        marks.append("gate:geometry")
    if not bool(row["pass_monotonicity_gate"]):
        marks.append("gate:monotonicity")
    return ";".join(marks) if marks else "included"


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def build_cycle_curves(file_name: str) -> dict[int, dict]:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, _, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()
    t = d["t_sync"].to_numpy()

    out: dict[int, dict] = {}
    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    for n, (s, e) in enumerate(runs, 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue

        i0, i1, pk = rb
        ramp = np.arange(i0, i1)
        if ramp.size < 4:
            continue

        f0 = float(f[i0])
        f_dyn = f[ramp] - f0
        if float(np.nanmax(f_dyn)) <= 0:
            continue

        early_n = min(ass.FORCE_ZERO_MAX_SAMPLES, len(ramp))
        early = np.arange(early_n)
        zero_idx = early[np.abs(f_dyn[early]) <= ass.FORCE_ZERO_BAND_N]

        if zero_idx.size >= ass.FORCE_ZERO_MIN_SAMPLES:
            th_ref = float(np.nanmedian(th[ramp[zero_idx]]))
        else:
            iref0 = max(0, i0 - ass.THETA_REF_WINDOW + 1)
            th_ref = float(np.nanmedian(th[iref0:i0 + 1]))

        if not np.isfinite(th_ref):
            th_ref = float(th[i0])

        out[int(n)] = {
            "theta_rel": th[ramp] - th_ref,
            "moment_rel": ls.force_to_moment_nm(f_dyn),
            "t_rel": t[ramp] - t[ramp[0]],
        }
    return out


def plot_examples(df: pd.DataFrame, reason_name: str, out_file: Path) -> None:
    if reason_name == "dtheta_too_small":
        sub = df[df["exclusion_reason"] == "dtheta_too_small"].copy()
    else:
        sub = df[df["exclusion_reason"].str.contains("monotonicity_gate", na=False)].copy()

    if sub.empty:
        print(f"No excluded cycles for reason group: {reason_name}")
        return

    sub = sub.sort_values(["intervention", "cycle"]).head(MAX_EXAMPLES_PER_REASON)

    cache: dict[str, dict[int, dict]] = {}
    for fn in sub["file_name"].drop_duplicates().tolist():
        cache[str(fn)] = build_cycle_curves(str(fn))

    n = len(sub)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.7 * nrows), squeeze=False)

    for i, rec in enumerate(sub.to_dict(orient="records")):
        ax = axes[i // ncols][i % ncols]

        fn = str(rec["file_name"])
        cyc = int(rec["cycle"])
        curve = cache.get(fn, {}).get(cyc)

        if curve is None:
            ax.text(0.5, 0.5, "curve unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue

        theta_rel = curve["theta_rel"]
        moment_rel = curve["moment_rel"]

        ax.plot(theta_rel, moment_rel, color="tab:blue", lw=1.4)
        ax.axhline(ass.SECANT_M_LOW_NM, color="red", ls="--", lw=1.0)
        ax.axhline(ass.SECANT_M_HIGH_NM, color="red", ls="--", lw=1.0)

        th_lo = rec.get("theta_low_deg")
        th_hi = rec.get("theta_high_deg")
        if pd.notna(th_lo) and pd.notna(th_hi):
            ax.plot([th_lo, th_hi], [ass.SECANT_M_LOW_NM, ass.SECANT_M_HIGH_NM], color="tab:orange", lw=2.0)
            ax.scatter([th_lo, th_hi], [ass.SECANT_M_LOW_NM, ass.SECANT_M_HIGH_NM], color="tab:orange", s=16, zorder=3)

        ax.set_title(
            f"{rec['intervention']} | cycle {cyc}\n"
            f"{rec['exclusion_reason']} | dtheta={rec.get('dtheta_secant_deg', np.nan):.3f}",
            fontsize=9,
        )
        ax.set_xlabel("theta_rel [deg]")
        ax.set_ylabel("moment_rel [Nm]")
        ax.grid(alpha=0.25)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Excluded cycle examples: {reason_name} (secant window {ass.SECANT_M_LOW_NM:.1f}-{ass.SECANT_M_HIGH_NM:.1f} Nm)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)
    print(f"saved {out_file}")


def save_all_excluded_cycle_plots(df_excl: pd.DataFrame) -> pd.DataFrame:
    """Save one marked moment-theta plot for each excluded cycle."""
    ALL_EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)

    cache: dict[str, dict[int, dict]] = {}
    for fn in df_excl["file_name"].drop_duplicates().tolist():
        cache[str(fn)] = build_cycle_curves(str(fn))

    rows = []
    for rec in df_excl.sort_values(["case_order", "intervention", "cycle"]).to_dict(orient="records"):
        fn = str(rec["file_name"])
        intervention = str(rec["intervention"])
        cyc = int(rec["cycle"])
        curve = cache.get(fn, {}).get(cyc)
        if curve is None:
            continue

        theta_rel = curve["theta_rel"]
        moment_rel = curve["moment_rel"]

        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.plot(theta_rel, moment_rel, color="tab:blue", lw=1.6, label="cycle trace")
        ax.axhline(ass.SECANT_M_LOW_NM, color="red", ls="--", lw=1.0)
        ax.axhline(ass.SECANT_M_HIGH_NM, color="red", ls="--", lw=1.0)

        th_lo = rec.get("theta_low_deg")
        th_hi = rec.get("theta_high_deg")
        if pd.notna(th_lo) and pd.notna(th_hi):
            ax.plot([th_lo, th_hi], [ass.SECANT_M_LOW_NM, ass.SECANT_M_HIGH_NM], color="tab:orange", lw=2.1, label="secant")
            ax.scatter([th_lo, th_hi], [ass.SECANT_M_LOW_NM, ass.SECANT_M_HIGH_NM], color="tab:orange", s=18, zorder=3)

        reason_box = (
            f"primary={rec['exclusion_reason']}\n"
            f"marks={rec['exclusion_reason_marks']}\n"
            f"secant_reason={rec['secant_reason']}\n"
            f"dtheta={rec.get('dtheta_secant_deg', np.nan):.3f} deg | "
            f"opp_ratio={rec.get('gate_opposite_ratio', np.nan):.3f}"
        )
        ax.text(
            0.02,
            0.98,
            reason_box,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.6,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.6"},
        )

        ax.set_title(f"{intervention} | cycle {cyc} | excluded", fontsize=10)
        ax.set_xlabel("theta_rel [deg]")
        ax.set_ylabel("moment_rel [Nm]")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()

        out_name = (
            f"{sanitize(intervention)}_cycle_{cyc:02d}_"
            f"{sanitize(str(rec['exclusion_reason']))}.png"
        )
        out_path = ALL_EXCLUDED_DIR / out_name
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

        rows.append(
            {
                "file_name": fn,
                "intervention": intervention,
                "cycle": cyc,
                "exclusion_reason": rec["exclusion_reason"],
                "exclusion_reason_marks": rec["exclusion_reason_marks"],
                "plot_file": str(out_path),
            }
        )

    out_idx = pd.DataFrame(rows)
    out_idx.to_csv(ALL_EXCLUDED_INDEX_CSV, index=False)
    print(f"saved {ALL_EXCLUDED_INDEX_CSV}")
    return out_idx


def main() -> None:
    if not CYCLE_CSV.exists():
        raise FileNotFoundError(f"Cycle summary not found: {CYCLE_CSV}")

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    cyc = pd.read_csv(CYCLE_CSV)
    cyc["exclusion_reason"] = cyc.apply(normalize_exclusion_reason, axis=1)
    cyc["exclusion_reason_marks"] = cyc.apply(exclusion_marks, axis=1)
    cyc["excluded"] = cyc["exclusion_reason"] != "included"

    cyc["mark_dtheta_too_small"] = cyc["secant_reason"].eq("dtheta_too_small")
    cyc["mark_mhigh_not_reached"] = cyc["secant_reason"].eq("mhigh_not_reached_on_final_climb")
    cyc["mark_geometry_gate"] = ~cyc["pass_geometry_gate"].astype(bool)
    cyc["mark_monotonicity_gate"] = ~cyc["pass_monotonicity_gate"].astype(bool)

    order_map = {name: i for i, name in enumerate(ass.CASE_ORDER)}
    cyc["case_order"] = cyc["file_name"].map(order_map).fillna(999).astype(int)

    excluded = cyc[cyc["excluded"]].copy().sort_values(["case_order", "intervention", "cycle"])
    excluded_export_cols = [
        "file_name",
        "intervention",
        "cycle",
        "excluded",
        "exclusion_reason",
        "exclusion_reason_marks",
        "secant_reason",
        "auto_exclusion_reason",
        "valid_secant",
        "pass_auto_exclusion",
        "pass_geometry_gate",
        "pass_monotonicity_gate",
        "mark_dtheta_too_small",
        "mark_mhigh_not_reached",
        "mark_geometry_gate",
        "mark_monotonicity_gate",
        "dtheta_secant_deg",
        "theta_low_deg",
        "theta_high_deg",
        "k_sec_Nm_per_deg",
        "gate_opposite_ratio",
        "gate_n_up_m",
        "gate_n_opposite",
        "gate_radius_std_mm",
        "gate_z_std_mm",
        "M_low_Nm",
        "M_high_Nm",
    ]
    excluded[excluded_export_cols].to_csv(EXCLUDED_CYCLES_CSV, index=False)
    print(f"saved {EXCLUDED_CYCLES_CSV}")

    counts = (
        cyc.groupby(["case_order", "intervention", "exclusion_reason"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    counts_pivot = (
        counts.pivot_table(
            index=["case_order", "intervention"],
            columns="exclusion_reason",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
        .sort_values("case_order")
    )
    counts_pivot["n_cycles"] = cyc.groupby(["case_order", "intervention"]).size().to_numpy()
    counts_pivot["n_excluded"] = counts_pivot["n_cycles"] - counts_pivot.get("included", 0)

    out_cols = [c for c in counts_pivot.columns if c not in {"case_order"}]
    counts_out = counts_pivot[out_cols]
    counts_out.to_csv(REASON_CSV, index=False)

    print("Exclusion reasons per intervention (1-4 Nm):")
    print(counts_out.to_string(index=False))
    print(f"saved {REASON_CSV}")

    plot_examples(cyc, "dtheta_too_small", EXAMPLES_DIR / "excluded_examples_dtheta_too_small_1to4.png")
    plot_examples(cyc, "monotonicity_gate", EXAMPLES_DIR / "excluded_examples_monotonicity_gate_1to4.png")
    all_idx = save_all_excluded_cycle_plots(excluded)
    print(f"saved {len(all_idx)} excluded-cycle plots in {ALL_EXCLUDED_DIR}")


if __name__ == "__main__":
    main()
