"""Pilot feasibility analysis: spreader vs MTS stiffness on one specimen (L231147 L5-S1).

Scope caveat: n = 6 paired points, all sequential destabilization states of ONE specimen.
Results are hypothesis-generating only; they cannot establish proxy validity.
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analyze_mts_flexion_linear_window as mts_flex
import analyze_mts_extension_linear_window as mts_ext
import analyze_mts_testing as mts_core

BASE = Path(__file__).parent
OUT_DIR = BASE / "output"
PILOT_DIR = OUT_DIR / "pilot_spreader_mts"

CSV_TRANSITION = OUT_DIR / "sync_check_transition_flags.csv"
CSV_SPREADER_ROT = OUT_DIR / "moment_theta_cycle_grids" / "all_interventions_moment_theta_metrics.csv"
CSV_SPREADER_LIN = OUT_DIR / "linear_region_stiffness_cycle_grids" / "all_interventions_linear_region_stiffness_cycles.csv"

N_BOOT = 5000
RANDOM_SEED = 12345
ALPHA = 0.05
MTS_SECANT_LOW_NM = 0.0
MTS_SECANT_HIGH_NM = 5.0
MTS_METHOD_LABEL = "Secant 0-5 Nm"

# Canonical condition order (surgical sequence).
CONDITION_ORDER = ["Intact", "PUF Left", "FUF Left", "FBF", "Posterior Release", "SPO"]

# Spreader intervention label -> canonical condition.
SPREADER_TO_CONDITION = {
    "intact, but holding": "Intact",
    "pubf, left": "PUF Left",
    "puf, left": "PUF Left",
    "fuf, left": "FUF Left",
    "fbf": "FBF",
    "posterior release": "Posterior Release",
    "spo": "SPO",
}

SPREADER_METRICS = {
    "k_rot": ("k_linear_nm_per_deg", "Spreader k_rot [Nm/deg]"),
    "k_lin": ("k_linear_n_per_mm", "Spreader k_lin [N/mm]"),
}

# The spreader distracts the posterior column, which is mechanically flexion-like at the segment.
PRIMARY_BRANCH = "Flexion"


def load_transition_flags() -> set[tuple[str, int]]:
    if not CSV_TRANSITION.exists():
        return set()

    df = pd.read_csv(CSV_TRANSITION)
    flagged = df[df["transition_flag"].astype(str).str.lower().isin({"true", "1", "yes"})]
    return {(str(r.file_name), int(r.cycle)) for r in flagged.itertuples()}


def load_spreader_cycles(csv_path: Path, value_col: str, flagged: set[tuple[str, int]]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["condition"] = df["intervention"].astype(str).str.strip().str.lower().map(SPREADER_TO_CONDITION)

    df["is_valid"] = df["valid"].astype(str).str.lower().isin({"true", "1", "yes"})
    df["is_transition"] = [
        (str(fn), int(cy)) in flagged for fn, cy in zip(df["file_name"], df["cycle"])
    ]

    keep = df["is_valid"] & ~df["is_transition"] & df["condition"].notna() & df[value_col].notna()
    out = df.loc[keep, ["condition", "cycle", value_col]].rename(columns={value_col: "value"})
    return out.reset_index(drop=True)


def secant_0to5(theta_deg: np.ndarray, moment_nm: np.ndarray, edge_tol_nm: float = 0.05) -> float:
    theta_deg = np.asarray(theta_deg, dtype=float)
    moment_nm = np.asarray(moment_nm, dtype=float)
    keep = np.isfinite(theta_deg) & np.isfinite(moment_nm)
    theta_deg = theta_deg[keep]
    moment_nm = moment_nm[keep]

    if theta_deg.size < 3:
        return np.nan

    order = np.argsort(moment_nm)
    m_sorted = moment_nm[order]
    t_sorted = theta_deg[order]
    m_unique, first_idx = np.unique(m_sorted, return_index=True)
    t_unique = t_sorted[first_idx]
    if m_unique.size < 2:
        return np.nan

    m_min = float(np.min(m_unique))
    m_max = float(np.max(m_unique))
    if MTS_SECANT_LOW_NM < (m_min - edge_tol_nm) or MTS_SECANT_HIGH_NM > (m_max + edge_tol_nm):
        return np.nan

    t_lo = float(np.interp(np.clip(MTS_SECANT_LOW_NM, m_min, m_max), m_unique, t_unique))
    t_hi = float(np.interp(np.clip(MTS_SECANT_HIGH_NM, m_min, m_max), m_unique, t_unique))
    dtheta = t_hi - t_lo
    if abs(dtheta) < 1e-10:
        return np.nan

    return float(abs((MTS_SECANT_HIGH_NM - MTS_SECANT_LOW_NM) / dtheta))


def bootstrap_mts_secant_ci(
    theta_deg: np.ndarray,
    moment_nm: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    theta_deg = np.asarray(theta_deg, dtype=float)
    moment_nm = np.asarray(moment_nm, dtype=float)
    point = secant_0to5(theta_deg, moment_nm)

    if theta_deg.size < 4:
        return point, np.nan, np.nan, np.array([], dtype=float)

    draws = np.full(N_BOOT, np.nan, dtype=float)
    n = theta_deg.size
    for idx in range(N_BOOT):
        boot_idx = rng.integers(0, n, size=n)
        draws[idx] = secant_0to5(theta_deg[boot_idx], moment_nm[boot_idx])

    finite = draws[np.isfinite(draws)]
    if finite.size < max(200, int(0.2 * N_BOOT)):
        return point, np.nan, np.nan, finite

    lo, hi = np.percentile(finite, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return point, float(lo), float(hi), finite


def build_mts_secant_summary(rng: np.random.Generator) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    books = mts_core.get_book_numbers(mts_core.DATA_DIR)
    intervention_map = mts_core.build_intervention_map(books)

    rows = []
    draws_map: dict[tuple[str, str], np.ndarray] = {}

    for cond in CONDITION_ORDER:
        if cond not in intervention_map:
            continue

        fe_book = int(intervention_map[cond]["Flexion/Extension"])
        flex_res = mts_flex.analyze_intervention(cond, fe_book)
        ext_res = mts_ext.analyze_intervention(cond, fe_book)

        for branch_name, res, mirror_branch in [
            ("Flexion", flex_res, True),
            ("Extension", ext_res, False),
        ]:
            if not bool(res.get("valid", False)):
                rows.append(
                    {
                        "branch": branch_name,
                        "condition": cond,
                        "method": MTS_METHOD_LABEL,
                        "k_mts": np.nan,
                        "k_ci_lo": np.nan,
                        "k_ci_hi": np.nan,
                        "n_branch_points": 0,
                        "valid": False,
                        "reason": str(res.get("reason", "invalid")),
                    }
                )
                draws_map[(branch_name, cond)] = np.array([], dtype=float)
                continue

            theta = np.asarray(res.get("theta_branch", np.array([])), dtype=float)
            moment = np.asarray(res.get("moment_branch", np.array([])), dtype=float)
            if mirror_branch:
                theta = -theta
                moment = -moment

            k_point, k_lo, k_hi, k_draws = bootstrap_mts_secant_ci(theta, moment, rng)
            rows.append(
                {
                    "branch": branch_name,
                    "condition": cond,
                    "method": MTS_METHOD_LABEL,
                    "k_mts": k_point,
                    "k_ci_lo": k_lo,
                    "k_ci_hi": k_hi,
                    "n_branch_points": int(theta.size),
                    "valid": bool(np.isfinite(k_point)),
                    "reason": "ok" if np.isfinite(k_point) else "secant_unavailable",
                }
            )
            draws_map[(branch_name, cond)] = k_draws

    return pd.DataFrame(rows), draws_map


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        return float(values[0]), np.nan, np.nan

    draws = rng.choice(values, size=(N_BOOT, values.size), replace=True)
    medians = np.median(draws, axis=1)
    lo, hi = np.percentile(medians, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(np.median(values)), float(lo), float(hi)


def bootstrap_ratio_ci(
    values: np.ndarray, ref_values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Percent change vs intact, with CI propagated from both conditions."""
    values = np.asarray(values, dtype=float)
    ref_values = np.asarray(ref_values, dtype=float)
    if values.size == 0 or ref_values.size == 0:
        return np.nan, np.nan, np.nan

    point = 100.0 * (np.median(values) - np.median(ref_values)) / np.median(ref_values)
    if values.size < 2 or ref_values.size < 2:
        return float(point), np.nan, np.nan

    a = np.median(rng.choice(values, size=(N_BOOT, values.size), replace=True), axis=1)
    b = np.median(rng.choice(ref_values, size=(N_BOOT, ref_values.size), replace=True), axis=1)
    pct = 100.0 * (a - b) / b
    lo, hi = np.percentile(pct, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(point), float(lo), float(hi)


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Spearman rho with an exact permutation p-value (feasible because n is tiny)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return np.nan, np.nan, n

    rho = float(stats.spearmanr(x, y).statistic)
    if not np.isfinite(rho):
        return np.nan, np.nan, n

    count = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        r = stats.spearmanr(x, y[list(perm)]).statistic
        if np.isfinite(r) and abs(r) >= abs(rho) - 1e-12:
            count += 1
        total += 1

    return rho, count / total, n


def n_for_correlation(rho_true: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Fisher-z sample size for detecting a correlation."""
    if not 0 < abs(rho_true) < 1:
        return np.nan
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return 3.0 + ((z_a + z_b) ** 2) / (np.arctanh(abs(rho_true)) ** 2)


def n_for_paired_effect(d: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Paired t-test sample size for a standardized effect size d."""
    if d <= 0:
        return np.nan
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return ((z_a + z_b) ** 2) / (d**2) + 2.0


def build_spreader_summary(cycles: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for metric_key, df in cycles.items():
        ref = df.loc[df["condition"] == "Intact", "value"].to_numpy()
        for cond in CONDITION_ORDER:
            vals = df.loc[df["condition"] == cond, "value"].to_numpy()
            med, lo, hi = bootstrap_median_ci(vals, rng)
            if cond == "Intact":
                pct, pct_lo, pct_hi = 0.0, 0.0, 0.0
            else:
                pct, pct_lo, pct_hi = bootstrap_ratio_ci(vals, ref, rng)
            rows.append(
                {
                    "metric": metric_key,
                    "condition": cond,
                    "n_cycles_used": int(vals.size),
                    "k_median": med,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "pct_change_vs_intact": pct,
                    "pct_ci_lo": pct_lo,
                    "pct_ci_hi": pct_hi,
                }
            )
    return pd.DataFrame(rows)


def build_mts_effects(
    mts_summary: pd.DataFrame,
    draws_map: dict[tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for (branch, method), grp in mts_summary.groupby(["branch", "method"]):
        lookup = dict(zip(grp["condition"], grp["k_mts"]))
        ci_lo_lookup = dict(zip(grp["condition"], grp["k_ci_lo"]))
        ci_hi_lookup = dict(zip(grp["condition"], grp["k_ci_hi"]))
        ref = lookup.get("Intact", np.nan)
        ref_draws = draws_map.get((branch, "Intact"), np.array([], dtype=float))

        for cond in CONDITION_ORDER:
            k = lookup.get(cond, np.nan)
            if cond == "Intact":
                pct = 0.0
            else:
                pct = 100.0 * (k - ref) / ref if np.isfinite(k) and np.isfinite(ref) and ref != 0 else np.nan

            cond_draws = draws_map.get((branch, cond), np.array([], dtype=float))
            if cond == "Intact":
                pct_lo, pct_hi = 0.0, 0.0
            elif cond_draws.size and ref_draws.size:
                a = cond_draws[rng.integers(0, cond_draws.size, size=N_BOOT)]
                b = ref_draws[rng.integers(0, ref_draws.size, size=N_BOOT)]
                pair_mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
                if np.any(pair_mask):
                    pct_draw = 100.0 * (a[pair_mask] - b[pair_mask]) / b[pair_mask]
                    pct_lo, pct_hi = np.percentile(pct_draw, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
                else:
                    pct_lo, pct_hi = np.nan, np.nan
            else:
                pct_lo, pct_hi = np.nan, np.nan

            rows.append(
                {
                    "branch": branch,
                    "method": method,
                    "condition": cond,
                    "k_mts": k,
                    "k_ci_lo": ci_lo_lookup.get(cond, np.nan),
                    "k_ci_hi": ci_hi_lookup.get(cond, np.nan),
                    "delta_abs": k - ref if np.isfinite(k) and np.isfinite(ref) else np.nan,
                    "pct_change_vs_intact": pct,
                    "pct_ci_lo": float(pct_lo) if np.isfinite(pct_lo) else np.nan,
                    "pct_ci_hi": float(pct_hi) if np.isfinite(pct_hi) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_rank_agreement(spreader_summary: pd.DataFrame, mts_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_key in SPREADER_METRICS:
        sp = spreader_summary[spreader_summary["metric"] == metric_key]
        sp_lookup = dict(zip(sp["condition"], sp["k_median"]))

        for (branch, method), grp in mts_summary.groupby(["branch", "method"]):
            mts_lookup = dict(zip(grp["condition"], grp["k_mts"]))
            conds = [
                c
                for c in CONDITION_ORDER
                if np.isfinite(sp_lookup.get(c, np.nan)) and np.isfinite(mts_lookup.get(c, np.nan))
            ]
            if len(conds) < 3:
                continue

            x = np.array([sp_lookup[c] for c in conds])
            y = np.array([mts_lookup[c] for c in conds])
            rho, p_exact, n = exact_spearman(x, y)
            pearson = stats.pearsonr(x, y) if n >= 3 else None

            rows.append(
                {
                    "spreader_metric": metric_key,
                    "branch": branch,
                    "mts_method": method,
                    "n_pairs": n,
                    "spearman_rho": rho,
                    "p_exact_permutation": p_exact,
                    "pearson_r": float(pearson.statistic) if pearson else np.nan,
                    "branch_mechanically_matched": branch == PRIMARY_BRANCH,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["spreader_metric", "branch", "mts_method"]
    ).reset_index(drop=True)


def build_power_table() -> pd.DataFrame:
    rows = []
    for rho in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        rows.append(
            {
                "target": "correlation",
                "assumed_effect": rho,
                "n_specimens_power80": np.ceil(n_for_correlation(rho, 0.80)),
                "n_specimens_power90": np.ceil(n_for_correlation(rho, 0.90)),
            }
        )
    for d in [0.5, 0.8, 1.0, 1.2, 1.5]:
        rows.append(
            {
                "target": "paired_effect_cohens_d",
                "assumed_effect": d,
                "n_specimens_power80": np.ceil(n_for_paired_effect(d, 0.80)),
                "n_specimens_power90": np.ceil(n_for_paired_effect(d, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(
    spreader_summary: pd.DataFrame,
    mts_effects: pd.DataFrame,
    rank_df: pd.DataFrame,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=180)

    # Panel 1: percent change vs intact.
    ax = axes[0]
    x = np.arange(len(CONDITION_ORDER))
    flex = mts_effects[(mts_effects["branch"] == PRIMARY_BRANCH) & (mts_effects["method"] == MTS_METHOD_LABEL)]
    sub = flex.set_index("condition").reindex(CONDITION_ORDER)
    ax.plot(
        x,
        sub["pct_change_vs_intact"],
        color="#1f77b4",
        marker="o",
        lw=1.8,
        ms=5,
        label=f"MTS {MTS_METHOD_LABEL}",
    )

    sp = spreader_summary[spreader_summary["metric"] == "k_rot"].set_index("condition").reindex(CONDITION_ORDER)
    ax.errorbar(
        x,
        sp["pct_change_vs_intact"],
        yerr=[
            (sp["pct_change_vs_intact"] - sp["pct_ci_lo"]).clip(lower=0).fillna(0),
            (sp["pct_ci_hi"] - sp["pct_change_vs_intact"]).clip(lower=0).fillna(0),
        ],
        color="black",
        marker="s",
        lw=2.0,
        ms=6,
        capsize=4,
        label="Spreader k_rot (95% CI)",
    )
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Change vs Intact [%]")
    ax.set_title(f"Intervention effect ({PRIMARY_BRANCH} branch)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)

    # Panel 2: spreader vs MTS scatter for the mechanically matched branch.
    ax = axes[1]
    matched = rank_df[
        (rank_df["spreader_metric"] == "k_rot")
        & (rank_df["branch"] == PRIMARY_BRANCH)
        & (rank_df["mts_method"] == MTS_METHOD_LABEL)
    ]
    mts_sub = mts_effects[
        (mts_effects["branch"] == PRIMARY_BRANCH) & (mts_effects["method"] == MTS_METHOD_LABEL)
    ].set_index("condition")
    mts_lookup = mts_sub["k_mts"].to_dict()

    sp_lookup = sp["k_median"].to_dict()
    sp_lo_lookup = sp["ci_lo"].to_dict()
    sp_hi_lookup = sp["ci_hi"].to_dict()

    for cond in CONDITION_ORDER:
        if np.isfinite(sp_lookup.get(cond, np.nan)) and np.isfinite(mts_lookup.get(cond, np.nan)):
            x_val = sp_lookup[cond]
            y_val = mts_lookup[cond]
            xerr = np.array(
                [
                    max(0.0, x_val - float(sp_lo_lookup.get(cond, np.nan))),
                    max(0.0, float(sp_hi_lookup.get(cond, np.nan)) - x_val),
                ]
            )
            if np.isfinite(xerr).all():
                ax.errorbar(
                    x_val,
                    y_val,
                    xerr=np.asarray([[xerr[0]], [xerr[1]]]),
                    fmt="o",
                    color="#1f77b4",
                    ecolor="#1f77b4",
                    elinewidth=1.0,
                    capsize=3,
                    ms=6,
                    zorder=3,
                )
            else:
                ax.scatter(x_val, y_val, s=60, zorder=3)

            ax.annotate(cond, (sp_lookup[cond], mts_lookup[cond]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if not matched.empty:
        r = matched.iloc[0]
        ax.set_title(
            f"Spreader k_rot vs MTS {PRIMARY_BRANCH} {MTS_METHOD_LABEL}\n"
            f"rho={r['spearman_rho']:.3f}, exact p={r['p_exact_permutation']:.3f}, n={int(r['n_pairs'])}"
        )
    ax.set_xlabel("Spreader k_rot median [Nm/deg]")
    ax.set_ylabel("MTS Secant 0-5 [Nm/deg]")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "PILOT (n=1 specimen, 6 sequential conditions) - feasibility only, not proxy validation",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def main() -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    flagged = load_transition_flags()
    print(f"transition-flagged cycles excluded: {len(flagged)}")

    cycles = {
        "k_rot": load_spreader_cycles(CSV_SPREADER_ROT, SPREADER_METRICS["k_rot"][0], flagged),
        "k_lin": load_spreader_cycles(CSV_SPREADER_LIN, SPREADER_METRICS["k_lin"][0], flagged),
    }

    mts_summary, mts_draws = build_mts_secant_summary(rng)

    spreader_summary = build_spreader_summary(cycles, rng)
    mts_effects = build_mts_effects(mts_summary, mts_draws, rng)
    rank_df = build_rank_agreement(spreader_summary, mts_summary)
    power_df = build_power_table()

    out_sp = PILOT_DIR / "pilot_spreader_bootstrap_effects.csv"
    out_mts = PILOT_DIR / "pilot_mts_intervention_effects.csv"
    out_rank = PILOT_DIR / "pilot_rank_agreement.csv"
    out_power = PILOT_DIR / "pilot_power_estimates.csv"

    spreader_summary.to_csv(out_sp, index=False)
    mts_effects.to_csv(out_mts, index=False)
    rank_df.to_csv(out_rank, index=False)
    power_df.to_csv(out_power, index=False)
    for p in (out_sp, out_mts, out_rank, out_power):
        print(f"saved {p}")

    plot_summary(spreader_summary, mts_effects, rank_df, PILOT_DIR / "pilot_spreader_mts_summary.png")

    print("\n=== Spreader cycle counts after exclusions ===")
    print(spreader_summary[["metric", "condition", "n_cycles_used"]].to_string(index=False))

    print("\n=== MTS secant 0-5 percent change vs Intact (Flexion branch) ===")
    flex = mts_effects[(mts_effects["branch"] == "Flexion") & (mts_effects["method"] == MTS_METHOD_LABEL)]
    print(
        flex.pivot(index="condition", columns="method", values="pct_change_vs_intact")
        .reindex(CONDITION_ORDER)
        .round(1)
        .to_string()
    )

    print("\n=== MTS secant 0-5 percent change vs Intact (Extension branch) ===")
    ext = mts_effects[(mts_effects["branch"] == "Extension") & (mts_effects["method"] == MTS_METHOD_LABEL)]
    print(
        ext.pivot(index="condition", columns="method", values="pct_change_vs_intact")
        .reindex(CONDITION_ORDER)
        .round(1)
        .to_string()
    )

    print("\n=== MTS intervention-effect definition vs Intact ===")
    print("For each branch and condition c:")
    print("  effect_abs(c) = k_secant_0to5(c) - k_secant_0to5(Intact)")
    print("  effect_pct(c) = 100 * effect_abs(c) / k_secant_0to5(Intact)")
    print("95% CI uses bootstrap resampling of branch (theta, moment) points for both c and Intact.")

    print("\n=== Spreader percent change vs Intact (median, 95% CI) ===")
    for metric_key in SPREADER_METRICS:
        print(f"\n{metric_key}:")
        sub = spreader_summary[spreader_summary["metric"] == metric_key].set_index("condition").reindex(CONDITION_ORDER)
        for cond, r in sub.iterrows():
            print(
                f"  {cond:<20} k={r['k_median']:.3f}  "
                f"delta={r['pct_change_vs_intact']:+.1f}%  "
                f"CI=[{r['pct_ci_lo']:+.1f}, {r['pct_ci_hi']:+.1f}]  n={int(r['n_cycles_used'])}"
            )

    print("\n=== Rank agreement (Spearman, exact permutation p) ===")
    print(
        rank_df[
            [
                "spreader_metric",
                "branch",
                "mts_method",
                "n_pairs",
                "spearman_rho",
                "p_exact_permutation",
                "branch_mechanically_matched",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n=== Sample size for a real validation study ===")
    print(power_df.to_string(index=False))

    print(
        "\nCAVEAT: 6 pairs are sequential destabilization states of ONE specimen."
        "\nOrder effects and release effects are confounded. Treat as feasibility only."
    )


if __name__ == "__main__":
    main()
