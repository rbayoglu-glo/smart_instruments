from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_mts_testing as mts


OUT_DIR = Path(__file__).parent / "output"
M_LOW_NM = 0.0
M_HIGH_NM = 5.0
MAX_MOMENT_NM = float(mts.TARGET_MOMENT_NM)
WINDOW_TAG = f"{int(M_LOW_NM)}to{int(M_HIGH_NM)}"
COMPARE_LINEAR_LOW_NM = 1.0
COMPARE_LINEAR_HIGH_NM = 5.0
COMPARE_CHORD_LOW_NM = 1.0
COMPARE_CHORD_HIGH_NM = 5.0
COMPARE_SECANT_0TO5_LOW_NM = 0.0
COMPARE_SECANT_0TO5_HIGH_NM = 5.0
COMPARE_SECANT_0TO4_LOW_NM = 0.0
COMPARE_SECANT_0TO4_HIGH_NM = 4.0

CASE_CONFIGS = [
    {"motion": "Lateral Bending", "branch_sign": 1.0, "branch_label": "positive"},
    {"motion": "Lateral Bending", "branch_sign": -1.0, "branch_label": "negative_mirrored"},
    {"motion": "Axial Rotation", "branch_sign": 1.0, "branch_label": "positive"},
    {"motion": "Axial Rotation", "branch_sign": -1.0, "branch_label": "negative_mirrored"},
]


def slugify(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def interpolate_theta_at_moment(
    theta_deg: np.ndarray,
    moment_nm: np.ndarray,
    target_moment_nm: float,
    edge_tol_nm: float = 0.05,
) -> float:
    if theta_deg.size < 2 or moment_nm.size < 2:
        return float("nan")

    order = np.argsort(moment_nm)
    m_sorted = moment_nm[order]
    t_sorted = theta_deg[order]

    m_unique, first_idx = np.unique(m_sorted, return_index=True)
    t_unique = t_sorted[first_idx]
    if m_unique.size < 2:
        return float("nan")

    m_min = float(np.min(m_unique))
    m_max = float(np.max(m_unique))
    if target_moment_nm < (m_min - edge_tol_nm) or target_moment_nm > (m_max + edge_tol_nm):
        return float("nan")

    target_clamped = float(np.clip(target_moment_nm, m_min, m_max))
    return float(np.interp(target_clamped, m_unique, t_unique))


def fit_linear_stiffness_on_window(
    theta_deg: np.ndarray,
    moment_nm: np.ndarray,
    m_low_nm: float,
    m_high_nm: float,
    min_points: int = 6,
) -> tuple[float, float, int]:
    mask = (
        np.isfinite(theta_deg)
        & np.isfinite(moment_nm)
        & (moment_nm >= m_low_nm)
        & (moment_nm <= m_high_nm)
    )
    idx = np.flatnonzero(mask)
    if idx.size < min_points:
        return np.nan, np.nan, int(idx.size)

    t_sel = theta_deg[idx]
    m_sel = moment_nm[idx]
    slope, _ = np.polyfit(t_sel, m_sel, 1)
    return float(abs(slope)), float(slope), int(idx.size)


def compute_two_point_stiffness(
    theta_deg: np.ndarray,
    moment_nm: np.ndarray,
    m_low_nm: float,
    m_high_nm: float,
) -> tuple[float, float, float]:
    theta_low = interpolate_theta_at_moment(theta_deg, moment_nm, m_low_nm)
    theta_high = interpolate_theta_at_moment(theta_deg, moment_nm, m_high_nm)

    if not np.isfinite(theta_low) or not np.isfinite(theta_high):
        return np.nan, np.nan, np.nan

    dtheta = float(theta_high - theta_low)
    if abs(dtheta) < 1e-10:
        return np.nan, theta_low, theta_high

    k_abs = float(abs((m_high_nm - m_low_nm) / dtheta))
    return k_abs, theta_low, theta_high


def extract_centered_branch(
    angle_cycle_deg: np.ndarray,
    moment_cycle_nm: np.ndarray,
    branch_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    m_avg, a_avg, _, _ = mts.compute_centered_average_curve(angle_cycle_deg, moment_cycle_nm)
    if m_avg.size < 3 or a_avg.size < 3:
        return np.array([]), np.array([]), a_avg, m_avg, "centered_curve_too_short"

    # branch_sign = +1 keeps the positive branch; branch_sign = -1 mirrors the
    # negative branch into positive axes so method windows remain identical.
    a_branch_space = branch_sign * a_avg
    m_branch_space = branch_sign * m_avg

    mask = (
        np.isfinite(a_branch_space)
        & np.isfinite(m_branch_space)
        & (a_branch_space >= 0.0)
        & (m_branch_space >= 0.0)
        & (m_branch_space <= (MAX_MOMENT_NM + 1e-9))
    )
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        return np.array([]), np.array([]), a_branch_space, m_branch_space, "no_valid_branch"

    breaks = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, breaks)
    run = max(runs, key=lambda r: r.size)
    if run.size < 3:
        return np.array([]), np.array([]), a_branch_space, m_branch_space, "branch_too_short"

    theta_branch = a_branch_space[run]
    moment_branch = m_branch_space[run]
    return theta_branch, moment_branch, a_branch_space, m_branch_space, "ok"


def analyze_intervention(intervention: str, book_num: int, motion_name: str, branch_sign: float) -> dict:
    file_path = mts.DATA_DIR / f"Book{book_num}.xlsx"
    angle, moment = mts.read_motion_data(file_path, mts.MOTION_CONFIG[motion_name]["moment_col"])
    _, _, angle_cycle, moment_cycle = mts.extract_third_cycle(angle, moment)

    theta_branch, moment_branch, theta_avg, moment_avg, branch_reason = extract_centered_branch(
        angle_cycle,
        moment_cycle,
        branch_sign,
    )

    if branch_reason != "ok":
        return {
            "Intervention": intervention,
            "Book": book_num,
            "Motion": motion_name,
            "valid": False,
            "reason": branch_reason,
            "n_points_window": 0,
            "k_signed_nm_per_deg": np.nan,
            "k_abs_nm_per_deg": np.nan,
            "intercept_nm": np.nan,
            "r2": np.nan,
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "theta_sel": np.array([]),
            "moment_sel": np.array([]),
            "theta_branch": theta_branch,
            "moment_branch": moment_branch,
            "theta_avg_centered": theta_avg,
            "moment_avg_centered": moment_avg,
            "k_linear_1to5_nm_per_deg": np.nan,
            "k_chord_1to5_nm_per_deg": np.nan,
            "k_secant_0to5_nm_per_deg": np.nan,
            "k_secant_0to4_nm_per_deg": np.nan,
        }

    k_linear_1to5, _, _ = fit_linear_stiffness_on_window(
        theta_branch,
        moment_branch,
        COMPARE_LINEAR_LOW_NM,
        COMPARE_LINEAR_HIGH_NM,
    )
    k_chord_1to5, _, _ = compute_two_point_stiffness(
        theta_branch,
        moment_branch,
        COMPARE_CHORD_LOW_NM,
        COMPARE_CHORD_HIGH_NM,
    )
    k_secant_0to5, _, _ = compute_two_point_stiffness(
        theta_branch,
        moment_branch,
        COMPARE_SECANT_0TO5_LOW_NM,
        COMPARE_SECANT_0TO5_HIGH_NM,
    )
    k_secant_0to4, _, _ = compute_two_point_stiffness(
        theta_branch,
        moment_branch,
        COMPARE_SECANT_0TO4_LOW_NM,
        COMPARE_SECANT_0TO4_HIGH_NM,
    )

    theta_m_low = interpolate_theta_at_moment(theta_branch, moment_branch, M_LOW_NM)
    theta_m_high = interpolate_theta_at_moment(theta_branch, moment_branch, M_HIGH_NM)
    if not np.isfinite(theta_m_low) or not np.isfinite(theta_m_high):
        return {
            "Intervention": intervention,
            "Book": book_num,
            "Motion": motion_name,
            "valid": False,
            "reason": f"missing_point_at_{WINDOW_TAG}_nm",
            "n_points_window": 0,
            "k_signed_nm_per_deg": np.nan,
            "k_abs_nm_per_deg": np.nan,
            "intercept_nm": np.nan,
            "r2": np.nan,
            "theta_at_m_low_deg": np.nan,
            "theta_at_m_high_deg": np.nan,
            "theta_sel": np.array([]),
            "moment_sel": np.array([]),
            "theta_branch": theta_branch,
            "moment_branch": moment_branch,
            "theta_avg_centered": theta_avg,
            "moment_avg_centered": moment_avg,
            "k_linear_1to5_nm_per_deg": k_linear_1to5,
            "k_chord_1to5_nm_per_deg": k_chord_1to5,
            "k_secant_0to5_nm_per_deg": k_secant_0to5,
            "k_secant_0to4_nm_per_deg": k_secant_0to4,
        }

    dtheta = float(theta_m_high - theta_m_low)
    if abs(dtheta) < 1e-10:
        return {
            "Intervention": intervention,
            "Book": book_num,
            "Motion": motion_name,
            "valid": False,
            "reason": f"zero_delta_theta_between_m{int(M_LOW_NM)}_and_m{int(M_HIGH_NM)}",
            "n_points_window": 0,
            "k_signed_nm_per_deg": np.nan,
            "k_abs_nm_per_deg": np.nan,
            "intercept_nm": np.nan,
            "r2": np.nan,
            "theta_at_m_low_deg": theta_m_low,
            "theta_at_m_high_deg": theta_m_high,
            "theta_sel": np.array([]),
            "moment_sel": np.array([]),
            "theta_branch": theta_branch,
            "moment_branch": moment_branch,
            "theta_avg_centered": theta_avg,
            "moment_avg_centered": moment_avg,
            "k_linear_1to5_nm_per_deg": k_linear_1to5,
            "k_chord_1to5_nm_per_deg": k_chord_1to5,
            "k_secant_0to5_nm_per_deg": k_secant_0to5,
            "k_secant_0to4_nm_per_deg": k_secant_0to4,
        }

    slope = float((M_HIGH_NM - M_LOW_NM) / dtheta)
    intercept = float(M_LOW_NM - slope * theta_m_low)
    theta_sel = np.asarray([theta_m_low, theta_m_high], dtype=float)
    moment_sel = np.asarray([M_LOW_NM, M_HIGH_NM], dtype=float)

    return {
        "Intervention": intervention,
        "Book": book_num,
        "Motion": motion_name,
        "valid": True,
        "reason": "ok",
        "n_points_window": int(theta_sel.size),
        "k_signed_nm_per_deg": slope,
        "k_abs_nm_per_deg": abs(slope),
        "intercept_nm": intercept,
        "r2": np.nan,
        "theta_at_m_low_deg": theta_m_low,
        "theta_at_m_high_deg": theta_m_high,
        "theta_min_deg": float(np.min(theta_sel)),
        "theta_max_deg": float(np.max(theta_sel)),
        "moment_min_nm": float(np.min(moment_sel)),
        "moment_max_nm": float(np.max(moment_sel)),
        "theta_sel": theta_sel,
        "moment_sel": moment_sel,
        "theta_branch": theta_branch,
        "moment_branch": moment_branch,
        "theta_avg_centered": theta_avg,
        "moment_avg_centered": moment_avg,
        "k_linear_1to5_nm_per_deg": k_linear_1to5,
        "k_chord_1to5_nm_per_deg": k_chord_1to5,
        "k_secant_0to5_nm_per_deg": k_secant_0to5,
        "k_secant_0to4_nm_per_deg": k_secant_0to4,
    }


def plot_fit_panels(results: list[dict], out_png: Path, title_text: str) -> None:
    n = len(results)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.2 * nrows), squeeze=False)

    for i, rec in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        theta_b = rec["theta_branch"]
        moment_b = rec["moment_branch"]
        theta_avg = rec["theta_avg_centered"]
        moment_avg = rec["moment_avg_centered"]

        if np.asarray(moment_avg).size and np.asarray(theta_avg).size:
            ax.plot(theta_avg, moment_avg, color="0.85", lw=1.0, ls="--", label="avg(load/unload), centered")

        ax.plot(theta_b, moment_b, color="0.80", lw=1.2, label="selected branch")

        if rec["moment_sel"].size:
            ax.scatter(
                rec["theta_sel"],
                rec["moment_sel"],
                s=28,
                color="tab:red",
                alpha=0.85,
                edgecolors="none",
                label=f"two points: {int(M_LOW_NM)} and {int(M_HIGH_NM)} Nm",
            )

        if rec["valid"]:
            theta_sec = np.asarray(rec["theta_sel"], dtype=float)
            moment_sec = np.asarray(rec["moment_sel"], dtype=float)
            ax.plot(
                theta_sec,
                moment_sec,
                color="tab:red",
                lw=1.3,
                ls=":",
                label=f"secant ({int(M_LOW_NM)} to {int(M_HIGH_NM)} Nm)",
            )
            title = f"{rec['Intervention']} | k_sec={rec['k_abs_nm_per_deg']:.3f} Nm/deg"
        else:
            title = f"{rec['Intervention']} | N/A ({rec['reason']})"

        ax.axhline(M_LOW_NM, color="0.5", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(M_HIGH_NM, color="0.5", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("theta_rel [deg]")
        ax.set_ylabel("moment_rel [Nm]")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved {out_png}")


def plot_bar(df: pd.DataFrame, out_png: Path, title_text: str) -> None:
    d = df.copy()
    d = d[d["valid"]].reset_index(drop=True)
    if d.empty:
        print("skip bar chart: no valid interventions")
        return

    x = np.arange(len(d))
    y = d["k_abs_nm_per_deg"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=180)
    bars = ax.bar(x, y, color="#4C78A8", edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(d["Intervention"].astype(str), rotation=18, ha="right")
    ax.set_ylabel("Stiffness [Nm/deg]")
    ax.set_title(title_text)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{y[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def plot_method_comparison_bar(df: pd.DataFrame, out_png: Path, title_text: str) -> None:
    d = df.copy()
    methods = [
        ("k_secant_0to4_nm_per_deg", "Secant 0-4 Nm"),
        ("k_secant_0to5_nm_per_deg", "Secant 0-5 Nm"),
        ("k_chord_1to5_nm_per_deg", "Chord 1-5 Nm"),
        ("k_linear_1to5_nm_per_deg", "Linear fit 1-5 Nm"),
    ]
    method_cols = [col for col, _ in methods]
    d = d[d[method_cols].notna().any(axis=1)].reset_index(drop=True)
    if d.empty:
        print("skip comparison bar chart: no valid interventions")
        return

    interventions = d["Intervention"].astype(str).tolist()
    n_methods = len(methods)
    n_interventions = len(interventions)

    x = np.arange(n_methods)
    width = min(0.13, 0.8 / max(n_interventions, 1))
    offsets = (np.arange(n_interventions, dtype=float) - (n_interventions - 1) / 2.0) * width
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756", "#9D755D", "#B279A2"]

    fig, ax = plt.subplots(figsize=(12.0, 5.8), dpi=180)

    for i, intervention in enumerate(interventions):
        color = colors[i % len(colors)]
        y = np.array([float(d.loc[i, col]) for col, _ in methods], dtype=float)
        bars = ax.bar(
            x + offsets[i],
            y,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            label=intervention,
        )
        for j, b in enumerate(bars):
            if np.isfinite(y[j]):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.015,
                    f"{y[j]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in methods], rotation=0, ha="center")
    ax.set_ylabel("Stiffness [Nm/deg]")
    ax.set_title(title_text)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def run_case(interventions: dict[str, dict[str, int]], motion_name: str, branch_sign: float, branch_label: str) -> None:
    motion_slug = slugify(motion_name)
    case_slug = f"mts_{motion_slug}_{branch_label}"
    branch_desc = "positive" if branch_sign > 0 else "negative (mirrored)"

    results = []
    for intervention, book_map in interventions.items():
        rec = analyze_intervention(intervention, int(book_map[motion_name]), motion_name, branch_sign)
        results.append(rec)

    title_base = f"MTS {motion_name}: {branch_desc} branch"

    out_fit = OUT_DIR / f"{case_slug}_{WINDOW_TAG}_secant_fit_panels.png"
    plot_fit_panels(
        results,
        out_fit,
        f"{title_base} two-point secant stiffness ({int(M_LOW_NM)} and {int(M_HIGH_NM)} Nm)",
    )

    rows = []
    for r in results:
        rows.append(
            {
                "Intervention": r["Intervention"],
                "Book": r["Book"],
                "Motion": r["Motion"],
                "BranchLabel": branch_label,
                "BranchSign": branch_sign,
                "valid": bool(r["valid"]),
                "reason": str(r["reason"]),
                "n_points_window": int(r["n_points_window"]),
                "k_signed_nm_per_deg": float(r["k_signed_nm_per_deg"]) if np.isfinite(r["k_signed_nm_per_deg"]) else np.nan,
                "k_abs_nm_per_deg": float(r["k_abs_nm_per_deg"]) if np.isfinite(r["k_abs_nm_per_deg"]) else np.nan,
                "intercept_nm": float(r["intercept_nm"]) if np.isfinite(r["intercept_nm"]) else np.nan,
                "r2": float(r["r2"]) if np.isfinite(r["r2"]) else np.nan,
                "theta_at_m_low_deg": float(r.get("theta_at_m_low_deg", np.nan)),
                "theta_at_m_high_deg": float(r.get("theta_at_m_high_deg", np.nan)),
                "theta_min_deg": float(r.get("theta_min_deg", np.nan)),
                "theta_max_deg": float(r.get("theta_max_deg", np.nan)),
                "moment_min_nm": float(r.get("moment_min_nm", np.nan)),
                "moment_max_nm": float(r.get("moment_max_nm", np.nan)),
                "k_linear_1to5_nm_per_deg": float(r.get("k_linear_1to5_nm_per_deg", np.nan)),
                "k_chord_1to5_nm_per_deg": float(r.get("k_chord_1to5_nm_per_deg", np.nan)),
                "k_secant_0to5_nm_per_deg": float(r.get("k_secant_0to5_nm_per_deg", np.nan)),
                "k_secant_0to4_nm_per_deg": float(r.get("k_secant_0to4_nm_per_deg", np.nan)),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"{case_slug}_{WINDOW_TAG}_secant_stiffness.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    out_bar = OUT_DIR / f"{case_slug}_{WINDOW_TAG}_secant_stiffness_bar.png"
    plot_bar(df, out_bar, f"{title_base} two-point secant stiffness ({int(M_LOW_NM)} to {int(M_HIGH_NM)} Nm)")

    out_compare_bar = OUT_DIR / f"{case_slug}_stiffness_four_methods_bar.png"
    plot_method_comparison_bar(df, out_compare_bar, f"MTS {motion_name}: stiffness comparison grouped by method ({branch_desc})")

    valid = df[df["valid"]].copy()
    if not valid.empty:
        valid = valid.sort_values("k_abs_nm_per_deg", ascending=False).reset_index(drop=True)
        print(f"\n{motion_name} | {branch_desc} | ranking ({int(M_LOW_NM)}-{int(M_HIGH_NM)} Nm two-point secant):")
        print(valid[["Intervention", "k_abs_nm_per_deg", "theta_at_m_low_deg", "theta_at_m_high_deg"]].to_string(index=False))

        print("\nFour-method comparison (Nm/deg):")
        print(
            valid[
                [
                    "Intervention",
                    "k_linear_1to5_nm_per_deg",
                    "k_chord_1to5_nm_per_deg",
                    "k_secant_0to5_nm_per_deg",
                    "k_secant_0to4_nm_per_deg",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    books = mts.get_book_numbers(mts.DATA_DIR)
    interventions = mts.build_intervention_map(books)

    for case in CASE_CONFIGS:
        run_case(
            interventions,
            motion_name=str(case["motion"]),
            branch_sign=float(case["branch_sign"]),
            branch_label=str(case["branch_label"]),
        )


if __name__ == "__main__":
    main()
