from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_mts_testing as mts


OUT_DIR = Path(__file__).parent / "output"
MOTION_NAME = "Flexion/Extension"
M_LOW_NM = 1.0
M_HIGH_NM = 5.0
MIN_POINTS = 6
MAX_MOMENT_NM = float(mts.TARGET_MOMENT_NM)
WINDOW_TAG = f"{int(M_LOW_NM)}to{int(M_HIGH_NM)}"


def fit_line(theta_deg: np.ndarray, moment_nm: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(theta_deg, moment_nm, 1)
    pred = slope * theta_deg + intercept
    ss_res = float(np.sum((moment_nm - pred) ** 2))
    ss_tot = float(np.sum((moment_nm - np.mean(moment_nm)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def extract_centered_positive_flexion_branch(
    angle_cycle_deg: np.ndarray,
    moment_cycle_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Use the same centered branch basis as the prior MTS flexion plot."""
    m_avg, a_avg, _, _ = mts.compute_centered_average_curve(angle_cycle_deg, moment_cycle_nm)
    if m_avg.size < 3 or a_avg.size < 3:
        return np.array([]), np.array([]), a_avg, m_avg, "centered_curve_too_short"

    mask = (
        np.isfinite(a_avg)
        & np.isfinite(m_avg)
        & (a_avg >= 0.0)
        & (m_avg >= 0.0)
        & (m_avg <= (MAX_MOMENT_NM + 1e-9))
    )
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        return np.array([]), np.array([]), a_avg, m_avg, "no_positive_flexion_branch"

    # Keep the largest contiguous run on the positive flexion branch.
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, breaks)
    run = max(runs, key=lambda r: r.size)
    if run.size < 3:
        return np.array([]), np.array([]), a_avg, m_avg, "positive_branch_too_short"

    theta_branch = a_avg[run]
    moment_branch = m_avg[run]
    return theta_branch, moment_branch, a_avg, m_avg, "ok"


def analyze_intervention(intervention: str, book_num: int) -> dict:
    file_path = mts.DATA_DIR / f"Book{book_num}.xlsx"
    angle, moment = mts.read_motion_data(file_path, mts.MOTION_CONFIG[MOTION_NAME]["moment_col"])
    _, _, angle_cycle, moment_cycle = mts.extract_third_cycle(angle, moment)

    theta_branch, moment_branch, theta_avg, moment_avg, branch_reason = extract_centered_positive_flexion_branch(
        angle_cycle,
        moment_cycle,
    )
    if branch_reason != "ok":
        return {
            "Intervention": intervention,
            "Book": book_num,
            "Motion": MOTION_NAME,
            "valid": False,
            "reason": branch_reason,
            "n_points_window": 0,
            "k_signed_nm_per_deg": np.nan,
            "k_abs_nm_per_deg": np.nan,
            "intercept_nm": np.nan,
            "r2": np.nan,
            "theta_sel": np.array([]),
            "moment_sel": np.array([]),
            "theta_branch": theta_branch,
            "moment_branch": moment_branch,
            "theta_avg_centered": theta_avg,
            "moment_avg_centered": moment_avg,
        }

    mask = (
        np.isfinite(theta_branch)
        & np.isfinite(moment_branch)
        & (moment_branch >= M_LOW_NM)
        & (moment_branch <= M_HIGH_NM)
    )
    sel_idx = np.flatnonzero(mask)
    if sel_idx.size < MIN_POINTS:
        return {
            "Intervention": intervention,
            "Book": book_num,
            "Motion": MOTION_NAME,
            "valid": False,
            "reason": f"too_few_points_in_{WINDOW_TAG}_window",
            "n_points_window": int(sel_idx.size),
            "k_signed_nm_per_deg": np.nan,
            "k_abs_nm_per_deg": np.nan,
            "intercept_nm": np.nan,
            "r2": np.nan,
            "theta_sel": theta_branch[sel_idx],
            "moment_sel": moment_branch[sel_idx],
            "theta_branch": theta_branch,
            "moment_branch": moment_branch,
            "theta_avg_centered": theta_avg,
            "moment_avg_centered": moment_avg,
        }

    # Use all branch points in [1, 5] Nm as requested.
    theta_sel = theta_branch[sel_idx]
    moment_sel = moment_branch[sel_idx]
    slope, intercept, r2 = fit_line(theta_sel, moment_sel)

    return {
        "Intervention": intervention,
        "Book": book_num,
        "Motion": MOTION_NAME,
        "valid": True,
        "reason": "ok",
        "n_points_window": int(theta_sel.size),
        "k_signed_nm_per_deg": slope,
        "k_abs_nm_per_deg": abs(slope),
        "intercept_nm": intercept,
        "r2": r2,
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
    }


def plot_fit_panels(results: list[dict], out_png: Path) -> None:
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

        ax.plot(theta_b, moment_b, color="0.80", lw=1.2, label="rising branch")

        if rec["moment_sel"].size:
            ax.scatter(rec["theta_sel"], rec["moment_sel"], s=12, color="tab:red", alpha=0.35, edgecolors="none", label=f"selected points in {int(M_LOW_NM)}-{int(M_HIGH_NM)} Nm")

        if rec["valid"]:
            theta_fit = np.linspace(float(np.min(rec["theta_sel"])), float(np.max(rec["theta_sel"])), 100)
            moment_fit = rec["k_signed_nm_per_deg"] * theta_fit + rec["intercept_nm"]
            ax.plot(theta_fit, moment_fit, color="tab:red", lw=1.3, ls=":", label="linear fit")
            title = f"{rec['Intervention']} | k={rec['k_abs_nm_per_deg']:.3f} Nm/deg | R2={rec['r2']:.3f}"
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

    fig.suptitle(f"MTS only: centered flexion branch linear fit using all points in {int(M_LOW_NM)}-{int(M_HIGH_NM)} Nm window", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved {out_png}")


def plot_bar(df: pd.DataFrame, out_png: Path) -> None:
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
    ax.set_title(f"MTS only (Flexion): linear-fit stiffness from {int(M_LOW_NM)}-{int(M_HIGH_NM)} Nm window")
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    books = mts.get_book_numbers(mts.DATA_DIR)
    interventions = mts.build_intervention_map(books)

    results = []
    for intervention, book_map in interventions.items():
        rec = analyze_intervention(intervention, int(book_map[MOTION_NAME]))
        results.append(rec)

    out_fit = OUT_DIR / f"mts_flexion_{WINDOW_TAG}_linear_fit_panels.png"
    plot_fit_panels(results, out_fit)

    rows = []
    for r in results:
        rows.append(
            {
                "Intervention": r["Intervention"],
                "Book": r["Book"],
                "Motion": r["Motion"],
                "valid": bool(r["valid"]),
                "reason": str(r["reason"]),
                "n_points_window": int(r["n_points_window"]),
                "k_signed_nm_per_deg": float(r["k_signed_nm_per_deg"]) if np.isfinite(r["k_signed_nm_per_deg"]) else np.nan,
                "k_abs_nm_per_deg": float(r["k_abs_nm_per_deg"]) if np.isfinite(r["k_abs_nm_per_deg"]) else np.nan,
                "intercept_nm": float(r["intercept_nm"]) if np.isfinite(r["intercept_nm"]) else np.nan,
                "r2": float(r["r2"]) if np.isfinite(r["r2"]) else np.nan,
                "theta_min_deg": float(r.get("theta_min_deg", np.nan)),
                "theta_max_deg": float(r.get("theta_max_deg", np.nan)),
                "moment_min_nm": float(r.get("moment_min_nm", np.nan)),
                "moment_max_nm": float(r.get("moment_max_nm", np.nan)),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"mts_flexion_{WINDOW_TAG}_linear_fit_stiffness.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    out_bar = OUT_DIR / f"mts_flexion_{WINDOW_TAG}_linear_fit_stiffness_bar.png"
    plot_bar(df, out_bar)

    valid = df[df["valid"]].copy()
    if not valid.empty:
        valid = valid.sort_values("k_abs_nm_per_deg", ascending=False).reset_index(drop=True)
        print(f"\nMTS flexion stiffness ranking ({int(M_LOW_NM)}-{int(M_HIGH_NM)} Nm linear fit):")
        print(valid[["Intervention", "k_abs_nm_per_deg", "r2", "n_points_window"]].to_string(index=False))


if __name__ == "__main__":
    main()
