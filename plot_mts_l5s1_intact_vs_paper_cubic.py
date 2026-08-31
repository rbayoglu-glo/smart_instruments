from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_mts_testing as mts


OUT_DIR = Path(__file__).parent / "output"
OUT_PNG = OUT_DIR / "mts_l5s1_intact_centered_vs_paper_cubic.png"
OUT_CSV = OUT_DIR / "mts_l5s1_intact_centered_vs_paper_cubic.csv"

MOTION_NAME = "Flexion/Extension"
INTERVENTION_NAME = "Intact"

# Zhang et al., J Biomech 2020, Table 3 (L5-S1 FE fixed effects)
PAPER_C1 = 0.7957
PAPER_C2 = -0.1119
PAPER_C3 = 0.0219
PAPER_CF = 0.0035


def paper_moment(theta_deg: np.ndarray, follower_load_n: float = 0.0) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=float)
    return PAPER_C3 * theta**3 + PAPER_C2 * theta**2 + PAPER_C1 * theta + PAPER_CF * float(follower_load_n) * theta


def fit_cubic_no_intercept(theta_deg: np.ndarray, moment_nm: np.ndarray) -> tuple[float, float, float, float]:
    theta = np.asarray(theta_deg, dtype=float)
    moment = np.asarray(moment_nm, dtype=float)

    keep = np.isfinite(theta) & np.isfinite(moment)
    theta = theta[keep]
    moment = moment[keep]
    if theta.size < 5:
        raise RuntimeError("Not enough finite samples to fit cubic coefficients.")

    X = np.column_stack((theta, theta**2, theta**3))
    coeff, *_ = np.linalg.lstsq(X, moment, rcond=None)
    pred = X @ coeff
    rmse = float(np.sqrt(np.mean((moment - pred) ** 2)))
    return float(coeff[0]), float(coeff[1]), float(coeff[2]), rmse


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    books = mts.get_book_numbers(mts.DATA_DIR)
    interventions = mts.build_intervention_map(books)
    if INTERVENTION_NAME not in interventions:
        raise RuntimeError(f"Intervention not found: {INTERVENTION_NAME}")

    book_fe = int(interventions[INTERVENTION_NAME][MOTION_NAME])
    file_path = mts.DATA_DIR / f"Book{book_fe}.xlsx"

    angle_all, moment_all = mts.read_motion_data(file_path, mts.MOTION_CONFIG[MOTION_NAME]["moment_col"])
    _, _, angle_cycle, moment_cycle = mts.extract_third_cycle(angle_all, moment_all)

    moment_avg_centered, angle_avg_centered, _, _ = mts.compute_centered_average_curve(angle_cycle, moment_cycle)

    theta_meas = np.asarray(angle_avg_centered, dtype=float)
    moment_meas = np.asarray(moment_avg_centered, dtype=float)

    # Re-fit C1/C2/C3 from full FE centered data (both extension and flexion).
    fit_c1, fit_c2, fit_c3, fit_rmse_nm = fit_cubic_no_intercept(theta_meas, moment_meas)

    theta_plot = np.linspace(float(np.min(theta_meas)), float(np.max(theta_meas)), 600)
    moment_fit_plot = fit_c3 * theta_plot**3 + fit_c2 * theta_plot**2 + fit_c1 * theta_plot
    moment_paper_f0_plot = paper_moment(theta_plot, follower_load_n=0.0)
    moment_paper_f500_plot = paper_moment(theta_plot, follower_load_n=500.0)

    moment_paper_f0_at_meas = paper_moment(theta_meas, follower_load_n=0.0)
    moment_paper_f500_at_meas = paper_moment(theta_meas, follower_load_n=500.0)
    diff_f0 = moment_meas - moment_paper_f0_at_meas
    diff_f500 = moment_meas - moment_paper_f500_at_meas
    rmse_paper_f0_nm = float(np.sqrt(np.mean(diff_f0**2)))
    rmse_paper_f500_nm = float(np.sqrt(np.mean(diff_f500**2)))

    out_df = pd.DataFrame(
        {
            "theta_deg": theta_meas,
            "moment_mts_centered_nm": moment_meas,
            "moment_mts_refit_cubic_nm": fit_c3 * theta_meas**3 + fit_c2 * theta_meas**2 + fit_c1 * theta_meas,
            "moment_paper_f0_nm": moment_paper_f0_at_meas,
            "moment_paper_f500_nm": moment_paper_f500_at_meas,
            "moment_diff_mts_minus_paper_f0_nm": diff_f0,
            "moment_diff_mts_minus_paper_f500_nm": diff_f500,
        }
    )
    out_df.to_csv(OUT_CSV, index=False)
    print(f"saved {OUT_CSV}")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    ax.plot(theta_meas, moment_meas, color="tab:red", lw=2.0, label="Intact avg(load/unload), centered")
    ax.plot(theta_plot, moment_fit_plot, color="tab:blue", lw=1.8, ls="-.", label="MTS refit cubic (full FE)")
    ax.plot(theta_plot, moment_paper_f0_plot, color="black", lw=1.8, ls="--", label="Paper Eqn4 at F=0 (L5-S1)")
    ax.plot(theta_plot, moment_paper_f500_plot, color="tab:green", lw=1.8, ls=":", label="Paper Eqn4 at F=500 (L5-S1)")

    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel("Moment (Nm)")
    ax.set_title("L5-S1 FE (Intact): MTS full-FE refit vs paper model at F=0")

    coeff_text = (
        f"MTS refit (full FE): C1={fit_c1:.4f}, C2={fit_c2:.4f}, C3={fit_c3:.4f}\n"
        f"RMSE of MTS refit: {fit_rmse_nm:.3f} Nm\n"
        f"Paper coeffs: C1={PAPER_C1:.4f}, C2={PAPER_C2:.4f}, C3={PAPER_C3:.4f}, Cf={PAPER_CF:.4f}\n"
        f"RMSE (MTS vs paper F=0): {rmse_paper_f0_nm:.3f} Nm\n"
        f"RMSE (MTS vs paper F=500): {rmse_paper_f500_nm:.3f} Nm"
    )
    ax.text(
        0.98,
        0.02,
        coeff_text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )

    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
