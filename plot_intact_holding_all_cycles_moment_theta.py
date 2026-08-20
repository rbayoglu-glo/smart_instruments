from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_spreader_secant_stiffness as ass
import plot_theta_moment_time_all_interventions as base


OUT_DIR = Path(__file__).parent / "output" / "intact_holding_cycle_traces"
FILE_NAME = "2 - Intact Holding Longer"


def plot_moment_theta_grid(rows: list[dict], file_name: str, intervention: str, out_png: Path) -> None:
    n = len(rows)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.0 * nrows), squeeze=False)

    for i, rec in enumerate(rows):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        th = rec["theta_rel"]
        mm = rec["moment_rel"]

        ax.plot(th, mm, color="tab:blue", lw=1.5)
        ax.axvline(0.0, color="0.7", lw=0.8)
        ax.axhline(0.0, color="0.7", lw=0.8)

        # Keep the same guide moments used elsewhere for quick comparison.
        for g in base.M_GUIDES_NM:
            ax.axhline(g, color="tab:red", lw=1.0, ls="--", alpha=0.85)

        thl = rec["theta_at_m_low_deg"]
        thh = rec["theta_at_m_high_deg"]
        if np.isfinite(thl):
            ax.scatter([thl], [base.M_LOW], s=28, marker="o", color="tab:red", zorder=5)
            ax.text(thl, base.M_LOW, f" \u03b8@{base.M_LOW:.0f}={thl:.2f}", color="tab:red", fontsize=7,
                    ha="left", va="bottom")
        if np.isfinite(thh):
            ax.scatter([thh], [base.M_HIGH], s=28, marker="o", color="tab:red", zorder=5)
            ax.text(thh, base.M_HIGH, f" \u03b8@{base.M_HIGH:.0f}={thh:.2f}", color="tab:red", fontsize=7,
                    ha="left", va="bottom")

        ax.set_title(
            f"cyc {int(rec['cycle'])} | dtheta={rec['new_dtheta_deg']:.3f} | {rec['new_reason']}",
            fontsize=8,
        )
        ax.set_xlabel("theta_rel [deg]")
        ax.set_ylabel("moment_rel [Nm]")
        ax.grid(alpha=0.25)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"{file_name} ({intervention}): moment_rel vs theta_rel for all loading cycles\n"
        f"No exclusion filtering applied; dashed red guides at {base.M_LOW:.0f} Nm and {base.M_HIGH:.0f} Nm",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved {out_png}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tracker = base.ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    sub = tracker[tracker["File Name"] == FILE_NAME]
    if sub.empty:
        raise RuntimeError(f"Case not found in tracker: {FILE_NAME}")

    intervention = str(sub.iloc[0]["Intervention"])
    print(f"=== {FILE_NAME} | {intervention} ===")

    # Reuse the cycle reconstruction path; do not apply exclusion filtering.
    rows = base.build_cycle_rows(FILE_NAME, intervention)
    if not rows:
        raise RuntimeError("No cycles produced for intact holding case.")

    out_png = OUT_DIR / "intact_holding_all_cycles_moment_theta.png"
    out_csv = OUT_DIR / "intact_holding_all_cycles_moment_theta_metrics.csv"

    plot_moment_theta_grid(rows, FILE_NAME, intervention, out_png)
    base.save_metrics(rows, out_csv)


if __name__ == "__main__":
    main()
