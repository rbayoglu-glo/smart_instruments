from pathlib import Path

import pandas as pd

import analyze_spreader_secant_stiffness as ass
import plot_intact_holding_all_cycles_force_tip_displacement as ref
import plot_theta_moment_time_all_interventions as base


OUT_DIR = Path(__file__).parent / "output" / "force_tip_displacement_cycle_grids"
SKIP_FILE = "2 - Intact Holding Longer"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tracker = base.ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    all_rows = []

    for file_name in ass.CASE_ORDER:
        if file_name == SKIP_FILE:
            continue

        sub = tracker[tracker["File Name"] == file_name]
        if sub.empty:
            print(f"skip (not found in tracker): {file_name}")
            continue

        intervention = str(sub.iloc[0]["Intervention"])
        print(f"\n=== {file_name} | {intervention} ===")

        rows = ref.build_cycle_rows_force_tip(file_name, intervention)
        if not rows:
            print("no cycles produced")
            continue

        stem = base.sanitize(file_name)
        out_png = OUT_DIR / f"{stem}_all_cycles_force_tip_displacement.png"
        out_csv = OUT_DIR / f"{stem}_all_cycles_force_tip_displacement_metrics.csv"

        ref.plot_force_tip_grid(rows, file_name, intervention, out_png)
        ref.save_metrics(rows, out_csv)

        for rec in rows:
            all_rows.append(
                {
                    "file_name": rec["file_name"],
                    "intervention": rec["intervention"],
                    "cycle": rec["cycle"],
                    "F_peak_dyn_N": rec["F_peak_dyn_N"],
                    "tip_span_mm": rec["tip_span_mm"],
                    "tip_at_peak_mm": rec["tip_at_peak_mm"],
                    "tip_at_1Nm_mm": rec["tip_at_markers_mm"][0],
                    "tip_at_4Nm_mm": rec["tip_at_markers_mm"][1],
                    "k_1to4_n_per_mm": rec["k_1to4_n_per_mm"],
                    "marker_reason": rec["marker_reason"],
                }
            )

    if all_rows:
        out_all = OUT_DIR / "all_other_interventions_force_tip_displacement_metrics.csv"
        pd.DataFrame(all_rows).to_csv(out_all, index=False)
        print(f"saved {out_all}")


if __name__ == "__main__":
    main()
