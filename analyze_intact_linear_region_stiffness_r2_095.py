from pathlib import Path

import pandas as pd

import analyze_intact_linear_region_stiffness as lr
import plot_intact_holding_all_cycles_force_tip_displacement as src
import plot_theta_moment_time_all_interventions as base


OUT_DIR = Path(__file__).parent / "output" / "intact_holding_cycle_traces"
FILE_NAME = "2 - Intact Holding Longer"
R2_STRICT_RELAXED = 0.95


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep larger-window settings and relax only strict R2 floor.
    lr.MIN_SPAN_MM = 2.0
    lr.MIN_R2_STRICT = R2_STRICT_RELAXED

    tracker = base.ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    sub = tracker[tracker["File Name"] == FILE_NAME]
    if sub.empty:
        raise RuntimeError(f"Case not found in tracker: {FILE_NAME}")

    intervention = str(sub.iloc[0]["Intervention"])
    print(f"=== {FILE_NAME} | {intervention} ===")
    print(f"Using relaxed strict R2 floor: {lr.MIN_R2_STRICT:.2f}")

    cycles = src.build_cycle_rows_force_tip(FILE_NAME, intervention)
    if not cycles:
        raise RuntimeError("No cycles produced for intact holding case.")

    results = [lr.analyze_cycle(rec) for rec in cycles]

    out_png = OUT_DIR / "intact_holding_all_cycles_force_tip_linear_region_stiffness_r2_095.png"
    out_csv = OUT_DIR / "intact_holding_all_cycles_force_tip_linear_region_stiffness_metrics_r2_095.csv"

    lr.plot_cycle_grid(results, FILE_NAME, intervention, out_png)
    lr.save_metrics(results, out_csv)

    df = pd.read_csv(out_csv)
    valid = df[df["valid"]]
    print(f"valid cycles: {len(valid)}/{len(df)}")
    if not valid.empty:
        print(
            f"k mean={valid['k_linear_n_per_mm'].mean():.3f} N/mm, "
            f"std={valid['k_linear_n_per_mm'].std(ddof=0):.3f} N/mm, "
            f"median={valid['k_linear_n_per_mm'].median():.3f} N/mm"
        )
        print("fit mode counts:")
        print(valid["fit_selection_mode"].value_counts().to_string())
        print(f"median fit span={valid['fit_span_mm'].median():.3f} mm")


if __name__ == "__main__":
    main()
