from pathlib import Path

import numpy as np
import pandas as pd

import analyze_intact_linear_region_stiffness as lr
import analyze_spreader_secant_stiffness as ass
import plot_intact_holding_all_cycles_force_tip_displacement as src
import plot_theta_moment_time_all_interventions as base


OUT_DIR = Path(__file__).parent / "output" / "linear_region_stiffness_cycle_grids"


def summarize_case(df: pd.DataFrame, file_name: str, intervention: str) -> dict:
    valid = df[df["valid"]]
    if valid.empty:
        return {
            "file_name": file_name,
            "intervention": intervention,
            "n_cycles": int(len(df)),
            "n_valid": 0,
            "k_mean_n_per_mm": np.nan,
            "k_median_n_per_mm": np.nan,
            "k_std_n_per_mm": np.nan,
            "r2_mean": np.nan,
        }

    return {
        "file_name": file_name,
        "intervention": intervention,
        "n_cycles": int(len(df)),
        "n_valid": int(len(valid)),
        "k_mean_n_per_mm": float(valid["k_linear_n_per_mm"].mean()),
        "k_median_n_per_mm": float(valid["k_linear_n_per_mm"].median()),
        "k_std_n_per_mm": float(valid["k_linear_n_per_mm"].std(ddof=0)),
        "r2_mean": float(valid["r2"].mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tracker = base.ai.tracker_cases().copy()
    tracker["File Name"] = tracker["File Name"].astype(str).str.strip()
    tracker["Intervention"] = tracker["Intervention"].astype(str).str.strip()

    summary_rows = []
    cycle_frames = []

    for file_name in ass.CASE_ORDER:
        sub = tracker[tracker["File Name"] == file_name]
        if sub.empty:
            print(f"skip (not found in tracker): {file_name}")
            continue

        intervention = str(sub.iloc[0]["Intervention"])
        print(f"\n=== {file_name} | {intervention} ===")

        cycles = src.build_cycle_rows_force_tip(file_name, intervention)
        if not cycles:
            print("no cycles produced")
            continue

        results = [lr.analyze_cycle(rec) for rec in cycles]

        stem = base.sanitize(file_name)
        out_png = OUT_DIR / f"{stem}_all_cycles_force_tip_linear_region_stiffness.png"
        out_csv = OUT_DIR / f"{stem}_all_cycles_force_tip_linear_region_stiffness_metrics.csv"

        lr.plot_cycle_grid(results, file_name, intervention, out_png)
        lr.save_metrics(results, out_csv)

        df_case = pd.read_csv(out_csv)
        df_case.insert(0, "intervention", intervention)
        df_case.insert(0, "file_name", file_name)
        cycle_frames.append(df_case)

        s = summarize_case(df_case, file_name, intervention)
        summary_rows.append(s)
        if s["n_valid"] > 0:
            print(
                f"valid cycles: {s['n_valid']}/{s['n_cycles']} | "
                f"k mean={s['k_mean_n_per_mm']:.3f} N/mm, "
                f"std={s['k_std_n_per_mm']:.3f} N/mm"
            )
        else:
            print(f"valid cycles: 0/{s['n_cycles']}")

    if cycle_frames:
        cyc_all = pd.concat(cycle_frames, ignore_index=True)
        out_all_cycles = OUT_DIR / "all_interventions_linear_region_stiffness_cycles.csv"
        cyc_all.to_csv(out_all_cycles, index=False)
        print(f"saved {out_all_cycles}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary["rank_k_median"] = summary["k_median_n_per_mm"].rank(ascending=False, method="dense")
        summary = summary.sort_values("rank_k_median").reset_index(drop=True)

        out_summary = OUT_DIR / "all_interventions_linear_region_stiffness_summary.csv"
        summary.to_csv(out_summary, index=False)
        print(f"saved {out_summary}")

        print("\nLinear-region stiffness ranking (higher = stiffer):")
        print(
            summary[
                [
                    "file_name",
                    "intervention",
                    "n_valid",
                    "n_cycles",
                    "k_mean_n_per_mm",
                    "k_median_n_per_mm",
                    "k_std_n_per_mm",
                    "r2_mean",
                    "rank_k_median",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
