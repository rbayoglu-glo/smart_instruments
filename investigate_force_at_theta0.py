from pathlib import Path

import numpy as np
import pandas as pd

import analyze_interventions as ai
import lamina_spreader as ls


OUT_DIR = Path(__file__).parent / "output"

CASES = [
    ("1 - Intact", "Intact"),
    ("2 - Intact Holding Longer", "intact, but holding"),
    ("3 - PUBF Left", "PUBF, left"),
    ("4 - FUF Left", "FUF, left"),
    ("5 - FBF", "FBF"),
    ("6 - Posterior Release", "Posterior Release"),
    ("7 - SPO", "SPO"),
]


def cycle_metrics(file_name: str, label: str) -> tuple[pd.DataFrame, dict]:
    kin = ls.compute_theta(ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID))
    force = ls.load_force(ai.find_force_csv(file_name))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, theta0, fmax = ls.sync(kin, force, lag)

    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()
    th_sm = d["theta_sm"].to_numpy()
    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    pcl = ls.per_cycle_lag(d, runs, fmax)

    rows = []
    for n, (s, e) in enumerate(runs, 1):
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, i1, pk = rb
        ramp = np.arange(i0, i1)

        # global theta-dist zero crossing within this ramp
        j = int(ramp[np.argmin(np.abs(th[ramp]))])

        rows.append(
            {
                "file_name": file_name,
                "intervention": label,
                "cycle": n,
                "F_start_N": float(f[i0]),
                "F_peak_N": float(f[pk]),
                "theta_global_start_deg": float(th[i0]),
                "theta_global_at_force_min_deg": float(th[np.argmin(f[ramp]) + i0]),
                "F_at_global_theta_near0_N": float(f[j]),
                "theta_at_F_global_theta_near0_deg": float(th[j]),
                "theta_sm_at_start_deg": float(th_sm[i0]),
                "theta0_global_deg": float(theta0),
                "lag_s": float(lag),
            }
        )

    meta = {
        "file_name": file_name,
        "intervention": label,
        "lag_s": float(lag),
        "theta0_global_deg": float(theta0),
        "fmax_N": float(fmax),
        "force_median_first200_raw_units_after_sign": float(force["force"].iloc[:200].median()),
        "force_median_low_force_region_N": float(d.loc[d["force"] < 0.15 * fmax, "force"].median()),
        "theta_sm_median_low_force_region_deg": float(d.loc[d["force"] < 0.15 * fmax, "theta_sm"].median()),
        "n_cycles": int(len(runs)),
        "residual_lag_med_s": float(np.median(pcl)) if len(pcl) else np.nan,
        "residual_lag_p95_abs_s": float(np.percentile(np.abs(pcl), 95)) if len(pcl) else np.nan,
        "residual_lag_n": int(len(pcl)),
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    all_rows = []
    meta_rows = []
    for file_name, label in CASES:
        print(f"\n=== {file_name} | {label} ===")
        try:
            cyc, meta = cycle_metrics(file_name, label)
        except Exception as exc:
            print(f"skip: {exc}")
            continue
        all_rows.append(cyc)
        meta_rows.append(meta)

    cyc_df = pd.concat(all_rows, ignore_index=True)
    meta_df = pd.DataFrame(meta_rows)

    cyc_path = OUT_DIR / "theta0_force_cycle_diagnostics.csv"
    meta_path = OUT_DIR / "theta0_force_case_meta.csv"
    cyc_df.to_csv(cyc_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    print("\nSaved diagnostics:")
    print(cyc_path)
    print(meta_path)

    print("\nFirst 3 cycles, intact-holding:")
    sub = cyc_df[cyc_df["file_name"].eq("2 - Intact Holding Longer") & cyc_df["cycle"].isin([1, 2, 3])]
    print(
        sub[
            [
                "cycle",
                "F_start_N",
                "theta_global_start_deg",
                "F_at_global_theta_near0_N",
                "theta_at_F_global_theta_near0_deg",
                "F_peak_N",
            ]
        ].to_string(index=False)
    )

    print("\nAcross interventions: median preload at ramp start")
    g = (
        cyc_df.groupby("intervention", as_index=False)["F_start_N"]
        .median()
        .sort_values("F_start_N", ascending=False)
    )
    print(g.to_string(index=False))

    print("\nResidual per-cycle lag summary (s)")
    print(
        meta_df[
            [
                "intervention",
                "lag_s",
                "residual_lag_med_s",
                "residual_lag_p95_abs_s",
                "residual_lag_n",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
