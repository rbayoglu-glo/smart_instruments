from pathlib import Path

import numpy as np
import pandas as pd

import lamina_spreader as ls


BASE = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab")
FORCE_DIR = BASE / "Force_Data"
ROBOT_DIR = BASE / "Robot_Camera_Data"
TRACKER_XLSX = FORCE_DIR / "Intervention Tracker.xlsx"
OUT_DIR = Path(__file__).parent / "output"
QUALITY_R2_MIN = 0.70
QUALITY_MIN_POINTS = 12


def tracker_cases() -> pd.DataFrame:
    """Interventions to analyze: exclude only 1 - Intact baseline row."""
    df = pd.read_excel(TRACKER_XLSX, sheet_name="8-20")
    df = df[df["File Name"].notna()].copy()
    df = df[df["File Name"].str.match(r"^[0-9]+\s-\s")].copy()
    df = df[df["File Name"] != "1 - Intact"].copy()
    return df[["Session", "Intervention", "File Name", "Scan Start", "Scan Stop"]].sort_values("Session")


def find_robot_folder(file_name: str) -> Path:
    p = ROBOT_DIR / file_name
    if p.exists():
        return p
    # fallback for minor naming differences
    candidates = [x for x in ROBOT_DIR.iterdir() if x.is_dir() and file_name.lower() in x.name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Cannot resolve robot folder for '{file_name}'")


def find_force_csv(file_name: str) -> Path:
    p = FORCE_DIR / f"{file_name}.csv"
    if p.exists():
        return p

    csvs = [x for x in FORCE_DIR.iterdir() if x.is_file() and x.suffix.lower() == ".csv"]
    q = file_name.lower()
    candidates = [x for x in csvs if q in x.stem.lower() or x.stem.lower() in q]
    if len(candidates) == 1:
        return candidates[0]

    prefix = file_name.split("-")[0].strip()
    by_prefix = [x for x in csvs if x.stem.split("-")[0].strip() == prefix]
    if len(by_prefix) == 1:
        return by_prefix[0]

    raise FileNotFoundError(f"Cannot resolve force CSV for '{file_name}'")


def analyze_one(file_name: str, intervention: str) -> tuple[pd.DataFrame, dict]:
    force_csv = find_force_csv(file_name)
    robot_folder = find_robot_folder(file_name)
    poses_txt = robot_folder / "pf" / "poses.txt"

    kin = ls.compute_theta(ls.parse_poses(poses_txt, ls.SPREADER_UUID))
    force = ls.load_force(force_csv)
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))

    d, theta0, fmax = ls.sync(kin, force, lag)
    runs = ls.segment_cycles(d["force"].to_numpy(), ls.CYCLE_THRESH * fmax)
    cyc, _ = ls.fit_cycles(d, runs, fmax)

    if cyc.empty:
        return cyc, {
            "file_name": file_name,
            "intervention": intervention,
            "n_cycles_raw": 0,
            "n_cycles_hq": 0,
            "lag_s": lag,
            "theta0_deg": theta0,
            "max_force_N": float(fmax),
            "max_theta_deg": float(d["theta_dist"].max()),
            "k_linear_med_N_mm": np.nan,
            "k_linear_mean_N_mm": np.nan,
            "k_rot_med_Nm_deg": np.nan,
            "k_rot_mean_Nm_deg": np.nan,
            "R2_med": np.nan,
        }

    # k_rot [N.m/deg] -> k_linear [N/mm]
    factor = 1000.0 / (ls.D_O**2 * np.pi / 180.0)
    cyc = cyc.copy()
    cyc["k_linear_N_mm"] = cyc["k_rot_Nm_deg"] * factor
    cyc["quality_ok"] = (cyc["R2"] >= QUALITY_R2_MIN) & (cyc["n"] >= QUALITY_MIN_POINTS)
    cyc.insert(0, "intervention", intervention)
    cyc.insert(0, "file_name", file_name)
    cyc_hq = cyc[cyc["quality_ok"]].copy()

    if cyc_hq.empty:
        return cyc, {
            "file_name": file_name,
            "intervention": intervention,
            "n_cycles_raw": int(len(cyc)),
            "n_cycles_hq": 0,
            "lag_s": float(lag),
            "theta0_deg": float(theta0),
            "max_force_N": float(cyc["F_peak_N"].max()),
            "max_theta_deg": float(cyc["theta_max_deg"].max()),
            "k_linear_med_N_mm": np.nan,
            "k_linear_mean_N_mm": np.nan,
            "k_rot_med_Nm_deg": np.nan,
            "k_rot_mean_Nm_deg": np.nan,
            "R2_med": np.nan,
        }

    out = {
        "file_name": file_name,
        "intervention": intervention,
        "n_cycles_raw": int(len(cyc)),
        "n_cycles_hq": int(len(cyc_hq)),
        "lag_s": float(lag),
        "theta0_deg": float(theta0),
        "max_force_N": float(cyc["F_peak_N"].max()),
        "max_theta_deg": float(cyc["theta_max_deg"].max()),
        "k_linear_med_N_mm": float(cyc_hq["k_linear_N_mm"].median()),
        "k_linear_mean_N_mm": float(cyc_hq["k_linear_N_mm"].mean()),
        "k_rot_med_Nm_deg": float(cyc_hq["k_rot_Nm_deg"].median()),
        "k_rot_mean_Nm_deg": float(cyc_hq["k_rot_Nm_deg"].mean()),
        "R2_med": float(cyc_hq["R2"].median()),
    }
    return cyc, out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cases = tracker_cases()

    all_cycles = []
    summary = []

    print("Analyzing interventions (including 2 - Intact Holding Longer, excluding 1 - Intact baseline)...")
    print(f"High-quality cycle filter: R2 >= {QUALITY_R2_MIN:.2f} and n >= {QUALITY_MIN_POINTS}")
    for _, r in cases.iterrows():
        file_name = str(r["File Name"]).strip()
        intervention = str(r["Intervention"]).strip()
        print(f"\n=== {file_name} | {intervention} ===")
        cyc, summ = analyze_one(file_name, intervention)
        all_cycles.append(cyc)
        summary.append(summ)

    cyc_df = pd.concat(all_cycles, ignore_index=True)
    sum_df = pd.DataFrame(summary)

    # Rank by median high-quality stiffness (descending: stiffest first)
    sum_df["rank_k_linear"] = sum_df["k_linear_med_N_mm"].rank(ascending=False, method="dense")
    sum_df["rank_k_rot"] = sum_df["k_rot_med_Nm_deg"].rank(ascending=False, method="dense")

    # Comparison rankings for maxima
    sum_df["rank_max_force"] = sum_df["max_force_N"].rank(ascending=False, method="dense")
    sum_df["rank_max_theta"] = sum_df["max_theta_deg"].rank(ascending=False, method="dense")

    sum_df = sum_df.sort_values("rank_k_rot").reset_index(drop=True)

    cyc_path = OUT_DIR / "interventions_cycles.csv"
    cyc_hq_path = OUT_DIR / "interventions_cycles_hq.csv"
    sum_path = OUT_DIR / "interventions_summary.csv"
    cyc_df.to_csv(cyc_path, index=False)
    cyc_df[cyc_df["quality_ok"]].to_csv(cyc_hq_path, index=False)
    sum_df.to_csv(sum_path, index=False)

    view_cols = [
        "file_name", "intervention", "n_cycles_raw", "n_cycles_hq", "k_linear_med_N_mm",
        "k_rot_med_Nm_deg", "max_force_N", "max_theta_deg", "R2_med",
        "rank_k_linear", "rank_k_rot", "rank_max_force", "rank_max_theta",
    ]
    print("\n===== SUMMARY (ranked by high-quality k_rot median) =====")
    print(sum_df[view_cols].to_string(index=False))
    print(f"\nSaved: {sum_path}")
    print(f"Saved: {cyc_path}")
    print(f"Saved: {cyc_hq_path}")


if __name__ == "__main__":
    main()
