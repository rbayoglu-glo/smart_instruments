from pathlib import Path

import numpy as np
import pandas as pd

import analyze_mts_testing as mts


OUT_DIR = Path(__file__).parent / "output"
SECANT_M_LOW_NM = 1.0
SECANT_M_HIGH_NM = 4.0
MIN_RAMP_SAMPLES = 6


def secant_on_rising_branch(theta_rel_deg: np.ndarray, moment_nm: np.ndarray) -> tuple[float, float, float]:
    """Secant stiffness on rising hull between fixed moment bounds."""
    if theta_rel_deg.size < 2 or moment_nm.size < 2:
        return np.nan, np.nan, np.nan

    hull = moment_nm >= np.maximum.accumulate(moment_nm) - 1e-9
    m = moment_nm[hull]
    a = theta_rel_deg[hull]
    if m.size < 2:
        return np.nan, np.nan, np.nan

    order = np.argsort(m)
    m = m[order]
    a = a[order]

    m_u, first = np.unique(m, return_index=True)
    a_u = a[first]
    if m_u.size < 2:
        return np.nan, np.nan, np.nan

    if SECANT_M_LOW_NM < float(np.min(m_u)) or SECANT_M_HIGH_NM > float(np.max(m_u)):
        return np.nan, np.nan, np.nan

    th_low = float(np.interp(SECANT_M_LOW_NM, m_u, a_u))
    th_high = float(np.interp(SECANT_M_HIGH_NM, m_u, a_u))
    dtheta = th_high - th_low
    if not np.isfinite(dtheta) or abs(dtheta) <= 1e-10:
        return np.nan, th_low, th_high

    k_sec = float((SECANT_M_HIGH_NM - SECANT_M_LOW_NM) / dtheta)
    return k_sec, th_low, th_high


def extract_loading_ramp(angle_cycle: np.ndarray, moment_cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extract loading ramp in third cycle: min->max moment segment."""
    if angle_cycle.size < MIN_RAMP_SAMPLES:
        return np.array([]), np.array([]), -1, -1

    pk = int(np.argmax(moment_cycle))
    if pk < MIN_RAMP_SAMPLES - 1:
        return np.array([]), np.array([]), -1, -1

    i0 = int(np.argmin(moment_cycle[: pk + 1]))
    if (pk - i0 + 1) < MIN_RAMP_SAMPLES:
        return np.array([]), np.array([]), -1, -1

    theta_rel = angle_cycle[i0 : pk + 1] - float(angle_cycle[i0])
    moment_rel = moment_cycle[i0 : pk + 1] - float(moment_cycle[i0])
    return theta_rel, moment_rel, i0, pk


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    books = mts.get_book_numbers(mts.DATA_DIR)
    interventions = mts.build_intervention_map(books)

    rows = []

    print("MTS secant stiffness on loading ramp only")
    print(f"Window: [{SECANT_M_LOW_NM:.1f}, {SECANT_M_HIGH_NM:.1f}] Nm")
    print("Moment basis: preload-corrected per cycle (M - M_start_ramp)")
    print("Angle basis: ramp-start referenced (theta - theta_start_ramp)")

    for intervention, book_map in interventions.items():
        for motion_name, cfg in mts.MOTION_CONFIG.items():
            file_path = mts.DATA_DIR / f"Book{book_map[motion_name]}.xlsx"
            angle, moment = mts.read_motion_data(file_path, cfg["moment_col"])
            _, _, angle_cycle, moment_cycle = mts.extract_third_cycle(angle, moment)

            theta_rel, moment_rel, i0, pk = extract_loading_ramp(angle_cycle, moment_cycle)
            k_sec, th_low, th_high = secant_on_rising_branch(theta_rel, moment_rel)

            rows.append(
                {
                    "Intervention": intervention,
                    "Motion": motion_name,
                    "Book": int(book_map[motion_name]),
                    "RampStartIdx": int(i0),
                    "RampPeakIdx": int(pk),
                    "M_peak_dyn_Nm": float(np.nanmax(moment_rel)) if moment_rel.size else np.nan,
                    "theta_peak_rel_deg": float(np.nanmax(theta_rel)) if theta_rel.size else np.nan,
                    "k_sec_signed_Nm_per_deg": float(k_sec) if np.isfinite(k_sec) else np.nan,
                    "k_sec_abs_Nm_per_deg": abs(float(k_sec)) if np.isfinite(k_sec) else np.nan,
                    "theta_low_deg": float(th_low) if np.isfinite(th_low) else np.nan,
                    "theta_high_deg": float(th_high) if np.isfinite(th_high) else np.nan,
                    "valid_secant": bool(np.isfinite(k_sec)),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "mts_secant_loading_ramp_1to4Nm.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    print("\n" + "=" * 72)
    print("Per-motion ranking (higher k_sec_abs = stiffer)")
    print("=" * 72)
    for motion in mts.MOTION_CONFIG.keys():
        sub = (
            df[df["Motion"] == motion]
            .sort_values("k_sec_abs_Nm_per_deg", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        print(f"\n{motion}:")
        for i, r in sub.iterrows():
            kval = r["k_sec_abs_Nm_per_deg"]
            kval_txt = f"{kval:.4f}" if np.isfinite(kval) else "N/A"
            print(f"  {i + 1:>2}. {r['Intervention']:<18} k_sec = {kval_txt} Nm/deg")

    overall = (
        df.groupby("Intervention", as_index=False)
        .agg(
            k_sec_mean_abs_Nm_per_deg=("k_sec_abs_Nm_per_deg", "mean"),
            n_valid=("valid_secant", "sum"),
            n_total=("valid_secant", "count"),
        )
        .sort_values("k_sec_mean_abs_Nm_per_deg", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    out_sum = OUT_DIR / "mts_secant_loading_ramp_1to4Nm_summary.csv"
    overall.to_csv(out_sum, index=False)
    print(f"saved {out_sum}")

    print("\n" + "=" * 72)
    print("Overall ranking (mean across FE, LB, AR)")
    print("=" * 72)
    for i, r in overall.iterrows():
        kval = r["k_sec_mean_abs_Nm_per_deg"]
        kval_txt = f"{kval:.4f}" if np.isfinite(kval) else "N/A"
        print(
            f"  {i + 1:>2}. {r['Intervention']:<18} "
            f"k_mean = {kval_txt} Nm/deg  ({int(r['n_valid'])}/{int(r['n_total'])} valid)"
        )


if __name__ == "__main__":
    main()
