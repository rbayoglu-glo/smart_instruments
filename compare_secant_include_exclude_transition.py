from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).parent / "output"
SECANT_CSV = OUT_DIR / "spreader_secant_cycles_preloadcorr_rampref_fzero.csv"
FLAG_CSV = OUT_DIR / "sync_check_transition_flags.csv"

TIME_TOL_S = 0.05


def summarize(cyc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (file_name, intervention), g in cyc.groupby(["file_name", "intervention"], sort=False):
        k = g.loc[g["valid_secant"], "k_sec_Nm_per_deg"].to_numpy(float)
        if k.size == 0:
            med = np.nan
            mean = np.nan
            q25 = np.nan
            q75 = np.nan
            n_valid = 0
        else:
            med = float(np.nanmedian(k))
            mean = float(np.nanmean(k))
            q25 = float(np.nanpercentile(k, 25))
            q75 = float(np.nanpercentile(k, 75))
            n_valid = int(np.count_nonzero(np.isfinite(k)))

        rows.append(
            {
                "file_name": file_name,
                "intervention": intervention,
                "n_cycles": int(len(g)),
                "n_valid": n_valid,
                "k_sec_med_Nm_per_deg": med,
                "k_sec_mean_Nm_per_deg": mean,
                "k_sec_q25_Nm_per_deg": q25,
                "k_sec_q75_Nm_per_deg": q75,
            }
        )

    out = pd.DataFrame(rows)
    out["rank_k_sec"] = out["k_sec_med_Nm_per_deg"].rank(ascending=False, method="dense")
    out = out.sort_values("rank_k_sec").reset_index(drop=True)
    return out


def main() -> None:
    cyc = pd.read_csv(SECANT_CSV)
    flg = pd.read_csv(FLAG_CSV)

    must_cols = {"file_name", "t_start_s", "transition_flag"}
    missing = must_cols.difference(set(flg.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in {FLAG_CSV.name}: {sorted(missing)}")

    flagged = flg[flg["transition_flag"].astype(bool)].copy()

    def mark_row(row: pd.Series) -> bool:
        sub = flagged[flagged["file_name"] == row["file_name"]]
        if sub.empty:
            return False
        dt = np.abs(sub["t_start_s"].to_numpy(float) - float(row["t_start_s"]))
        return bool(np.any(dt <= TIME_TOL_S))

    cyc["exclude_transition"] = cyc.apply(mark_row, axis=1)

    include = summarize(cyc)
    exclude = summarize(cyc[~cyc["exclude_transition"]].copy())

    comp = include.merge(
        exclude,
        on=["file_name", "intervention"],
        how="outer",
        suffixes=("_include", "_exclude"),
    )
    comp["delta_k_sec_med"] = comp["k_sec_med_Nm_per_deg_exclude"] - comp["k_sec_med_Nm_per_deg_include"]
    comp["delta_rank"] = comp["rank_k_sec_exclude"] - comp["rank_k_sec_include"]
    comp = comp.sort_values("rank_k_sec_exclude", na_position="last").reset_index(drop=True)

    excluded_rows = cyc[cyc["exclude_transition"]].copy()

    print("=" * 84)
    print("Secant stiffness comparison: INCLUDE ALL vs EXCLUDE TRANSITION-FLAGGED")
    print("=" * 84)
    print(f"Flag matching tolerance: +/- {TIME_TOL_S:.3f} s on cycle start time")
    print(f"Excluded cycles: {len(excluded_rows)}")
    if not excluded_rows.empty:
        print(
            excluded_rows[
                [
                    "file_name",
                    "intervention",
                    "cycle",
                    "t_start_s",
                    "F_peak_raw_N",
                    "k_sec_Nm_per_deg",
                ]
            ].to_string(index=False)
        )

    view_cols = [
        "intervention",
        "n_valid_include",
        "n_valid_exclude",
        "k_sec_med_Nm_per_deg_include",
        "k_sec_med_Nm_per_deg_exclude",
        "delta_k_sec_med",
        "rank_k_sec_include",
        "rank_k_sec_exclude",
        "delta_rank",
    ]
    print("\n" + comp[view_cols].to_string(index=False))

    include_out = OUT_DIR / "spreader_secant_summary_include_all.csv"
    exclude_out = OUT_DIR / "spreader_secant_summary_exclude_transition.csv"
    comp_out = OUT_DIR / "spreader_secant_summary_include_vs_exclude_transition.csv"
    ex_rows_out = OUT_DIR / "spreader_secant_cycles_excluded_transition.csv"

    include.to_csv(include_out, index=False)
    exclude.to_csv(exclude_out, index=False)
    comp.to_csv(comp_out, index=False)
    excluded_rows.to_csv(ex_rows_out, index=False)

    print(f"\nSaved: {include_out}")
    print(f"Saved: {exclude_out}")
    print(f"Saved: {comp_out}")
    print(f"Saved: {ex_rows_out}")


if __name__ == "__main__":
    main()
