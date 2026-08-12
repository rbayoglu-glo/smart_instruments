from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).parent / "output"
IN_CSV = OUT / "spreader_theta_at_moment_cycles.csv"
TARGETS = (2.0, 3.0, 4.0, 5.0)
ORDER = ["Intact Holding", "PUBF Left", "FUF Left", "FBF", "Posterior Release", "SPO"]


def rank_corr(summary: pd.DataFrame, col: str) -> float:
    r = summary[["intervention", col]].dropna().sort_values(col).reset_index(drop=True)
    ordered = [x for x in r["intervention"] if x in ORDER]
    if len(ordered) <= 2:
        return np.nan
    x = [ORDER.index(k) for k in ordered]
    y = list(range(len(ordered)))
    return float(np.corrcoef(x, y)[0, 1])


def summarize(cyc: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = {"cycle": "count"}
    for t in TARGETS:
        agg[f"th_eng@{t}"] = "median"
    s = cyc.groupby(["file_name", "intervention"], as_index=False).agg(agg)
    s = s.rename(columns={"cycle": "n_cycles"})

    rows = []
    for t in TARGETS:
        col = f"th_eng@{t}"
        reach = cyc.assign(ok=cyc[col].notna()).groupby("intervention", as_index=False)["ok"].sum()
        ncyc = cyc.groupby("intervention", as_index=False)["cycle"].count().rename(columns={"cycle": "n_cycles"})
        m = s[["intervention", col]].merge(reach, on="intervention").merge(ncyc, on="intervention")
        rho = rank_corr(s, col)
        rows.append({"scenario": label, "moment_Nm": t, "rank_corr": rho})
        print(f"\n[{label}] M={t:.1f} Nm, rank corr {rho:+.2f}")
        r = m.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
        for i, row in r.iterrows():
            print(f"  {i+1}. {row['intervention']:<18} theta={row[col]:.3f} [{int(row['ok'])}/{int(row['n_cycles'])}]")

    return s, pd.DataFrame(rows)


def main() -> None:
    cyc = pd.read_csv(IN_CSV)

    s_all, rc_all = summarize(cyc, "all")

    mx = cyc.groupby("file_name")["cycle"].transform("max")
    trimmed = cyc[(cyc["cycle"] > 3) & (cyc["cycle"] < mx)].copy()

    print("\nCycles kept after dropping first 3 and last:")
    print(trimmed.groupby("intervention")["cycle"].count().to_string())

    s_trim, rc_trim = summarize(trimmed, "trim_3first_1last")

    # compare theta shifts at each moment/intervention
    comp = s_all.merge(s_trim, on=["file_name", "intervention"], how="outer", suffixes=("_all", "_trim"))
    for t in TARGETS:
        comp[f"delta_th_eng@{t}"] = comp[f"th_eng@{t}_trim"] - comp[f"th_eng@{t}_all"]

    rank_comp = rc_all.merge(rc_trim, on="moment_Nm", suffixes=("_all", "_trim"))
    rank_comp["delta_rank_corr"] = rank_comp["rank_corr_trim"] - rank_comp["rank_corr_all"]

    comp_path = OUT / "cycle_trimming_theta_comparison.csv"
    rank_path = OUT / "cycle_trimming_rankcorr_comparison.csv"
    comp.to_csv(comp_path, index=False)
    rank_comp.to_csv(rank_path, index=False)

    print("\nRank-correlation comparison:")
    print(rank_comp.to_string(index=False))
    print("\nSaved:")
    print(comp_path)
    print(rank_path)


if __name__ == "__main__":
    main()
