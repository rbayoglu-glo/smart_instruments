from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).parent / "output"
CYCLES_CSV = OUT_DIR / "spreader_secant_cycles_preloadcorr_rampref_fzero.csv"

STYLE = {
    "intact, but holding": {"color": "#000000", "marker": "o"},
    "PUBF, left": {"color": "#0072B2", "marker": "s"},
    "FUF, left": {"color": "#E69F00", "marker": "^"},
    "FBF": {"color": "#009E73", "marker": "D"},
    "Posterior Release": {"color": "#D55E00", "marker": "v"},
    "SPO": {"color": "#CC79A7", "marker": "P"},
}

ORDER = [
    "intact, but holding",
    "PUBF, left",
    "FUF, left",
    "FBF",
    "Posterior Release",
    "SPO",
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    cyc = pd.read_csv(CYCLES_CSV)
    req = {"intervention", "cycle", "F_preload_start_N"}
    missing = req.difference(set(cyc.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    cyc = cyc.sort_values(["intervention", "cycle"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.0, 6.0))

    interventions = [x for x in ORDER if x in set(cyc["intervention"])]
    for label in interventions:
        g = cyc[cyc["intervention"] == label].copy()
        if g.empty:
            continue
        st = STYLE.get(label, {"color": "#444444", "marker": "o"})

        x = g["cycle"].to_numpy(int)
        y = g["F_preload_start_N"].to_numpy(float)
        ax.plot(
            x,
            y,
            color=st["color"],
            marker=st["marker"],
            lw=1.8,
            ms=5.5,
            alpha=0.95,
            label=label,
        )

    ax.set_xlabel("cycle number")
    ax.set_ylabel("ramp start force (preload) [N]")
    ax.set_title("Ramp start preload force across cycles and interventions")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Show cycle coverage extent for quick comparison.
    max_cycle = int(np.nanmax(cyc["cycle"].to_numpy(float)))
    ax.set_xlim(0.5, max_cycle + 0.5)

    fig.tight_layout()
    out_png = OUT_DIR / "spreader_ramp_start_force_across_cycles_interventions.png"
    fig.savefig(out_png, dpi=160)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
