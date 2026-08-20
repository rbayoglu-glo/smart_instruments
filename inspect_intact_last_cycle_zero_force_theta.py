from pathlib import Path

import numpy as np

import analyze_interventions as ai
import lamina_spreader as ls


FILE_NAME = "2 - Intact Holding Longer"
THETA_REF_WINDOW = 10


def main() -> None:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(FILE_NAME) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(FILE_NAME))
    lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
    d, _, fmax = ls.sync(kin, force, lag)

    t = d["t_sync"].to_numpy()
    f = d["force"].to_numpy()
    th = d["theta_dist"].to_numpy()

    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    if not runs:
        print("No cycles found")
        return

    n = len(runs)
    s, e = runs[-1]
    rb = ls.ramp_bounds(f, s, e, fmax)
    if rb is None:
        print("Last cycle had no valid ramp bounds")
        return

    i0, i1, pk = rb
    ramp = np.arange(i0, i1)

    f0 = float(f[i0])
    iref0 = max(0, i0 - THETA_REF_WINDOW + 1)
    th_ref = float(np.nanmedian(th[iref0:i0 + 1]))

    f_dyn = f[ramp] - f0
    th_dyn = th[ramp] - th_ref

    near_zero = np.where(np.abs(f_dyn) <= 0.5)[0]

    print(f"last loaded run idx: {n}, s={s}, e={e}, i0={i0}, pk={pk}, i1={i1}")
    print(f"time at i0={t[i0]:.3f} s, time at pk={t[pk]:.3f} s")
    print(f"raw theta_dist at i0={th[i0]:.4f} deg")
    print(f"theta_ref(median {THETA_REF_WINDOW} samples)={th_ref:.4f} deg")
    print(f"theta_dyn at i0={th_dyn[0]:.4f} deg")
    print(f"f0 at i0={f0:.4f} N")
    print(f"f_dyn min/max in ramp = {np.nanmin(f_dyn):.4f} / {np.nanmax(f_dyn):.4f} N")
    print(f"theta_dyn min/max in ramp = {np.nanmin(th_dyn):.4f} / {np.nanmax(th_dyn):.4f} deg")

    if near_zero.size == 0:
        print("No near-zero force points in ramp within +/-0.5 N")
    else:
        print("\nNear-zero force points in this ramp (|f_dyn| <= 0.5 N):")
        for j in near_zero[:20]:
            idx = ramp[j]
            print(
                f"  t={t[idx]:8.3f} s | f_dyn={f_dyn[j]:7.3f} N | "
                f"theta_dyn={th_dyn[j]:7.3f} deg | theta_dist={th[idx]:7.3f} deg"
            )


if __name__ == "__main__":
    main()
