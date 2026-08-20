import numpy as np

import analyze_interventions as ai
import lamina_spreader as ls
import plot_sync_check_all_interventions_stacked as p

FILE_NAME = "7 - SPO"

kin = ls.compute_theta(
    ls.parse_poses(ai.find_robot_folder(FILE_NAME) / "pf" / "poses.txt", ls.SPREADER_UUID)
)
force = ls.load_force(ai.find_force_csv(FILE_NAME))
lag, _ = ls.refine_lag(kin, force, ls.find_lag(force, kin))
d, _, fmax = ls.sync(kin, force, lag)

t = d["t_sync"].to_numpy(float)
f = d["force"].to_numpy(float)

runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
print(f"n_runs={len(runs)}")
for k, (s, e) in enumerate(runs, 1):
    rb = ls.ramp_bounds(f, s, e, fmax)
    if rb is None:
        continue
    i0, _, pk = rb
    ir = p.refined_ramp_start(f, int(pk), int(max(0, s - p.START_LOOKBACK)))
    print(
        f"{k:2d}: s={s:5d} t={t[s]:8.3f}  e={e:5d} t={t[e-1]:8.3f}  "
        f"pk={pk:5d} t={t[pk]:8.3f}  i0={i0:5d} t={t[i0]:8.3f}  "
        f"ir={ir:5d} t={t[ir]:8.3f}  f(ir)={f[ir]:6.2f}"
    )

peaks = p.detect_cycle_peaks(f, float(fmax))
print(f"\npeak_based_cycles={len(peaks)}")
lo = 0
for i, pk in enumerate(peaks, 1):
    st = p.refined_ramp_start(f, int(pk), int(lo))
    print(
        f"{i:2d}: start={st:5d} t={t[st]:8.3f} f={f[st]:6.2f}  "
        f"peak={pk:5d} t={t[pk]:8.3f} f={f[pk]:6.2f}"
    )
    lo = int(pk) + 1
