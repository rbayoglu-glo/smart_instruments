import numpy as np

import analyze_interventions as ai
import lamina_spreader as ls


FILE_NAME = "2 - Intact Holding Longer"
FORCE_ZERO_BAND_N = 0.5
FORCE_ZERO_MAX_SAMPLES = 20
FORCE_ZERO_MIN_SAMPLES = 3
THETA_REF_WINDOW = 10


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
s, e = runs[-1]
i0, i1, pk = ls.ramp_bounds(f, s, e, fmax)
ramp = np.arange(i0, i1)

f0 = float(f[i0])
f_dyn = f[ramp] - f0
early_n = min(FORCE_ZERO_MAX_SAMPLES, len(ramp))
early = np.arange(early_n)
zero_idx = early[np.abs(f_dyn[early]) <= FORCE_ZERO_BAND_N]

if zero_idx.size >= FORCE_ZERO_MIN_SAMPLES:
    th_ref = float(np.nanmedian(th[ramp[zero_idx]]))
else:
    iref0 = max(0, i0 - THETA_REF_WINDOW + 1)
    th_ref = float(np.nanmedian(th[iref0:i0 + 1]))

th_dyn = th[ramp] - th_ref
near_zero = np.where(np.abs(f_dyn) <= FORCE_ZERO_BAND_N)[0]

print(f"last cycle i0={i0}, pk={pk}, t0={t[i0]:.3f}s")
print(f"force baseline f0={f0:.4f} N")
print(f"theta_ref={th_ref:.4f} deg, source points in zero band={zero_idx.size}")
print(f"theta_dyn at i0={th_dyn[0]:.4f} deg")
print("near-zero force samples in ramp (first 10):")
for j in near_zero[:10]:
    idx = ramp[j]
    print(f"  t={t[idx]:8.3f} s | f_dyn={f_dyn[j]:7.3f} N | theta_dyn={th_dyn[j]:7.3f} deg")
