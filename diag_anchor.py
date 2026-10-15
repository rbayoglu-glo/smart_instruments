import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

import lamina_spreader_lr as L
import plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart as R
from plot_sync_check_all_levels_sides_stacked_true_raw_shpose_only import compute_theta_from_sh_pose
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered import filter_force, filter_theta, load_force_true_raw
from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline import compute_force_baseline, compute_theta0_clean

for name in L.TRIALS:
    kin = compute_theta_from_sh_pose(R.TRACK_DIR / L.TRIALS[name], valid_only=False)
    force = load_force_true_raw(R.FORCE_DIR / f"{name}.csv")
    t0 = min(kin["t_true_est"].min(), force["t_true_est"].min())
    kin_t = (kin["t_true_est"] - t0).dt.total_seconds().to_numpy()
    force_t = (force["t_true_est"] - t0).dt.total_seconds().to_numpy()
    kin_fs = 1.0 / np.nanmedian(np.diff(kin_t))
    force_fs = 1.0 / np.nanmedian(np.diff(force_t))
    theta_raw = np.where(kin["valid"].to_numpy(), kin["theta_deg"].to_numpy(), np.nan)
    theta_filt = filter_theta(theta_raw, kin_fs)
    force_filt = filter_force(force["force_dir"].to_numpy(), force_fs)
    theta_ok = np.isfinite(theta_filt)
    kin_for_lag = pd.DataFrame({"t_rel": kin_t[theta_ok], "theta_deg": theta_filt[theta_ok]})
    force_for_lag = pd.DataFrame({"t_rel": force_t, "force": force_filt})
    lag = L.find_lag(force_for_lag, kin_for_lag)
    t = force_t - lag

    seed_bl = compute_force_baseline(force_filt)
    seed_dyn = force_filt - seed_bl
    f_on_kin = np.interp(kin_t, t, seed_dyn, left=np.nan, right=np.nan)
    fmax = float(np.nanmax(f_on_kin))
    seed_th0 = compute_theta0_clean(kin_t, f_on_kin, theta_filt, fmax)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cands, *_ = R.detect_with_baseline(t, force_filt, seed_bl, kin_t, theta_filt, seed_th0, require_force_hold=False)
    starts = [round(float(t[c["i0"]]), 2) for c in cands[:4]]
    print(f"{name}: lag={lag:.3f} seed_bl={seed_bl:.3f} seed_th0={seed_th0:.3f} t_first={t[0]:.2f} seed_starts={starts}")
    if cands:
        a = int(cands[0]["i0"])
        end = float(t[a] - R.FORCE_BASELINE_GUARD_S)
        st = end - R.FORCE_BASELINE_WINDOW_S
        m = (t >= st) & (t < end)
        print(f"   anchor_t={t[a]:.2f} window=[{st:.2f},{end:.2f}) n={m.sum()} force_med={np.nanmedian(force_filt[m]):.3f} force_min={np.nanmin(force_filt[m]):.3f} force_max={np.nanmax(force_filt[m]):.3f}")
        m2 = (kin_t >= st) & (kin_t < end)
        print(f"   theta_finite_in_window={int(np.isfinite(theta_filt[m2]).sum())} of {int(m2.sum())}")
