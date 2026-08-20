import numpy as np

import analyze_interventions as ai
import lamina_spreader as ls
import plot_sync_check_all_interventions_stacked as p

CASES = ["6 - Posterior Release", "7 - SPO", "2 - Intact Holding Longer", "3 - PUBF Left"]

for file_name in CASES:
    kin = ls.compute_theta(
        ls.parse_poses(ai.find_robot_folder(file_name) / "pf" / "poses.txt", ls.SPREADER_UUID)
    )
    force = ls.load_force(ai.find_force_csv(file_name))
    lag0 = ls.find_lag(force, kin)
    lag, _ = ls.refine_lag(kin, force, lag0)
    d, _, fmax = ls.sync(kin, force, lag)

    t = d["t_sync"].to_numpy(float)
    f = d["force"].to_numpy(float)
    th = d["theta_dist"].to_numpy(float)

    runs = ls.segment_cycles(f, ls.CYCLE_THRESH * fmax)
    cands = []
    for s, e in runs:
        rb = ls.ramp_bounds(f, s, e, fmax)
        if rb is None:
            continue
        i0, _, pk = rb
        cands.append(
            {
                "s": int(s),
                "e": int(e),
                "i0": int(i0),
                "pk": int(pk),
                "fpk": float(f[pk]),
                "dur": int(e - s),
                "theta_dp": float(th[pk] - th[i0]),
            }
        )

    kept = p.dedupe_peak_candidates(cands, t)

    print("\n" + "=" * 80)
    print(file_name)
    print(f"lag={lag:+.3f}s, raw={len(cands)}, kept={len(kept)}")

    print("-- kept cycles --")
    for i, c in enumerate(sorted(kept, key=lambda x: x["pk"]), 1):
        print(
            f"{i:2d}: t0={t[c['i0']]:8.3f}s  tpk={t[c['pk']]:8.3f}s  "
            f"Fpk={c['fpk']:7.2f}N  dt={(t[c['pk']] - t[c['i0']]):6.3f}s  "
            f"dtheta={c['theta_dp']:7.3f}deg  run_dur={c['dur']:4d}"
        )

    print("-- close raw gaps --")
    raw = sorted(cands, key=lambda x: x["pk"])
    for a, b in zip(raw[:-1], raw[1:]):
        gap = float(t[b["pk"]] - t[a["pk"]])
        if gap < 16.0:
            print(
                f"gap={gap:6.3f}s | A tpk={t[a['pk']]:8.3f} Fpk={a['fpk']:7.2f} dth={a['theta_dp']:7.3f} dur={a['dur']:4d}"
                f" || B tpk={t[b['pk']]:8.3f} Fpk={b['fpk']:7.2f} dth={b['theta_dp']:7.3f} dur={b['dur']:4d}"
            )

    final_case = p.load_synced_case(file_name)
    idx = final_case["start_idx"]
    print("-- final plotted starts --")
    for i, j in enumerate(idx, 1):
        print(f"{i:2d}: t0={t[j]:8.3f}s  f0={f[j]:7.2f}N  th0={th[j]:7.3f}deg")
