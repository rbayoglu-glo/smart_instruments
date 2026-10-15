import contextlib
import io
import sys
from pathlib import Path

analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

import lamina_spreader_lr as L
import plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart as R

for name in L.TRIALS:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            case = R.process_trial(name)
        print(f"RESULT {name}: n_cycles={case['n_cycles']} force_baseline={case['force_baseline']:.3f} theta0={case['theta0']:.3f}")
    except Exception as exc:  # report and continue so every trial is listed
        print(f"RESULT {name}: ERROR {type(exc).__name__}: {exc}")
