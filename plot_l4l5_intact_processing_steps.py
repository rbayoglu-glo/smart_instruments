"""Regenerate raw-to-ramp-start plots for the configured intact level."""

from pathlib import Path
import importlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEGACY_ANALYSIS_DIR = Path(
	r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis"
)
PILOT_ANALYSIS_DIR = Path(
	r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis"
)
SEGMENT_ROOT = Path(
	r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\UTN24_26016\segment\spreader"
)
FORCE_DIR = SEGMENT_ROOT / "Force" / "lamspreader"
TRACKING_DIR = SEGMENT_ROOT / "Displacement" / "gns-sensorhub"
OUT_DIR = Path(
	r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis\output"
)
LEVEL_NAME = "L4-L5"
OUTPUT_PREFIX = ""
RELATIVE_TIME_MODE = False
TRACKING_START_OFFSETS_S = {}

TRIALS = {
	"L4-L5 Left": {
		"force_file": "Session #80.csv",
		"tracking_folder": "2026-12-17_[01-56-11]",
	},
	"L4-L5 Right": {
		"force_file": "Session #81.csv",
		"tracking_folder": "2026-12-17_[02-05-40]",
	},
}

def configure(module) -> None:
	module.FORCE_DIR = FORCE_DIR
	module.OUT_DIR = OUT_DIR
	module.TRIALS = TRIALS
	module.TRACK_DIR = TRACKING_DIR
	module.TRACKING_DIR = TRACKING_DIR
	module.RELATIVE_TIME_MODE = RELATIVE_TIME_MODE
	module.TRACKING_START_OFFSETS_S = TRACKING_START_OFFSETS_S


def plot_lumbar_force_review(records: list[dict], output_dir: Path) -> Path:
	from analyze_l4l5_intact_prior_method import load_force

	lumbar_records = [record for record in records if record["worksheet"] == "Lumbar, Prone Specimen"]
	if len(lumbar_records) != 8:
		raise ValueError("The lumbar force review requires all eight workbook level/side records.")
	fig, axes = plt.subplots(4, 2, figsize=(14.0, 12.0), squeeze=False)
	for ax, record in zip(axes.ravel(), lumbar_records):
		force_t, force_raw, force_filtered = load_force(Path(record["source_force_csv"]))
		elapsed_t = force_t - force_t[0]
		ax.plot(elapsed_t, force_raw, color="0.55", lw=0.6, label="raw, direction-corrected")
		ax.plot(elapsed_t, force_filtered, color="tab:blue", lw=1.0, label="filtered")
		ax.axhline(0.0, color="0.4", lw=0.7, ls="--")
		ax.set_title(f"{record['spine_level']} {record['side']}", loc="left", fontsize=10)
		ax.set_ylabel("force [N]")
		ax.set_xlabel("elapsed force time [s]")
		ax.grid(alpha=0.25)
		if ax is axes[0, 0]:
			ax.legend(loc="upper left", fontsize=8)
	fig.suptitle(
		"UTN24_26016: lumbar intraoperative force review | acquisition 2026-09-04\n"
		"Force only: tracking pairs unresolved; no theta or cycle detection",
		fontsize=12,
	)
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	output_path = output_dir / "Lumbar_intraoperative_force_preflight.png"
	fig.savefig(output_path, dpi=160)
	plt.close(fig)
	print(f"saved {output_path}")
	return output_path


def plot_updated_baseline_and_ramp(analysis_module, break_tracking_gaps) -> None:
	from plot_sync_check_all_levels_sides_stacked_true_raw_filtered_baseline_rampstart import find_force_peaks

	results = {
		trial_name: analysis_module.process_trial(trial_name)[0]
		for trial_name in TRIALS
	}

	fig, axes = plt.subplots(len(TRIALS), 1, figsize=(14.0, 6.0), squeeze=False)
	axes = axes.ravel()
	for ax, trial_name in zip(axes, TRIALS):
		info = results[trial_name]
		force_t = info["force_t_sync"]
		force_dynamic = info["force_dynamic"]
		theta_t = info["theta_t"]
		theta_dynamic = info["theta_dynamic"]
		h_force, = ax.plot(force_t, force_dynamic, color="tab:blue", lw=1.0, label="force")
		ax.set_ylabel("force [N]", color="tab:blue")
		ax.set_ylim(-25.0, 150.0)
		ax.tick_params(axis="y", labelcolor="tab:blue")
		ax.grid(alpha=0.3)
		ax2 = ax.twinx()
		theta_t_brk, theta_brk = break_tracking_gaps(theta_t, theta_dynamic)
		h_theta, = ax2.plot(theta_t_brk, theta_brk, color="tab:orange", lw=0.95, label="theta")
		ax2.axhline(0.0, color="tab:orange", lw=0.9, ls="--", alpha=0.8)
		ax2.set_ylabel("theta [deg]", color="tab:orange")
		ax2.set_ylim(-4.0, 7.0)
		ax2.tick_params(axis="y", labelcolor="tab:orange")
		window_start, window_end = info["force_baseline_window_s"]
		ax.set_title(
			f"{trial_name} | force baseline {info['force_baseline_N']:+.2f} N "
			f"from [{window_start:.2f}, {window_end:.2f}) s | theta0 {info['theta0_deg']:+.2f} deg",
			loc="left", fontsize=10,
		)
		if ax is axes[0]:
			ax.legend([h_force, h_theta], ["force", "theta"], loc="upper left", fontsize=8)
	fig.suptitle(
		f"{LEVEL_NAME} intact: filtered + pre-cycle baseline-subtracted sync\n"
		"Baseline = median of the stable 1 s window ending 0.25 s before the first ramp.",
		fontsize=12,
	)
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	fig.savefig(OUT_DIR / f"Step_03_{OUTPUT_PREFIX}{LEVEL_NAME}_true_raw_filtered_baseline_sync_stacked_shpose_only.png", dpi=160)
	plt.close(fig)

	fig, axes = plt.subplots(len(TRIALS), 1, figsize=(14.0, 6.0), squeeze=False)
	axes = axes.ravel()
	for ax, trial_name in zip(axes, TRIALS):
		info = results[trial_name]
		force_t = info["force_t_sync"]
		force_dynamic = info["force_dynamic"]
		theta_t = info["theta_t"]
		theta_dynamic = info["theta_dynamic"]
		theta_on_force = info["theta_on_force"]
		candidates = info["candidates"]
		peaks = find_force_peaks(
			force_t, force_dynamic, float(np.nanmax(force_dynamic))
		)
		accepted_ids = {int(candidate["cycle_id"]) for candidate in candidates}
		invalid_peaks = [
			(cycle_id, int(peak))
			for cycle_id, peak in enumerate(peaks, start=1)
			if cycle_id not in accepted_ids
		]
		h_force, = ax.plot(force_t, force_dynamic, color="tab:blue", lw=1.0, label="force")
		ax.set_ylabel("force [N]", color="tab:blue")
		ax.set_ylim(-25.0, 150.0)
		ax.tick_params(axis="y", labelcolor="tab:blue")
		ax.grid(alpha=0.3)
		ax2 = ax.twinx()
		theta_t_brk, theta_brk = break_tracking_gaps(theta_t, theta_dynamic)
		h_theta, = ax2.plot(theta_t_brk, theta_brk, color="tab:orange", lw=0.95, label="theta")
		ax2.axhline(0.0, color="tab:orange", lw=0.9, ls="--", alpha=0.8)
		ax2.set_ylabel("theta [deg]", color="tab:orange")
		ax2.set_ylim(-4.0, 7.0)
		ax2.tick_params(axis="y", labelcolor="tab:orange")
		starts = np.asarray([candidate["i0"] for candidate in candidates], dtype=int)
		start_t = force_t[starts] if starts.size else np.array([])
		start_force = force_dynamic[starts] if starts.size else np.array([])
		start_theta = theta_on_force[starts] if starts.size else np.array([])
		h_start_force = ax.scatter(
			start_t, start_force, s=28, marker="o", facecolor="white",
			edgecolor="tab:red", label="ramp start (force)",
		)
		h_start_theta = ax2.scatter(
			start_t, start_theta, s=28, marker="D", facecolor="white",
			edgecolor="tab:red", label="ramp start (theta)",
		)
		invalid_indices = np.asarray([peak for _, peak in invalid_peaks], dtype=int)
		h_invalid_force = ax.scatter(
			force_t[invalid_indices], force_dynamic[invalid_indices], s=34,
			marker="x", color="tab:purple", zorder=6,
			label="invalid cycle (force peak)",
		)
		h_invalid_theta = ax2.scatter(
			force_t[invalid_indices], theta_on_force[invalid_indices], s=34,
			marker="x", color="tab:purple", zorder=6,
			label="invalid cycle (theta)",
		)
		for candidate in candidates:
			idx = int(candidate["i0"])
			ax2.annotate(
				f"C{candidate['cycle_id']}",
				(force_t[idx], theta_on_force[idx]),
				xytext=(0, 5), textcoords="offset points",
				ha="center", va="bottom", fontsize=7,
				color="tab:red", zorder=7,
			)
		for cycle_id, peak in invalid_peaks:
			ax.annotate(
				f"C{cycle_id}",
				(force_t[peak], force_dynamic[peak]),
				xytext=(0, 5), textcoords="offset points",
				ha="center", va="bottom", fontsize=7,
				color="tab:purple", zorder=7,
			)
		window_start, window_end = info["force_baseline_window_s"]
		ax.set_title(
			f"{trial_name} | {len(candidates)} accepted, {len(invalid_peaks)} invalid cycles | "
			f"baseline {info['force_baseline_N']:+.2f} N "
			f"from [{window_start:.2f}, {window_end:.2f}) s",
			loc="left", fontsize=10,
		)
		if ax is axes[0]:
			ax.legend(
				[h_force, h_theta, h_start_force, h_start_theta, h_invalid_force, h_invalid_theta],
				[
					"force", "theta", "ramp start (force)", "ramp start (theta)",
					"invalid cycle (force peak)", "invalid cycle (theta)",
				],
				loc="upper left", fontsize=8,
			)
	fig.suptitle(
		f"{LEVEL_NAME} intact: ramp starts after pre-cycle force baseline subtraction",
		fontsize=12,
	)
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	fig.savefig(OUT_DIR / f"Step_04_{OUTPUT_PREFIX}{LEVEL_NAME}_true_raw_filtered_baseline_rampstart_stacked_shpose_only.png", dpi=160)
	plt.close(fig)


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	sys.path.insert(0, str(PILOT_ANALYSIS_DIR))
	sys.path.insert(0, str(LEGACY_ANALYSIS_DIR))

	modules = [
		importlib.import_module("plot_l4l5_true_raw_sync_stacked"),
		importlib.import_module("plot_l4l5_true_raw_filtered_sync_stacked"),
	]
	for module in modules:
		configure(module)
		module.main()

	analysis_module = importlib.import_module("analyze_l4l5_intact_prior_method")
	from lamina_spreader_lr import break_tracking_gaps
	plot_updated_baseline_and_ramp(analysis_module, break_tracking_gaps)

	stage_files = [
		"L4-L5_true_raw_sync_stacked_shpose_only.png",
		"L4-L5_true_raw_filtered_sync_stacked_shpose_only.png",
	]
	for step, filename in enumerate(stage_files, start=1):
		source = OUT_DIR / filename
		target_filename = filename.replace("L4-L5", f"{OUTPUT_PREFIX}{LEVEL_NAME}", 1)
		target = OUT_DIR / f"Step_{step:02d}_{target_filename}"
		source.replace(target)

	print(f"saved {LEVEL_NAME} processing-stage plots to {OUT_DIR}")


if __name__ == "__main__":
	main()
