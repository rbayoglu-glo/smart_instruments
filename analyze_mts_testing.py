#%%

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Research_Team_Data\L231147_L5-S1\6DOF")
OUTPUT_DIR = Path(r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Data_analysis\output")
TEST_LABEL = "L231147 L5-S1"
TARGET_MOMENT_NM = 7.5
AVERAGE_CURVE_POINTS = 500
SECANT_MOMENT_LOW_NM = 1.0
SECANT_MOMENT_HIGH_NM = 5.0
SECANT_CURVE_MIN_ABS_M_NM = 0.5

MOTION_CONFIG = {
	"Flexion/Extension": {
		"moment_col": "Mx (Nm)",
		"axis_label": "Mx (Nm)",
		"fit_angle_min_deg": -2.0,
		"fit_angle_max_deg": -0.5,
	},
	"Lateral Bending": {
		"moment_col": "My (Nm)",
		"axis_label": "My (Nm)",
		"fit_angle_min_deg": 0.5,
		"fit_angle_max_deg": 2.0,
	},
	"Axial Rotation": {
		"moment_col": "Mz (Nm)",
		"axis_label": "Mz (Nm)",
		"fit_angle_min_deg": 0.5,
		"fit_angle_max_deg": 1.0,
	},
}

# Intervention labels follow the existing lab naming order.
DEFAULT_INTERVENTION_NAMES = [
	"Intact",
	"PUF Left",
	"FUF Left",
	"FBF",
	"Posterior Release",
	"SPO",
]


def get_book_numbers(data_dir: Path) -> list[int]:
	book_numbers = []
	for fpath in data_dir.glob("Book*.xlsx"):
		match = re.fullmatch(r"Book(\d+)\.xlsx", fpath.name, flags=re.IGNORECASE)
		if match:
			book_numbers.append(int(match.group(1)))
	return sorted(book_numbers)


def build_intervention_map(book_numbers: list[int]) -> dict[str, dict[str, int]]:
	if len(book_numbers) % 3 != 0:
		raise ValueError(
			f"Expected book count to be divisible by 3 (FE/LB/AR per intervention), got {len(book_numbers)}"
		)

	intervention_map: dict[str, dict[str, int]] = {}
	n_interventions = len(book_numbers) // 3

	for idx in range(n_interventions):
		name = (
			DEFAULT_INTERVENTION_NAMES[idx]
			if idx < len(DEFAULT_INTERVENTION_NAMES)
			else f"Intervention {idx + 1}"
		)
		fe_book, lb_book, ar_book = book_numbers[3 * idx : 3 * idx + 3]
		intervention_map[name] = {
			"Flexion/Extension": fe_book,
			"Lateral Bending": lb_book,
			"Axial Rotation": ar_book,
		}

	return intervention_map


def read_motion_data(file_path: Path, moment_col: str) -> tuple[np.ndarray, np.ndarray]:
	df = pd.read_excel(file_path)

	angle = pd.to_numeric(df["Angle (deg)"], errors="coerce")
	moment = pd.to_numeric(df[moment_col], errors="coerce")

	# Drop NaNs and all-zero idle samples.
	mask = np.isfinite(angle) & np.isfinite(moment) & ((angle.abs() > 1e-4) | (moment.abs() > 1e-4))
	angle = angle[mask].to_numpy()
	moment = moment[mask].to_numpy()

	if angle.size < 2:
		raise ValueError(f"Not enough valid data in {file_path.name} for {moment_col}")

	return angle, moment


def find_turning_points(angle_deg: np.ndarray) -> np.ndarray:
	grad = np.diff(angle_deg)
	sign = np.sign(grad)

	for idx in range(1, sign.size):
		if sign[idx] == 0:
			sign[idx] = sign[idx - 1]
	sign[sign == 0] = 1

	return np.where(np.diff(sign) != 0)[0] + 1


def fill_zero_sign(signal: np.ndarray) -> np.ndarray:
	sign = np.sign(signal).astype(float)

	for idx in range(1, sign.size):
		if sign[idx] == 0:
			sign[idx] = sign[idx - 1]

	for idx in range(sign.size - 2, -1, -1):
		if sign[idx] == 0:
			sign[idx] = sign[idx + 1]

	sign[sign == 0] = 1
	return sign


def extract_third_cycle(
	angle_deg: np.ndarray,
	moment_nm: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray]:
	sign = fill_zero_sign(angle_deg)
	pos_crossings = np.where((sign[:-1] <= 0) & (sign[1:] > 0))[0]
	neg_crossings = np.where((sign[:-1] >= 0) & (sign[1:] < 0))[0]

	if pos_crossings.size >= 4:
		start_idx = int(pos_crossings[2] + 1)
		end_idx = int(pos_crossings[3])
	elif neg_crossings.size >= 4:
		start_idx = int(neg_crossings[2] + 1)
		end_idx = int(neg_crossings[3])
	else:
		# Fallback for traces that do not complete a final zero crossing.
		turning_points = find_turning_points(angle_deg)
		if turning_points.size >= 6:
			start_idx = int(turning_points[3])
			end_idx = int(turning_points[5])
		else:
			raise ValueError("Unable to detect third cycle boundaries.")

	if end_idx <= start_idx or (end_idx - start_idx) < 10:
		raise ValueError("Detected third cycle is too short for fitting.")

	cycle_slice = slice(start_idx, end_idx + 1)
	return start_idx, end_idx, angle_deg[cycle_slice], moment_nm[cycle_slice]


def select_loading_fit_region(
	angle_deg: np.ndarray,
	moment_nm: np.ndarray,
	angle_min_deg: float,
	angle_max_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	candidate_idx = np.where((angle_deg >= angle_min_deg) & (angle_deg <= angle_max_deg))[0]
	if candidate_idx.size < 3:
		raise ValueError(
			"Not enough data points in third-cycle fit window "
			f"[{angle_min_deg}, {angle_max_deg}] deg"
		)

	run_breaks = np.where(np.diff(candidate_idx) > 1)[0] + 1
	runs = [run for run in np.split(candidate_idx, run_breaks) if run.size >= 3]
	if not runs:
		raise ValueError("Unable to find a contiguous branch in fit window.")

	# Third-cycle traces pass through this angle window twice.
	# Use the later pass, which matches the requested loading branch.
	fit_idx = max(runs, key=lambda run: int(run[0]))

	return fit_idx, angle_deg[fit_idx], moment_nm[fit_idx]


def estimate_stiffness(angle_deg: np.ndarray, moment_nm: np.ndarray) -> tuple[float, float, float, float]:
	slope_nm_per_deg, intercept_nm = np.polyfit(angle_deg, moment_nm, 1)
	moment_fit = slope_nm_per_deg * angle_deg + intercept_nm

	ss_res = np.sum((moment_nm - moment_fit) ** 2)
	ss_tot = np.sum((moment_nm - np.mean(moment_nm)) ** 2)
	r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

	return (
		abs(float(slope_nm_per_deg)),
		float(slope_nm_per_deg),
		float(intercept_nm),
		float(r2),
	)


def fit_cubic_curve_no_c0(angle_deg: np.ndarray, moment_nm: np.ndarray) -> tuple[np.ndarray, float]:
	# Constrained cubic fit without intercept: M = C1*theta + C2*theta^2 + C3*theta^3
	X = np.column_stack((angle_deg, angle_deg**2, angle_deg**3))
	coeff, *_ = np.linalg.lstsq(X, moment_nm, rcond=None)
	moment_fit = X @ coeff

	ss_res = np.sum((moment_nm - moment_fit) ** 2)
	ss_tot = np.sum((moment_nm - np.mean(moment_nm)) ** 2)
	r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

	return coeff.astype(float), float(r2)


def extract_branch_containing_fit_region(
	angle_cycle: np.ndarray,
	moment_cycle: np.ndarray,
	fit_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	turning_points = find_turning_points(angle_cycle)
	boundaries = np.concatenate(([0], turning_points, [angle_cycle.size - 1]))

	fit_start = int(np.min(fit_idx))
	fit_end = int(np.max(fit_idx))

	for start, end in zip(boundaries[:-1], boundaries[1:]):
		if fit_start >= start and fit_end <= end:
			branch_slice = slice(int(start), int(end) + 1)
			return angle_cycle[branch_slice], moment_cycle[branch_slice]

	# Fallback if segmentation fails.
	return angle_cycle, moment_cycle


def extract_positive_loading_branch(
	angle_cycle: np.ndarray,
	moment_cycle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	if moment_cycle.size < 3:
		return angle_cycle, moment_cycle

	peak_idx = int(np.argmax(moment_cycle))
	if peak_idx < 2:
		return angle_cycle, moment_cycle

	branch_slice = slice(0, peak_idx + 1)
	return angle_cycle[branch_slice], moment_cycle[branch_slice]


def _prepare_monotonic_curve(moment_nm: np.ndarray, angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	order = np.argsort(moment_nm)
	m_sorted = moment_nm[order]
	a_sorted = angle_deg[order]

	# Collapse repeated moment values using average angle at each repeated moment.
	m_unique, inv = np.unique(m_sorted, return_inverse=True)
	a_unique = np.zeros_like(m_unique, dtype=float)
	counts = np.bincount(inv)
	np.add.at(a_unique, inv, a_sorted)
	a_unique = a_unique / np.maximum(counts, 1)

	return m_unique, a_unique


def _cyclic_segment(values: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
	if start_idx <= end_idx:
		idx = np.arange(start_idx, end_idx + 1)
	else:
		idx = np.concatenate((np.arange(start_idx, values.size), np.arange(0, end_idx + 1)))
	return values[idx]


def _moving_average(signal: np.ndarray, window: int = 11) -> np.ndarray:
	if signal.size < 5:
		return signal

	window = min(window, signal.size if signal.size % 2 == 1 else signal.size - 1)
	if window < 3:
		return signal

	pad = window // 2
	kernel = np.ones(window, dtype=float) / float(window)
	padded = np.pad(signal, (pad, pad), mode="edge")
	return np.convolve(padded, kernel, mode="valid")


def compute_centered_average_curve(
	angle_cycle: np.ndarray,
	moment_cycle: np.ndarray,
	n_points: int = AVERAGE_CURVE_POINTS,
) -> tuple[np.ndarray, np.ndarray, float, float]:
	if angle_cycle.size < 20 or moment_cycle.size < 20:
		raise ValueError("Not enough samples in cycle to average loading/unloading branches.")

	# Use major extrema to split the cycle into two full traversals between min and max moment.
	i_max = int(np.argmax(moment_cycle))
	i_min = int(np.argmin(moment_cycle))

	if i_max == i_min:
		raise ValueError("Unable to detect distinct extrema for branch averaging.")

	m_branch_a = _cyclic_segment(moment_cycle, i_min, i_max)
	a_branch_a = _cyclic_segment(angle_cycle, i_min, i_max)
	m_branch_b = _cyclic_segment(moment_cycle, i_max, i_min)
	a_branch_b = _cyclic_segment(angle_cycle, i_max, i_min)

	m_asc, a_asc = _prepare_monotonic_curve(m_branch_a, a_branch_a)
	m_desc, a_desc = _prepare_monotonic_curve(m_branch_b, a_branch_b)

	m_min = max(float(np.min(m_asc)), float(np.min(m_desc)))
	m_max = min(float(np.max(m_asc)), float(np.max(m_desc)))
	if m_max <= m_min:
		raise ValueError("Loading/unloading branches do not overlap in moment domain.")

	m_common = np.linspace(m_min, m_max, n_points)
	a_asc_i = np.interp(m_common, m_asc, a_asc)
	a_desc_i = np.interp(m_common, m_desc, a_desc)
	a_avg = 0.5 * (a_asc_i + a_desc_i)
	a_avg = _moving_average(a_avg, window=11)

	if m_min <= 0.0 <= m_max:
		theta_at_zero_m = float(np.interp(0.0, m_common, a_avg))
		moment_offset = 0.0
	else:
		zero_idx = int(np.argmin(np.abs(m_common)))
		theta_at_zero_m = float(a_avg[zero_idx])
		moment_offset = float(m_common[zero_idx])

	m_centered = m_common - moment_offset
	a_centered = a_avg - theta_at_zero_m

	return m_centered, a_centered, moment_offset, theta_at_zero_m


def select_fit_region_on_curve(
	angle_deg: np.ndarray,
	moment_nm: np.ndarray,
	angle_min_deg: float,
	angle_max_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	candidate_idx = np.where((angle_deg >= angle_min_deg) & (angle_deg <= angle_max_deg))[0]
	if candidate_idx.size < 3:
		raise ValueError(
			"Not enough data points in averaged-curve fit window "
			f"[{angle_min_deg}, {angle_max_deg}] deg"
		)

	run_breaks = np.where(np.diff(candidate_idx) > 1)[0] + 1
	runs = [run for run in np.split(candidate_idx, run_breaks) if run.size >= 3]
	if runs:
		fit_idx = max(runs, key=lambda run: run.size)
	else:
		# Fallback for sparse/discontinuous indexing from interpolation grid.
		fit_idx = candidate_idx
	return fit_idx, angle_deg[fit_idx], moment_nm[fit_idx]


def estimate_angle_at_target_moment(
	angle_deg: np.ndarray,
	moment_nm: np.ndarray,
	target_moment_nm: float,
) -> float:
	if angle_deg.size < 2 or moment_nm.size < 2:
		return float("nan")

	m_min = float(np.min(moment_nm))
	m_max = float(np.max(moment_nm))
	if target_moment_nm < m_min or target_moment_nm > m_max:
		return float("nan")

	sort_idx = np.argsort(moment_nm)
	m_sorted = moment_nm[sort_idx]
	a_sorted = angle_deg[sort_idx]

	m_unique, first_idx = np.unique(m_sorted, return_index=True)
	a_unique = a_sorted[first_idx]

	if m_unique.size < 2:
		return float("nan")

	return float(np.interp(target_moment_nm, m_unique, a_unique))


def compute_secant_stiffness_metrics(
	angle_deg: np.ndarray,
	moment_nm: np.ndarray,
	m_low_nm: float,
	m_high_nm: float,
	n_points: int = 60,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
	sort_idx = np.argsort(moment_nm)
	m_sorted = moment_nm[sort_idx]
	a_sorted = angle_deg[sort_idx]

	m_unique, first_idx = np.unique(m_sorted, return_index=True)
	a_unique = a_sorted[first_idx]

	if m_unique.size < 2:
		return float("nan"), float("nan"), float("nan"), np.array([]), np.array([])

	m_min = float(np.min(m_unique))
	m_max = float(np.max(m_unique))

	if m_low_nm < m_min or m_high_nm > m_max:
		return float("nan"), float("nan"), float("nan"), np.array([]), np.array([])

	theta_low = float(np.interp(m_low_nm, m_unique, a_unique))
	theta_high = float(np.interp(m_high_nm, m_unique, a_unique))

	dtheta = theta_high - theta_low
	if abs(dtheta) < 1e-10:
		k_pm = float("nan")
	else:
		k_pm = float((m_high_nm - m_low_nm) / dtheta)

	m_abs_max = min(abs(m_low_nm), abs(m_high_nm))
	m_abs_vals = np.linspace(SECANT_CURVE_MIN_ABS_M_NM, m_abs_max, n_points)

	m_curve = []
	k_curve = []
	for m_abs in m_abs_vals:
		if (-m_abs) < m_min or m_abs > m_max:
			continue

		theta_neg = float(np.interp(-m_abs, m_unique, a_unique))
		theta_pos = float(np.interp(m_abs, m_unique, a_unique))
		dtheta_pm = theta_pos - theta_neg
		if abs(dtheta_pm) < 1e-10:
			continue

		k_val = float((2.0 * m_abs) / dtheta_pm)
		m_curve.append(m_abs)
		k_curve.append(k_val)

	return k_pm, theta_low, theta_high, np.asarray(m_curve, dtype=float), np.asarray(k_curve, dtype=float)


def sanitize_filename(name: str) -> str:
	return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def secant_window_label() -> str:
	return f"[{SECANT_MOMENT_LOW_NM:.1f},{SECANT_MOMENT_HIGH_NM:.1f}]"


def plot_intervention(
	intervention: str,
	motion_results: dict[str, dict[str, np.ndarray | float]],
	output_dir: Path,
) -> None:
	fig, axes = plt.subplots(1, 3, figsize=(18, 5))

	for ax, motion_name in zip(axes, MOTION_CONFIG.keys()):
		result = motion_results[motion_name]
		angle = result["angle_cycle_centered"]
		moment = result["moment_cycle_centered"]
		angle_avg = result["angle_avg_centered"]
		moment_avg = result["moment_avg_centered"]
		poly3_coeff = result["poly3_coeff"]
		axis_label = MOTION_CONFIG[motion_name]["axis_label"]

		ax.plot(angle, moment, marker="o", markersize=1.8, linewidth=1.0, alpha=0.4, label="Measured (3rd cycle)")
		ax.plot(angle_avg, moment_avg, color="red", linewidth=2.2, label="Avg(load/unload), centered")

		theta_poly = np.linspace(float(np.min(angle_avg)), float(np.max(angle_avg)), 300)
		moment_poly = (
			poly3_coeff[0] * theta_poly
			+ poly3_coeff[1] * theta_poly**2
			+ poly3_coeff[2] * theta_poly**3
		)
		ax.plot(theta_poly, moment_poly, color="tab:green", linestyle="-.", linewidth=1.8, label="3rd-order poly fit (no C0)")

		ax.set_xlabel("Angle (deg)", fontsize=11)
		ax.set_ylabel(axis_label, fontsize=11)
		ax.set_title(motion_name, fontsize=11)
		ax.grid(True, linestyle="--", alpha=0.6)
		ax.axhline(0, color="k", linewidth=0.8)
		ax.axvline(0, color="k", linewidth=0.8)
		ax.legend(loc="lower right", fontsize=8)

	fig.suptitle(f"{intervention} Stiffness - {TEST_LABEL}", fontsize=14)
	plt.tight_layout()
	output_name = f"{sanitize_filename(intervention)}_stiffness.png"
	fig.savefig(output_dir / output_name, dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_all_interventions_by_motion(
	all_motion_results: dict[str, dict[str, dict[str, np.ndarray | float]]],
	output_dir: Path,
) -> None:
	for motion_name, cfg in MOTION_CONFIG.items():
		fig, ax = plt.subplots(figsize=(8, 6))

		for intervention, motion_results in all_motion_results.items():
			if motion_name not in motion_results:
				continue

			angle_avg = np.asarray(motion_results[motion_name]["angle_avg_centered"], dtype=float)
			moment_avg = np.asarray(motion_results[motion_name]["moment_avg_centered"], dtype=float)

			if angle_avg.size < 2 or moment_avg.size < 2:
				continue

			ax.plot(
				angle_avg,
				moment_avg,
				linewidth=2.0,
				label=intervention,
			)

		ax.set_xlabel("Angle (deg)", fontsize=11)
		ax.set_ylabel(cfg["axis_label"], fontsize=11)
		ax.set_title(f"{motion_name} | Avg(load/unload), centered", fontsize=12)
		ax.grid(True, linestyle="--", alpha=0.6)
		ax.axhline(0, color="k", linewidth=0.8)
		ax.axvline(0, color="k", linewidth=0.8)
		ax.legend(loc="upper left", fontsize=9)

		out_name = f"all_interventions_{sanitize_filename(motion_name)}_avg_centered.png"
		fig.tight_layout()
		fig.savefig(output_dir / out_name, dpi=300, bbox_inches="tight")
		plt.close(fig)


def plot_coefficient_illustration(
	intervention: str,
	fe_result: dict[str, np.ndarray | float],
	output_dir: Path,
) -> None:
	"""Create one FE illustration that explains C1/C2/C3 contributions."""
	angle = np.asarray(fe_result["angle_cycle_centered"], dtype=float)
	moment = np.asarray(fe_result["moment_cycle_centered"], dtype=float)
	angle_avg = np.asarray(fe_result["angle_avg_centered"], dtype=float)
	moment_avg = np.asarray(fe_result["moment_avg_centered"], dtype=float)
	poly3_coeff = np.asarray(fe_result["poly3_coeff"], dtype=float)

	if angle_avg.size < 3:
		return

	theta_poly = np.linspace(float(np.min(angle_avg)), float(np.max(angle_avg)), 400)
	m1 = poly3_coeff[0] * theta_poly
	m2 = poly3_coeff[1] * theta_poly**2
	m3 = poly3_coeff[2] * theta_poly**3
	m_total = m1 + m2 + m3

	fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=220)

	# Left: same style as FE angle-vs-Mx plot (Mx predicted from angle).
	ax0 = axes[0]
	ax0.plot(angle, moment, marker="o", markersize=1.8, linewidth=1.0, alpha=0.35, label="Measured (3rd cycle)")
	ax0.plot(angle_avg, moment_avg, color="red", linewidth=2.2, label="Avg(load/unload), centered")
	ax0.plot(theta_poly, m_total, color="tab:green", linestyle="-.", linewidth=1.9, label="3rd-order poly fit")
	ax0.set_xlabel("Angle (deg)")
	ax0.set_ylabel("Mx (Nm)")
	ax0.set_title(f"Flexion/Extension | {intervention}")
	ax0.grid(True, linestyle="--", alpha=0.6)
	ax0.axhline(0, color="k", linewidth=0.8)
	ax0.axvline(0, color="k", linewidth=0.8)
	ax0.legend(loc="lower right", fontsize=8)

	# Right: illustrate each polynomial contribution with angle on x and Mx on y.
	ax1 = axes[1]
	ax1.plot(theta_poly, m1, color="#1f77b4", linewidth=1.8, label="C1*theta (linear baseline)")
	ax1.plot(theta_poly, m2, color="#ff7f0e", linewidth=1.8, label="C2*theta^2 (curvature)")
	ax1.plot(theta_poly, m3, color="#9467bd", linewidth=1.8, label="C3*theta^3 (high-angle nonlinearity)")
	ax1.plot(theta_poly, m_total, color="tab:green", linestyle="-.", linewidth=2.0, label="C1*theta + C2*theta^2 + C3*theta^3")
	ax1.set_xlabel("Angle (deg)")
	ax1.set_ylabel("Mx contribution (Nm)")
	ax1.set_title("How C1, C2, C3 shape Mx(theta)")
	ax1.grid(True, linestyle="--", alpha=0.6)
	ax1.axhline(0, color="k", linewidth=0.8)
	ax1.axvline(0, color="k", linewidth=0.8)
	ax1.legend(loc="lower right", fontsize=8)

	coeff_text = (
		f"C1={poly3_coeff[0]:.3f}\n"
		f"C2={poly3_coeff[1]:.3f}\n"
		f"C3={poly3_coeff[2]:.3f}"
	)
	ax1.text(
		0.03,
		0.97,
		coeff_text,
		transform=ax1.transAxes,
		fontsize=8,
		verticalalignment="top",
		bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
	)

	fig.suptitle("Coefficient Illustration on FE Angle-vs-Mx Plot", fontsize=13)
	fig.tight_layout()
	fig.savefig(output_dir / "fe_coefficient_illustration.png", bbox_inches="tight")
	plt.close(fig)


def print_rankings(stiffness_df: pd.DataFrame) -> None:
	print("\n" + "=" * 72)
	print(f"Secant stiffness between {SECANT_MOMENT_LOW_NM:.1f} and {SECANT_MOMENT_HIGH_NM:.1f} Nm")
	print("(Higher secant stiffness = stiffer)")
	print("=" * 72)

	for motion_name in MOTION_CONFIG.keys():
		ranked_secant = (
			stiffness_df[stiffness_df["Motion"] == motion_name]
			.sort_values("SecantStiffness_Nm_per_deg", ascending=False, na_position="last")
			.reset_index(drop=True)
		)

		print(f"\n{motion_name}:")
		for idx, row in ranked_secant.iterrows():
			k_sec = row["SecantStiffness_Nm_per_deg"]
			if pd.isna(k_sec):
				print(f"  {idx + 1:>2}. {row['Intervention']:<18} k_sec{secant_window_label()} = N/A")
			else:
				print(f"  {idx + 1:>2}. {row['Intervention']:<18} k_sec{secant_window_label()} = {k_sec:.4f} Nm/deg")

	overall_secant = (
		stiffness_df.groupby("Intervention", as_index=False)["SecantStiffness_Nm_per_deg"]
		.mean()
		.rename(columns={"SecantStiffness_Nm_per_deg": "Overall_SecantStiffness_Nm_per_deg"})
		.sort_values("Overall_SecantStiffness_Nm_per_deg", ascending=False)
		.reset_index(drop=True)
	)

	print("\n" + "=" * 72)
	print("Overall Ranking (mean secant stiffness across FE, LB, AR)")
	print("=" * 72)
	for idx, row in overall_secant.iterrows():
		print(
			f"  {idx + 1:>2}. {row['Intervention']:<18} "
			f"k_sec_mean = {row['Overall_SecantStiffness_Nm_per_deg']:.4f} Nm/deg"
		)


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	print(f"Saving charts to: {OUTPUT_DIR}")

	book_numbers = get_book_numbers(DATA_DIR)
	interventions = build_intervention_map(book_numbers)

	stiffness_rows: list[dict[str, float | str]] = []
	all_motion_results: dict[str, dict[str, dict[str, np.ndarray | float]]] = {}

	for intervention, books in interventions.items():
		motion_results: dict[str, dict[str, np.ndarray | float]] = {}

		for motion_name, cfg in MOTION_CONFIG.items():
			book_num = books[motion_name]
			file_path = DATA_DIR / f"Book{book_num}.xlsx"
			fit_angle_min_deg = float(cfg["fit_angle_min_deg"])
			fit_angle_max_deg = float(cfg["fit_angle_max_deg"])
			angle, moment = read_motion_data(file_path, cfg["moment_col"])
			cycle_start_idx, cycle_end_idx, angle_cycle, moment_cycle = extract_third_cycle(angle, moment)

			moment_avg_centered, angle_avg_centered, moment_offset, angle_offset = compute_centered_average_curve(
				angle_cycle,
				moment_cycle,
			)
			moment_cycle_centered = moment_cycle - moment_offset
			angle_cycle_centered = angle_cycle - angle_offset

			fit_idx, angle_fit_points, moment_fit_points = select_fit_region_on_curve(
				angle_avg_centered,
				moment_avg_centered,
				fit_angle_min_deg,
				fit_angle_max_deg,
			)
			theta_target_deg = estimate_angle_at_target_moment(
				angle_avg_centered,
				moment_avg_centered,
				TARGET_MOMENT_NM,
			)
			secant_k, theta_m_low, theta_m_high, m_sec_curve, k_sec_curve = compute_secant_stiffness_metrics(
				angle_avg_centered,
				moment_avg_centered,
				SECANT_MOMENT_LOW_NM,
				SECANT_MOMENT_HIGH_NM,
			)
			k_abs, k_signed, intercept_nm, r2 = estimate_stiffness(angle_fit_points, moment_fit_points)
			poly3_coeff, poly3_r2 = fit_cubic_curve_no_c0(angle_avg_centered, moment_avg_centered)

			motion_results[motion_name] = {
				"angle_cycle_centered": angle_cycle_centered,
				"moment_cycle_centered": moment_cycle_centered,
				"angle_avg_centered": angle_avg_centered,
				"moment_avg_centered": moment_avg_centered,
				"moment_offset": moment_offset,
				"angle_offset": angle_offset,
				"cycle_start_idx": cycle_start_idx,
				"cycle_end_idx": cycle_end_idx,
				"fit_idx": fit_idx,
				"angle_fit_points": angle_fit_points,
				"moment_fit_points": moment_fit_points,
				"theta_at_target_moment_deg": theta_target_deg,
				"target_moment_nm": TARGET_MOMENT_NM,
				"secant_k": secant_k,
				"theta_m_low": theta_m_low,
				"theta_m_high": theta_m_high,
				"m_sec_curve": m_sec_curve,
				"k_sec_curve": k_sec_curve,
				"fit_angle_min_deg": fit_angle_min_deg,
				"fit_angle_max_deg": fit_angle_max_deg,
				"k_abs": k_abs,
				"k_signed": k_signed,
				"intercept_nm": intercept_nm,
				"r2": r2,
				"poly3_coeff": poly3_coeff,
				"poly3_r2": poly3_r2,
			}

			stiffness_rows.append(
				{
					"Intervention": intervention,
					"Motion": motion_name,
					"Book": book_num,
					"Stiffness_Nm_per_deg": k_abs,
					"Signed_Slope_Nm_per_deg": k_signed,
					"Theta_at_target_moment_deg": theta_target_deg,
					"SecantStiffness_Nm_per_deg": secant_k,
					"Intercept_Nm": intercept_nm,
					"R2": r2,
				}
			)

		all_motion_results[intervention] = motion_results

		plot_intervention(intervention, motion_results, OUTPUT_DIR)

	plot_all_interventions_by_motion(all_motion_results, OUTPUT_DIR)

	stiffness_df = pd.DataFrame(stiffness_rows)
	print_rankings(stiffness_df)


if __name__ == "__main__":
	main()

