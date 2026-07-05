import argparse
import warnings
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_object_dtype, is_string_dtype

warnings.simplefilter("ignore", FutureWarning)

DATASET_MAX_TRAIN_SIZE = {"carpk": 791}


class MetricCol(str, Enum):  # extend with other metrics from mlflow if relevant
	ENTROPY_GAIN = "metrics.entropy_gain"
	MEAN_ERROR = "metrics.mean_error"
	ENERGY_CPU = "metrics.energy_cpu"
	DURATION_SEC = "metrics.duration_sec"
	MAE = "metrics.mae"
	ENERGY_JOULE = "metrics.energy_joule"
	ENERGY_GPU = "metrics.energy_gpu"
	ENERGY_EFFICIENCY = "metrics.energy_efficiency"
	EARLY_STOPPED_AT_EPOCH = "metrics.early_stopped_at_epoch"


class ParamCol(str, Enum):  # extend with other params from mlflow if relevant
	MODEL_NAME = "params.model_name"
	MODEL_VERSION = "params.version"
	DATASET = "params.dataset_name"
	TRAIN_SIZE = "params.train_size"
	APPROACH = "params.approach"
	USER_TAG = "tags.mlflow.user"
	IMG_SIZE = "params.img_size"
	BATCH_SIZE = "params.batch_size"
	PATIENCE = "params.patience"


user_gpu_mapping = {
	"vossi": "RTX A500 Laptop GPU",
	"raphael": "RTX A500 Laptop GPU",
	"imvo807h": "NVIDIA H100-SXM5",
	"tz51aquc": "Nvidia Tesla V100",
	"ex71ezel": "Nvidia Tesla V100",
}


class Datasets(str, Enum):
	CARPK = "carpk"


def replace_train_size_False_from_runs(runs: pd.DataFrame) -> pd.DataFrame:
	# replace params.train_size False by max train_size to allow train_size plots
	assert isinstance(runs, pd.DataFrame)
	mask = runs["params.train_size"].eq("False")
	runs.loc[mask, "params.train_size"] = runs.loc[mask, "params.dataset_name"].map(DATASET_MAX_TRAIN_SIZE)
	runs["params.train_size"] = runs["params.train_size"].astype(int)
	runs.sort_values(by=[ParamCol.TRAIN_SIZE])
	return runs


def plot_scatter(
	runs: pd.DataFrame,
	x_col="metrics.mae",
	y_col="metrics.energy_joule",
	hue_param: str | None = "params.train_size",
	symbol_param: str | None = "params.dataset_name",
	dataset_name: str | None = None,
	size_param: str | None = None,
	show: bool = False,
	out_dir: Path = Path("plots"),
):
	runs_filtered = runs[runs["params.dataset_name"] == dataset_name].copy() if dataset_name else runs
	dataset_label = dataset_name if dataset_name else "all_datasets"

	x_data = runs_filtered[x_col]
	is_categorical = isinstance(x_data.dtype, pd.CategoricalDtype) or is_object_dtype(x_data) or is_string_dtype(x_data)

	if is_categorical:
		# Convert categories to numerical codes and jitter
		x_codes = pd.Categorical(x_data).codes.astype(float)
		x_codes += np.random.uniform(-0.1, 0.1, size=len(x_data))
		runs_filtered["_x_jittered"] = x_codes
		x_plot_col = "_x_jittered"
		x_tick_labels = pd.Categorical(x_data).categories
	else:
		x_plot_col = x_col
		x_tick_labels = None

	sns.scatterplot(
		data=runs_filtered,
		x=x_plot_col,
		y=y_col,
		style=symbol_param,
		palette="colorblind",
		size=size_param,
		hue=hue_param,
		legend=True,
	)

	x_label = x_col.replace("params.", "").replace("metrics.", "").replace("/", "_")
	y_label = y_col.replace("params.", "").replace("metrics.", "").replace("/", "_")
	hue_label = hue_param.replace("params.", "").replace("metrics.", "") if hue_param else ""
	plt.xlabel(x_label)
	if x_tick_labels is not None:
		# Restore original categorical labels on x-axis
		plt.xticks(ticks=range(len(x_tick_labels)), labels=x_tick_labels, rotation=45)
	else:
		plt.xticks(rotation=45)
	plt.ylabel(y_label)
	plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
	plt.tight_layout()
	out_path = out_dir / f"{x_label}-{y_label}-{hue_label}-{dataset_label}-scatter.png"
	plt.savefig(out_path)
	plt.show() if show else plt.close()


def plot_epoch_metrics_across_runs(run_ids: list[str], metric: MetricCol, show: bool = False, out_dir=Path("plots")):
	"""Line plots with epoch as x-axis."""
	metric_name = metric.value.replace("metrics.", "")
	client = (
		mlflow.MlflowClient()
	)  # does not work for copied runs even if URI is set.. workaround read file from reconstructed path?
	records = []
	for run_id in run_ids:
		metric_list = client.get_metric_history(run_id, key=metric_name)
		if not metric_list:
			continue
		for m in metric_list:
			records.append({"run_id": run_id, "step": m.step, "value": m.value})

	df = pd.DataFrame(records).sort_values(["run_id", "step"])
	plt.figure(figsize=(10, 6))
	sns.lineplot(data=df, x="step", y="value", hue="run_id", marker="o", legend=False, markersize=3)
	plt.xlabel("epoch")
	plt.ylabel(metric_name)
	plt.tight_layout()
	out_path = out_dir / f"{metric_name}-epochs.png"
	plt.savefig(out_path)
	plt.show() if show else plt.close()


def init_plot_env(mlrun_path: Path, out_dir: Path = Path("plots")):
	sns.set_theme(style="darkgrid", context="paper")
	out_dir.mkdir(exist_ok=True, parents=True)
	if mlrun_path.is_absolute():
		mlflow.set_tracking_uri(f"file://{mlrun_path}")
	else:
		mlflow.set_tracking_uri(mlrun_path)


def plot_line_with_uncertainty(
	runs: pd.DataFrame,
	x_col="params.train_size",
	y_col="metrics.energy_efficiency",
	hue_param="params.model_version",
	style_param: str | None = None,
	dataset_name: str | None = None,
	alpha: float = 0.25,
	show: bool = True,
	out_dir: Path = Path("plots"),
):
	"""from tilman plot_scatter_with_uncertainty adapted"""
	runs_filtered = runs[runs["params.dataset_name"] == dataset_name].copy() if dataset_name else runs
	dataset_label = dataset_name if dataset_name else "all_datasets"

	y_var_col = y_col + "_var"
	if y_var_col not in runs_filtered.columns:
		raise ValueError(f"Expected column '{y_var_col}' for uncertainty, but not found.")

	x_label = x_col.replace("params.", "").replace("metrics.", "").replace("/", "_")
	y_label = y_col.replace("params.", "").replace("metrics.", "").replace("/", "_")
	hue_label = hue_param.replace("params.", "").replace("metrics.", "") if hue_param else ""
	style_label = style_param.replace("params.", "").replace("metrics.", "") if style_param else ""

	plt.figure(figsize=(8, 5))
	sns.set(style="darkgrid", context="paper", palette="colorblind")

	color_palette = sns.color_palette("colorblind")
	linestyles = ["solid", "dotted", "dashed", "dashdot"]

	groupers = [hue_param]
	if style_param:
		groupers.append(style_param)

	for group_keys, group_df in runs_filtered.groupby(groupers):
		# unpack grouping keys
		if style_param:
			hue_val, style_val = group_keys  # ty: ignore
		else:
			hue_val, style_val = group_keys if not isinstance(group_keys, tuple) else group_keys[0], None

		group_df = group_df.sort_values(x_col)
		x = group_df[x_col].values
		y = group_df[y_col].values
		y_var = group_df[y_var_col].values
		y_lower = y - y_var
		y_upper = y + y_var

		color_idx = list(runs_filtered[hue_param].unique()).index(hue_val) % len(color_palette)
		color = color_palette[color_idx]
		linestyle = (
			linestyles[list(runs_filtered[style_param].unique()).index(style_val) % len(linestyles)]
			if style_param
			else "solid"
		)

		label = f"{hue_val}" if not style_param else f"{hue_val}, {style_val}"
		plt.plot(x, y, marker="o", label=label, color=color, linewidth=2, linestyle=linestyle)
		plt.fill_between(x, y_lower, y_upper, color=color, alpha=alpha, linewidth=0)

	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend()
	plt.tight_layout()

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{x_label}-{y_label}-{hue_label}-{style_label}-{dataset_label}-uncertainty.png"
	plt.savefig(out_path, dpi=300)
	if show:
		plt.show()
	else:
		plt.close()


def aggregate_experiments_by_seed(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
	"""
	from Tilman -> adapted
	Combine MLflow experiment runs that differ only by random seed.

	Groups runs by all `params.*` columns except `params.seed`,
	and aggregates all `metrics.*` columns by median (and adds std as *_var).

	Non-numeric metrics and non-param columns are taken from the first entry in each group.

	Usage: df_aggregated = aggregate_experiments_by_seed(runs, verbose=True)

	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame containing MLflow experiment runs.
	verbose : bool, optional (default=True)
		If True, prints summary information about the groups found and aggregated.

	Returns
	-------
	pd.DataFrame
		Aggregated DataFrame, one row per group of similar experiments
		(i.e., differing only by random seed).
	"""

	# --- Identify relevant columns ---
	param_cols = [c for c in df.columns if c.startswith("params.")]
	metric_cols = [c for c in df.columns if c.startswith("metrics.")]

	# Grouping columns: all params.* except params.seed -> in density map "mean" and "std" can differ
	group_cols = [
		c
		for c in param_cols
		if c != "params.seed"
		and c != "params.config_file"
		and c != "params.data_yaml"
		and c != "params.path"
		and c != "params.mean"
		and c != "params.std"
	]
	group_cols.append("tags.mlflow.user")

	# --- Prepare storage for output ---
	aggregated_rows = []

	# --- Group by all params except seed ---
	grouped = df.groupby(group_cols, dropna=False)

	if verbose:
		print(f"Found {len(grouped)} experiment groups (grouped by {len(group_cols)} params, excluding seed).")

	# --- Iterate over groups ---
	for i, (group_key, group_df) in enumerate(grouped, start=1):
		# Take representative row for non-metric columns
		base_row = group_df.iloc[0].copy()

		# Compute median and std for numeric metrics
		metric_medians = {}
		metric_means = {}
		metric_stds = {}

		for col in metric_cols:
			col_data = group_df[col]

			if pd.api.types.is_numeric_dtype(col_data):
				# Check if all values are NaN
				if col_data.notna().any():
					# Compute median
					metric_medians[col] = col_data.median()
					metric_means[col + "_mean"] = np.round(col_data.mean(), decimals=2)

					# --- Robust alternative to std (less sensitive to outliers) ---
					q75, q25 = np.percentile(col_data.dropna(), [75, 25])
					std = np.std(col_data.dropna())
					metric_stds[col + "_var"] = std  # robust_std

					# --- Original std (optional, keep commented) ---
					# metric_stds[col + "_var"] = col_data.std(ddof=1)
				else:
					# Skip the metric but log a message
					if verbose:
						print(f"  Skipping numeric metric '{col}' for this group: all values are NaN.")
					metric_medians[col] = np.nan
					metric_means[col + "_mean"] = np.nan
					metric_stds[col + "_var"] = np.nan
			else:
				# Non-numeric: keep first value
				metric_medians[col] = col_data.iloc[0]

		# Merge results into one row
		combined = base_row.to_dict()
		combined.update(metric_medians)
		combined.update(metric_stds)
		combined.update(metric_means)

		aggregated_rows.append(combined)

		# Verbose group-level output
		if verbose:
			print(f"\nGroup {i}:")
			print("  Group key (params.* except seed):")
			for k, v in zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key], strict=True):
				print(f"    {k}: {v}")
			print(f"  Contains {len(group_df)} runs with different seeds: {list(group_df['params.seed'].unique())}")

	# Combine all groups into a new DataFrame
	aggregated_df = pd.DataFrame(aggregated_rows)

	# Reset index for cleanliness
	aggregated_df.reset_index(drop=True, inplace=True)

	if verbose:
		print(f"\nAggregation complete. Output has {len(aggregated_df)} grouped experiment rows.")

	return aggregated_df


def clean_agg_runs(runs_agg, sort_by_model_version: bool = False) -> pd.DataFrame:
	runs_agg["params.version"] = runs_agg["params.version"].apply(
		lambda x: str(x).replace(
			".pt",
			"",
		)
	)
	runs_agg["metrics.energy_joule"] = runs_agg["metrics.energy_joule"] / 1000
	runs_agg.rename(
		columns={"metrics.energy_joule": "Energy [kJ]", "tags.mlflow.user": "GPU", ParamCol.DATASET: "Dataset"},
		inplace=True,
	)
	if sort_by_model_version:
		custom_order = [
			"yolov8n",
			"yolov8s",
			"yolov8m",
			"yolov8l",
			"yolo11n",
			"yolo11s",
			"yolo11m",
			"yolo11l",
			"fcrn-base",
			"SAUnet",
		]
		runs_agg["params.version"] = pd.Categorical(runs_agg["params.version"], categories=custom_order, ordered=True)
		runs_agg = runs_agg.sort_values(by=[ParamCol.MODEL_VERSION], ascending=True)
	return runs_agg


if __name__ == "__main__":
	# sample plots
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--mlflow_dir",
		default="mlruns_shared_no_artifacts",
		help="Absolute directory path to mlflow experiments. Folder should contain one subfolder per experiments \
		which contains one folder per run. Defaults to 'mlruns_shared_no_artifacts' tracked by lfs.",
	)
	parser.add_argument("--output_dir", help="location to store result plots.", default="plots")
	args = parser.parse_args()
	mlflow_path = Path(args.mlflow_dir)
	assert mlflow_path.exists()
	output_dir = Path(args.output_dir)
	init_plot_env(mlflow_path, output_dir)

	runs_model_size_tag = mlflow.search_runs(
		filter_string="tags.analyze='model_size'",
		max_results=700,
		output_format="pandas",
		experiment_names=[
			"foci_leipzig_yolo_model_size_iv",
			"carpk_leipzig_yolo_model_size_iv",
		],
	)
	runs_part2 = mlflow.search_runs(
		max_results=1000,
		output_format="pandas",
		experiment_names=[
			"tagderforschung_leipzig_rf",
			"carpk_leipzig_yolo11_model_size_iv",
			"foci_leipzig_yolo11_model_size_iv",
			"carpk_model_size_yolo_local_iv",
			"carpk_dresden_yolo_model_size_v11_iv",
			"tagderforschung_local_rf",
			"carpk_dresden_yolo_model_size_iv",
			"foci_dresden_yolo_model_size_iv",
			"foci_local_iv",
			"carpk_train_size_yolo_local_iv",
		],
	)
	assert isinstance(runs_model_size_tag, pd.DataFrame)
	assert isinstance(runs_part2, pd.DataFrame)
	# merge runs
	runs = pd.concat([runs_model_size_tag, runs_part2], ignore_index=True)
	runs["tags.mlflow.user"] = runs["tags.mlflow.user"].map(user_gpu_mapping)

	runs_agg = aggregate_experiments_by_seed(runs, verbose=False)
	runs_agg = clean_agg_runs(runs_agg, sort_by_model_version=True)

	plot_scatter(
		runs=runs_agg,
		out_dir=output_dir,
		x_col="params.version",
		size_param="Energy [kJ]",
		y_col="metrics.mae_mean",
		hue_param="GPU",
		symbol_param="Dataset",
		show=False,
	)

	plot_scatter(
		runs=runs_agg,
		out_dir=output_dir,
		x_col=ParamCol.MODEL_VERSION,
		y_col=MetricCol.ENERGY_EFFICIENCY,
		hue_param="GPU",
		symbol_param="Dataset",
		show=False,
	)
