#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MetricTransform = Callable[[pd.DataFrame], np.ndarray]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = args.title_suffix or embedding_title_suffix(
        read_embedding_dimension(args.embeddings)
    )

    print(f"reading metrics: {args.metrics}")
    metrics = pd.read_csv(args.metrics, sep="\t", low_memory=False)
    if "index" not in metrics.columns:
        raise SystemExit(f"{args.metrics} does not contain an index column")
    metrics = metrics.set_index("index", drop=False)

    methods = []
    for method in ["tsne", "tmap"]:
        coordinates = args.plot_dir / f"{method}_coordinates.tsv"
        if coordinates.exists():
            methods.append((method, coordinates))
    if not methods:
        raise SystemExit(f"no tsne/tmap coordinate TSV files found in {args.plot_dir}")

    metric_specs = available_metrics(metrics)
    for method, coordinates_path in methods:
        print(f"aligning {method}: {coordinates_path}")
        coordinates = read_coordinates(coordinates_path)
        x = coordinates["x"].to_numpy(dtype=np.float64)
        y = coordinates["y"].to_numpy(dtype=np.float64)
        row_indices = coordinates["index"].to_numpy(dtype=np.int64)
        point_metrics = metrics.loc[row_indices]
        if len(point_metrics) != len(x):
            raise RuntimeError(f"{method} point/metric length mismatch")

        write_density_heatmap(
            args.output_dir / f"{method}_density.png",
            x,
            y,
            f"{method.upper()} hexbin point density\n{title_suffix}",
            args.bins,
            args.dpi,
        )
        for name, title, transform in metric_specs:
            values = transform(point_metrics)
            output = args.output_dir / f"{method}_{slugify(name)}.png"
            write_metric_heatmap(
                output,
                x,
                y,
                values,
                f"{method.upper()} hexbin mean {title}\n{title_suffix}",
                title,
                args.bins,
                args.min_count,
                args.dpi,
            )
    print(f"wrote heatmaps under {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("runs/annotated-ms2-flat/plots"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("runs/annotated-ms2-flat/embeddings.tsv"),
        help="Kept for command compatibility; coordinates are loaded from --plot-dir.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/annotated-ms2-flat/spectrum_metrics.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/annotated-ms2-flat/plots/metric_heatmaps"),
    )
    parser.add_argument("--bins", type=int, default=180, help="hexbin grid size")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--title-suffix",
        default=None,
        help="Optional second title line. Defaults to the inferred embedding dimension.",
    )
    return parser.parse_args()


def read_embedding_dimension(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        header = file.readline().rstrip("\n").split("\t")
    z_columns = [
        column for column in header if column.startswith("z") and column[1:].isdigit()
    ]
    if not z_columns:
        raise SystemExit(f"no latent columns named z0..zN found in {path}")
    return len(z_columns)


def embedding_title_suffix(dimension: int) -> str:
    return f"from {dimension} dimensional embeddings from the spectral autoencoder"


def read_coordinates(path: Path) -> pd.DataFrame:
    coordinates = pd.read_csv(path, sep="\t")
    required = {"index", "x", "y"}
    missing = required.difference(coordinates.columns)
    if missing:
        raise ValueError(f"{path} is missing coordinate columns: {', '.join(sorted(missing))}")
    return coordinates


def available_metrics(metrics: pd.DataFrame) -> list[tuple[str, str, MetricTransform]]:
    specs: list[tuple[str, str, MetricTransform]] = [
        ("num_peaks", "number of peaks", column("num_peaks")),
        ("mean_mz", "mean fragment m/z", column("mean_mz")),
        ("median_mz", "median fragment m/z", column("median_mz")),
        ("pepmass", "PEPMASS", column("pepmass")),
        ("mz_range", "fragment m/z range", column("mz_range")),
        (
            "intensity_weighted_mean_mz",
            "intensity-weighted mean m/z",
            column("intensity_weighted_mean_mz"),
        ),
        ("tic_log10", "log10 TIC", log10_column("tic")),
        (
            "base_peak_intensity_log10",
            "log10 base peak intensity",
            log10_column("base_peak_intensity"),
        ),
        ("base_peak_fraction", "base peak fraction", column("base_peak_fraction")),
        ("intensity_entropy", "intensity entropy", column("intensity_entropy")),
        (
            "normalized_intensity_entropy",
            "normalized intensity entropy",
            column("normalized_intensity_entropy"),
        ),
        (
            "fragment_to_precursor_mean_ratio",
            "mean fragment m/z over PEPMASS",
            column("fragment_to_precursor_mean_ratio"),
        ),
        (
            "fraction_peaks_below_precursor",
            "fraction of peaks below precursor",
            column("fraction_peaks_below_precursor"),
        ),
        ("charge_abs", "absolute charge", column("charge_abs")),
    ]
    return [spec for spec in specs if required_column(spec[2]) in metrics.columns]


def column(name: str) -> MetricTransform:
    def transform(df: pd.DataFrame) -> np.ndarray:
        return df[name].to_numpy(dtype=np.float64)

    transform.required_column = name  # type: ignore[attr-defined]
    return transform


def log10_column(name: str) -> MetricTransform:
    def transform(df: pd.DataFrame) -> np.ndarray:
        values = df[name].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(values)

    transform.required_column = name  # type: ignore[attr-defined]
    return transform


def required_column(transform: MetricTransform) -> str:
    return getattr(transform, "required_column")


def write_density_heatmap(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    bins: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    ax.set_facecolor("#f4f6f8")
    collection = ax.hexbin(
        x,
        y,
        gridsize=bins,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    colorbar = fig.colorbar(collection, ax=ax)
    colorbar.set_label("point count, log scale")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_metric_heatmap(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    bins: int,
    min_count: int,
    dpi: int,
) -> None:
    finite = np.isfinite(values)
    x = x[finite]
    y = y[finite]
    values = values[finite]
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    ax.set_facecolor("#f4f6f8")
    collection = ax.hexbin(
        x,
        y,
        C=values,
        reduce_C_function=np.nanmean,
        gridsize=bins,
        mincnt=min_count,
        cmap="viridis",
        linewidths=0,
    )
    finite_values = collection_values(collection)
    if finite_values.size == 0:
        raise ValueError(f"no finite heatmap bins for {path}")
    vmin, vmax = np.nanpercentile(finite_values, [1, 99])
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
        collection.set_clim(vmin, vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    colorbar = fig.colorbar(collection, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def collection_values(collection: Any) -> np.ndarray:
    values = collection.get_array()
    if np.ma.isMaskedArray(values):
        array = values.compressed()
    else:
        array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


if __name__ == "__main__":
    main()
