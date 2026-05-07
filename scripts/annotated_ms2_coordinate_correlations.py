#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.11",
# ]
# ///

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading coordinates: {args.coordinates}")
    coordinates = read_coordinates(args.coordinates).sort_values("index")
    xy = coordinates.loc[:, ["x", "y"]].to_numpy(dtype=np.float64)
    axes = {
        "x": xy[:, 0],
        "y": xy[:, 1],
        "xy": xy[:, 0] * xy[:, 1],
    }

    print(f"reading metrics: {args.metrics}")
    metrics = pd.read_csv(args.metrics, sep="\t", low_memory=False).sort_values("index")
    if not np.array_equal(
        metrics["index"].to_numpy(dtype=np.int64),
        coordinates["index"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("metrics and coordinate indexes do not match")

    correlations = metric_correlations(metrics, axes)
    correlations.to_csv(args.output, sep="\t", index=False)
    print(f"wrote correlations: {args.output}")
    print(
        correlations.loc[:, ["metric", "max_abs_rho", "rho_x", "rho_y", "rho_xy", "n"]]
        .head(args.print_top)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coordinates",
        type=Path,
        required=True,
        help="Coordinate TSV with index, x and y columns.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Spectrum metric TSV with an index column.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-top", type=int, default=20)
    return parser.parse_args()


def read_coordinates(path: Path) -> pd.DataFrame:
    coordinates = pd.read_csv(path, sep="\t")
    required = {"index", "x", "y"}
    missing = required.difference(coordinates.columns)
    if missing:
        raise ValueError(f"{path} is missing coordinate columns: {', '.join(sorted(missing))}")
    return coordinates


def metric_correlations(metrics: pd.DataFrame, axes: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in metrics.columns:
        if column == "index":
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 3 or np.unique(values[finite]).size < 2:
            continue

        row: dict[str, Any] = {"metric": column, "n": int(finite.sum())}
        max_abs_rho = 0.0
        for axis_name, axis_values in axes.items():
            correlation = spearmanr(axis_values[finite], values[finite])
            rho = float(correlation.statistic)
            row[f"rho_{axis_name}"] = rho
            row[f"abs_rho_{axis_name}"] = abs(rho)
            row[f"p_{axis_name}"] = float(correlation.pvalue)
            max_abs_rho = max(max_abs_rho, abs(rho))
        row["max_abs_rho"] = max_abs_rho
        rows.append(row)

    return pd.DataFrame(rows).sort_values("max_abs_rho", ascending=False)


if __name__ == "__main__":
    main()
