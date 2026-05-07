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
import base64
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HOVER_COLUMNS = [
    "feature_id",
    "spectrum_id",
    "name",
    "smiles",
    "inchikey",
    "npc_pathway",
    "npc_superclass",
    "npc_class",
    "classyfire_superclass",
    "classyfire_class",
    "classyfire_subclass",
]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading TMAP coordinates: {args.tmap_coordinates}")
    coordinates = read_coordinates(args.tmap_coordinates).sort_values("index")
    row_xy = coordinates.loc[:, ["x", "y"]].to_numpy(dtype=np.float64)

    pc1_index = first_principal_component(row_xy)
    try:
        print("reconstructing TMAP tree")
        edges = read_edge_rows(args.tmap_edges, row_xy)
        tmap_index = tree_diameter_index(row_xy, edges)
        index_source = "tree_diameter"
    except ValueError as error:
        print(f"warning: {error}; using TMAP coordinate PC1 as the scalar index")
        tmap_index = pc1_index
        index_source = "coordinate_pc1"

    print(f"reading metrics: {args.metrics}")
    metrics = pd.read_csv(args.metrics, sep="\t", low_memory=False).sort_values("index")
    if not np.array_equal(metrics["index"].to_numpy(dtype=np.int64), coordinates["index"].to_numpy(dtype=np.int64)):
        raise ValueError("metrics and TMAP coordinate indexes do not match")
    if len(metrics) != len(tmap_index):
        raise ValueError(
            f"metrics rows ({len(metrics)}) do not match TMAP rows ({len(tmap_index)})"
        )

    correlations = metric_correlations(metrics, tmap_index, index_source, pc1_index, row_xy)
    correlations.to_csv(args.output, sep="\t", index=False)
    print(f"wrote correlations: {args.output}")
    print(
        correlations.loc[:, ["metric", "rho_tmap_index", "abs_rho_tmap_index", "n"]]
        .head(args.print_top)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tmap-coordinates",
        type=Path,
        default=Path("runs/annotated-ms2-flat/plots/tmap_coordinates.tsv"),
    )
    parser.add_argument(
        "--tmap-edges",
        type=Path,
        default=Path("runs/annotated-ms2-flat/plots/tmap_edges.tsv"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("runs/annotated-ms2-flat/embeddings.tsv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/annotated-ms2-flat/spectrum_metrics.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/annotated-ms2-flat/plots/tmap_metric_spearman.tsv"),
    )
    parser.add_argument("--print-top", type=int, default=20)
    return parser.parse_args()


def read_coordinates(path: Path) -> pd.DataFrame:
    coordinates = pd.read_csv(path, sep="\t")
    required = {"index", "x", "y"}
    missing = required.difference(coordinates.columns)
    if missing:
        raise ValueError(f"{path} is missing coordinate columns: {', '.join(sorted(missing))}")
    return coordinates


def read_edge_rows(path: Path, row_xy: np.ndarray) -> list[tuple[int, int, float]]:
    edges = pd.read_csv(path, sep="\t")
    required = {"source", "target"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"{path} is missing edge columns: {', '.join(sorted(missing))}")

    rows: list[tuple[int, int, float]] = []
    for source, target in zip(edges["source"], edges["target"], strict=True):
        left = int(source)
        right = int(target)
        dx = row_xy[left, 0] - row_xy[right, 0]
        dy = row_xy[left, 1] - row_xy[right, 1]
        rows.append((left, right, float(np.hypot(dx, dy))))
    return rows


def extract_tmap(path: Path) -> tuple[np.ndarray, np.ndarray, list[list[Any]], list[Any], list[Any]]:
    data = read_plotly_data(path)
    edge_trace = next(
        trace for trace in data if trace.get("mode") == "lines" and "customdata" not in trace
    )
    edge_x = edge_trace["x"]
    edge_y = edge_trace["y"]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metadata: list[list[Any]] = []
    for trace in data:
        customdata = trace.get("customdata")
        if not customdata:
            continue
        x = decode_plotly_array(trace["x"])
        y = decode_plotly_array(trace["y"])
        if len(x) != len(customdata) or len(y) != len(customdata):
            raise ValueError(f"trace length mismatch in {path}")
        xs.append(x)
        ys.append(y)
        metadata.extend(customdata)

    if not xs:
        raise ValueError(f"no scatter traces with customdata found in {path}")
    return np.concatenate(xs), np.concatenate(ys), metadata, edge_x, edge_y


def read_plotly_data(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    start = text.rfind("Plotly.newPlot(")
    if start < 0:
        raise ValueError(f"Plotly.newPlot call not found in {path}")

    args = parse_js_call_args(text, start + len("Plotly.newPlot("), expected=2)
    return json.loads(args[1])


def parse_js_call_args(text: str, pos: int, expected: int) -> list[str]:
    args: list[str] = []
    while len(args) < expected:
        while text[pos].isspace():
            pos += 1
        start = pos
        opener = text[pos]
        if opener in "[{":
            closer = "]" if opener == "[" else "}"
            depth = 0
            in_string = False
            escaped = False
            while pos < len(text):
                char = text[pos]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char == opener:
                        depth += 1
                    elif char == closer:
                        depth -= 1
                        if depth == 0:
                            pos += 1
                            break
                pos += 1
        elif opener == '"':
            pos += 1
            escaped = False
            while pos < len(text):
                char = text[pos]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    pos += 1
                    break
                pos += 1
        else:
            while pos < len(text) and text[pos] not in ",)":
                pos += 1
        args.append(text[start:pos])
        while pos < len(text) and text[pos] not in ",)":
            pos += 1
        if pos < len(text) and text[pos] == ",":
            pos += 1
    return args


def decode_plotly_array(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, dict) and "bdata" in value:
        dtype = plotly_dtype(value["dtype"])
        raw = base64.b64decode(value["bdata"])
        return np.frombuffer(raw, dtype=dtype).astype(np.float64, copy=False)
    raise TypeError(f"unsupported Plotly array encoding: {type(value).__name__}")


def plotly_dtype(dtype: str) -> np.dtype:
    mapping = {
        "f8": np.dtype("<f8"),
        "f4": np.dtype("<f4"),
        "i4": np.dtype("<i4"),
        "u4": np.dtype("<u4"),
        "i8": np.dtype("<i8"),
        "u8": np.dtype("<u8"),
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported Plotly dtype {dtype!r}") from error


def read_embedding_metadata(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, sep="\t", nrows=0).columns
    usecols = [column for column in ["index", *HOVER_COLUMNS] if column in columns]
    return pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)


def align_points_to_embedding_rows(
    point_metadata: list[list[Any]],
    embeddings: pd.DataFrame,
) -> np.ndarray:
    width = len(point_metadata[0])
    key_columns = [column for column in HOVER_COLUMNS if column in embeddings.columns][:width]
    if len(key_columns) != width:
        raise ValueError(
            f"Plotly customdata has {width} fields but only {len(key_columns)} metadata "
            "columns are available in the embeddings TSV"
        )

    buckets: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    key_arrays = [embeddings[column].to_numpy() for column in key_columns]
    for row_position in range(len(embeddings)):
        key = tuple(normalize_key_value(array[row_position]) for array in key_arrays)
        buckets[key].append(row_position)

    row_indices = np.empty(len(point_metadata), dtype=np.int64)
    for point_index, row in enumerate(point_metadata):
        key = tuple(normalize_key_value(value) for value in row[:width])
        bucket = buckets.get(key)
        if not bucket:
            raise ValueError(f"could not align point metadata key: {key}")
        row_indices[point_index] = bucket.popleft()
    return row_indices


def normalize_key_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    if isinstance(value, (np.floating,)):
        return normalize_key_value(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def edge_rows(edge_x: list[Any], edge_y: list[Any], row_xy: np.ndarray) -> list[tuple[int, int, float]]:
    coord_to_row: dict[tuple[str, str], int] = {}
    for row_index, (x, y) in enumerate(row_xy):
        key = coord_key(x, y)
        if key in coord_to_row:
            raise ValueError(f"duplicate TMAP coordinate: {key}")
        coord_to_row[key] = row_index

    if len(edge_x) != len(edge_y) or len(edge_x) % 3 != 0:
        raise ValueError("TMAP edge trace is not encoded as x1,x2,None triplets")

    edges: list[tuple[int, int, float]] = []
    for offset in range(0, len(edge_x), 3):
        left = coord_to_row[coord_key(edge_x[offset], edge_y[offset])]
        right = coord_to_row[coord_key(edge_x[offset + 1], edge_y[offset + 1])]
        dx = row_xy[left, 0] - row_xy[right, 0]
        dy = row_xy[left, 1] - row_xy[right, 1]
        edges.append((left, right, float(np.hypot(dx, dy))))
    return edges


def coord_key(x: Any, y: Any) -> tuple[str, str]:
    return (float(x).hex(), float(y).hex())


def tree_diameter_index(row_xy: np.ndarray, edges: list[tuple[int, int, float]]) -> np.ndarray:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(row_xy))]
    for left, right, weight in edges:
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))

    endpoint, _ = farthest_tree_node(0, adjacency)
    _, distances = farthest_tree_node(endpoint, adjacency)
    if not np.isfinite(distances).all():
        raise ValueError("TMAP tree is not connected")
    max_distance = distances.max()
    if max_distance <= 0.0:
        raise ValueError("TMAP tree diameter is zero")
    return distances / max_distance


def farthest_tree_node(
    start: int, adjacency: list[list[tuple[int, float]]]
) -> tuple[int, np.ndarray]:
    distances = np.full(len(adjacency), np.nan, dtype=np.float64)
    distances[start] = 0.0
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor, weight in adjacency[node]:
            if np.isnan(distances[neighbor]):
                distances[neighbor] = distances[node] + weight
                stack.append(neighbor)
    return int(np.nanargmax(distances)), distances


def first_principal_component(row_xy: np.ndarray) -> np.ndarray:
    centered = row_xy - row_xy.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    values = centered @ right[0]
    values -= values.min()
    max_value = values.max()
    return values / max_value if max_value > 0 else values


def metric_correlations(
    metrics: pd.DataFrame,
    tmap_index: np.ndarray,
    index_source: str,
    pc1_index: np.ndarray,
    row_xy: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in metrics.columns:
        if column == "index":
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 3 or np.unique(values[finite]).size < 2:
            continue
        tmap = spearmanr(tmap_index[finite], values[finite])
        pc1 = spearmanr(pc1_index[finite], values[finite])
        x_corr = spearmanr(row_xy[finite, 0], values[finite])
        y_corr = spearmanr(row_xy[finite, 1], values[finite])
        rows.append(
            {
                "metric": column,
                "n": int(finite.sum()),
                "tmap_index_source": index_source,
                "rho_tmap_index": float(tmap.statistic),
                "abs_rho_tmap_index": abs(float(tmap.statistic)),
                "p_tmap_index": float(tmap.pvalue),
                "rho_tmap_pc1": float(pc1.statistic),
                "abs_rho_tmap_pc1": abs(float(pc1.statistic)),
                "rho_tmap_x": float(x_corr.statistic),
                "rho_tmap_y": float(y_corr.statistic),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_rho_tmap_index", ascending=False)


if __name__ == "__main__":
    main()
