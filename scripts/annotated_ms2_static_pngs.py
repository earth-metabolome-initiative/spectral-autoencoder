#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///

from __future__ import annotations

import argparse
import base64
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NPC_COLUMN_INDEX = 5
CLASSYFIRE_CLASS_INDEX = 9


def main() -> None:
    args = parse_args()
    embeddings = args.embeddings or args.plot_dir.parent / "embeddings.tsv"
    title_suffix = embedding_title_suffix(read_embedding_dimension(embeddings))
    write_pngs(args.plot_dir, args.output_dir, args.classyfire_top, args.dpi, title_suffix)


def write_pngs(
    plot_dir: Path,
    output_dir: Path,
    classyfire_top: int,
    dpi: int,
    title_suffix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in ["tsne", "tmap"]:
        html = plot_dir / f"{method}_npc_pathway.html"
        if not html.exists():
            continue
        x, y, metadata = extract_scatter_points(html)
        write_npc_plot(
            output_dir / f"{method}_npc_pathway_matplotlib.png",
            x,
            y,
            metadata,
            f"{method.upper()} colored by NPC pathway",
            title_suffix,
            dpi,
        )
        write_classyfire_plot(
            output_dir / f"{method}_classyfire_class_matplotlib.png",
            x,
            y,
            metadata,
            f"{method.upper()} colored by ClassyFire class",
            title_suffix,
            classyfire_top,
            dpi,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("runs/annotated-ms2-flat/plots"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/annotated-ms2-flat/plots/matplotlib_pngs"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Embedding TSV used to infer the latent dimension for image titles.",
    )
    parser.add_argument("--classyfire-top", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=220)
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


def extract_scatter_points(path: Path) -> tuple[np.ndarray, np.ndarray, list[list[Any]]]:
    data = read_plotly_data(path)
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

    return np.concatenate(xs), np.concatenate(ys), metadata


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


def write_npc_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    metadata: list[list[Any]],
    title: str,
    title_suffix: str,
    dpi: int,
) -> None:
    labels_per_point = [
        split_pipe_labels(row[NPC_COLUMN_INDEX] if len(row) > NPC_COLUMN_INDEX else None)
        for row in metadata
    ]
    labels = sorted({label for labels in labels_per_point for label in labels})
    expanded_x: list[np.ndarray] = []
    expanded_y: list[np.ndarray] = []
    expanded_labels: list[str] = []
    for label in labels:
        mask = np.fromiter((label in item for item in labels_per_point), dtype=bool, count=len(x))
        if mask.any():
            expanded_x.append(x[mask])
            expanded_y.append(y[mask])
            expanded_labels.append(label)
    draw_scatter(path, expanded_x, expanded_y, expanded_labels, title, title_suffix, dpi)


def write_classyfire_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    metadata: list[list[Any]],
    title: str,
    title_suffix: str,
    top_n: int,
    dpi: int,
) -> None:
    raw_labels = [
        clean_label(row[CLASSYFIRE_CLASS_INDEX] if len(row) > CLASSYFIRE_CLASS_INDEX else None)
        for row in metadata
    ]
    counts = Counter(label for label in raw_labels if label != "Unannotated")
    top_labels = [label for label, _ in counts.most_common(top_n)]

    arrays_x: list[np.ndarray] = []
    arrays_y: list[np.ndarray] = []
    shown_labels: list[str] = []
    raw_labels_array = np.asarray(raw_labels, dtype=object)
    for label in top_labels:
        mask = raw_labels_array == label
        if mask.any():
            arrays_x.append(x[mask])
            arrays_y.append(y[mask])
            shown_labels.append(label)
    draw_scatter(path, arrays_x, arrays_y, shown_labels, title, title_suffix, dpi)


def split_pipe_labels(value: Any) -> list[str]:
    label = clean_label(value)
    if label == "Unannotated":
        return [label]
    parts = [part.strip() for part in label.split("|") if part.strip()]
    return parts or ["Unannotated"]


def clean_label(value: Any) -> str:
    if value is None:
        return "Unannotated"
    label = str(value).strip()
    if not label or label.lower() in {"nan", "none", "null"}:
        return "Unannotated"
    return label


def draw_scatter(
    path: Path,
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    labels: list[str],
    title: str,
    title_suffix: str,
    dpi: int,
) -> None:
    colors = color_cycle(len(labels))
    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    ax.set_facecolor("#f4f6f8")
    for x, y, label, color in zip(xs, ys, labels, colors, strict=True):
        ax.scatter(
            x,
            y,
            s=0.55,
            c=[color],
            alpha=0.65,
            linewidths=0,
            rasterized=True,
            label=f"{label} ({len(x):,})",
        )
    ax.set_title(f"{title}\n{title_suffix}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(color="white", linewidth=0.8)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        markerscale=6,
        fontsize=7,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def color_cycle(count: int) -> list[Any]:
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors: list[Any] = []
    for cmap_name in cmap_names:
        cmap = plt.get_cmap(cmap_name)
        colors.extend(cmap(i) for i in range(cmap.N))
    if count <= len(colors):
        return colors[:count]
    extra = plt.get_cmap("hsv")
    colors.extend(extra(i / max(1, count - len(colors))) for i in range(count - len(colors)))
    return colors[:count]


if __name__ == "__main__":
    main()
