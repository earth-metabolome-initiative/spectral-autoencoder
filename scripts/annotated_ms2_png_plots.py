#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "MulticoreTSNE==0.1",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.4",
#   "tmap2==0.2.0",
# ]
# ///

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

EXCLUDED_CLASSYFIRE_LABELS = {"Other", "Unannotated"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.embeddings, sep="\t", low_memory=False)
    z_columns = sorted(
        [column for column in df.columns if column.startswith("z") and column[1:].isdigit()],
        key=lambda column: int(column[1:]),
    )
    if not z_columns:
        raise SystemExit(f"no latent columns named z0..zN found in {args.embeddings}")
    title_suffix = args.title_suffix or embedding_title_suffix(len(z_columns))

    embeddings = df[z_columns].to_numpy(dtype=np.float32, copy=True)
    embeddings = normalize_embeddings(
        StandardScaler().fit_transform(embeddings).astype(np.float32),
        args.normalization,
    )

    npc_labels = split_label_series(df, args.npc_column)
    classyfire_labels = clean_label_series(df, args.classyfire_column)
    npc_counts = Counter(label for labels in npc_labels for label in labels)
    classyfire_counts = Counter(
        label for label in classyfire_labels if label not in EXCLUDED_CLASSYFIRE_LABELS
    )
    write_counts(args.output_dir / "npc_pathway_counts.tsv", npc_counts)
    write_counts(args.output_dir / f"{args.classyfire_column}_counts.tsv", classyfire_counts)
    write_count_png(
        args.output_dir / "npc_pathway_counts.png",
        npc_counts,
        "NPC pathway",
        title_suffix,
    )
    write_count_png(
        args.output_dir / f"{args.classyfire_column}_counts.png",
        classyfire_counts,
        args.classyfire_column.replace("_", " "),
        title_suffix,
    )

    if not args.skip_tsne:
        tsne_coordinates = args.output_dir / "tsne_coordinates.tsv"
        if args.reuse_coordinates and tsne_coordinates.exists():
            tsne_xy = read_coordinates(tsne_coordinates)
        else:
            tsne_xy = compute_tsne(embeddings, args)
            write_coordinates(tsne_coordinates, tsne_xy)
        write_multilabel_scatter_png(
            args.output_dir / "tsne_npc_pathway.png",
            tsne_xy,
            npc_labels,
            "t-SNE colored by NPC pathway",
            title_suffix,
            args.top_labels,
            args.png_dpi,
        )
        write_single_label_scatter_png(
            args.output_dir / f"tsne_{args.classyfire_column}.png",
            tsne_xy,
            classyfire_labels,
            f"t-SNE colored by {args.classyfire_column.replace('_', ' ')}",
            title_suffix,
            args.classyfire_png_top,
            args.png_dpi,
        )

    if not args.skip_tmap:
        tmap_coordinates = args.output_dir / "tmap_coordinates.tsv"
        if args.reuse_coordinates and tmap_coordinates.exists():
            tmap_xy = read_coordinates(tmap_coordinates)
        else:
            tmap_xy, edges = compute_tmap(embeddings, args)
            write_coordinates(tmap_coordinates, tmap_xy)
            write_edges(args.output_dir / "tmap_edges.tsv", edges)
        write_multilabel_scatter_png(
            args.output_dir / "tmap_npc_pathway.png",
            tmap_xy,
            npc_labels,
            "TMAP colored by NPC pathway",
            title_suffix,
            args.top_labels,
            args.png_dpi,
        )
        write_single_label_scatter_png(
            args.output_dir / f"tmap_{args.classyfire_column}.png",
            tmap_xy,
            classyfire_labels,
            f"TMAP colored by {args.classyfire_column.replace('_', ' ')}",
            title_suffix,
            args.classyfire_png_top,
            args.png_dpi,
        )

    print(f"wrote PNG plots and coordinates under {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("runs/annotated-ms2-flat/embeddings.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/annotated-ms2-flat/plots"),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-labels", type=int, default=25)
    parser.add_argument("--npc-column", default="npc_pathway")
    parser.add_argument("--classyfire-column", default="classyfire_class")
    parser.add_argument("--tsne-perplexity", type=float, default=50.0)
    parser.add_argument(
        "--tsne-pca-dim",
        type=int,
        default=50,
        help="Reduce embeddings to at most this many PCA dimensions before t-SNE; 0 disables PCA.",
    )
    parser.add_argument("--tsne-iterations", type=int, default=1_000)
    parser.add_argument("--tsne-learning-rate", type=float, default=None)
    parser.add_argument("--tsne-jobs", type=int, default=-1)
    parser.add_argument("--tmap-neighbors", type=int, default=20)
    parser.add_argument("--tmap-layout-iterations", type=int, default=1_000)
    parser.add_argument("--tmap-metric", default="cosine")
    parser.add_argument(
        "--normalization",
        choices=["standard", "standard-l2", "standard-l1"],
        default="standard",
    )
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-tmap", action="store_true")
    parser.add_argument(
        "--reuse-coordinates",
        action="store_true",
        help="Reuse existing tsne/tmap coordinate TSVs instead of recomputing layouts.",
    )
    parser.add_argument(
        "--title-suffix",
        default=None,
        help="Optional second title line. Defaults to the inferred embedding dimension.",
    )
    parser.add_argument("--png-dpi", type=int, default=220)
    parser.add_argument("--classyfire-png-top", type=int, default=8)
    return parser.parse_args()


def normalize_embeddings(embeddings: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "standard":
        return embeddings
    if normalization == "standard-l2":
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    elif normalization == "standard-l1":
        norms = np.linalg.norm(embeddings, ord=1, axis=1, keepdims=True)
    else:
        raise ValueError(f"unsupported normalization {normalization!r}")
    return embeddings / np.clip(norms, 1e-12, None)


def embedding_title_suffix(dimension: int) -> str:
    return f"from {dimension} dimensional embeddings from the spectral autoencoder"


def split_label_series(df: pd.DataFrame, column: str) -> list[list[str]]:
    if column not in df.columns:
        return [["Unannotated"] for _ in range(len(df))]
    return [split_pipe_labels(value) for value in df[column].to_numpy()]


def clean_label_series(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return ["Unannotated"] * len(df)
    return [clean_label(value) for value in df[column].to_numpy()]


def split_pipe_labels(value: object) -> list[str]:
    label = clean_label(value)
    if label == "Unannotated":
        return [label]
    parts = [part.strip() for part in label.split("|") if part.strip()]
    return parts or ["Unannotated"]


def clean_label(value: object) -> str:
    if value is None:
        return "Unannotated"
    label = str(value).strip()
    if not label or label.lower() in {"nan", "none", "null"}:
        return "Unannotated"
    return label


def write_counts(path: Path, counts: Counter[str]) -> None:
    rows = counts.most_common()
    pd.DataFrame(rows, columns=["label", "count"]).to_csv(path, sep="\t", index=False)


def write_count_png(path: Path, counts: Counter[str], title: str, title_suffix: str) -> None:
    shown = list(reversed(counts.most_common(40)))
    if not shown:
        return
    labels, values = zip(*shown, strict=True)
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    ax.barh(labels, values)
    ax.set_title(f"Most common {title} annotations\n{title_suffix}")
    ax.set_xlabel("Count")
    ax.set_ylabel(title)
    ax.grid(axis="x", color="#d8dde3", linewidth=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_tsne(embeddings: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    embeddings = pca_for_tsne(embeddings, args.tsne_pca_dim, args.seed)
    perplexity = min(args.tsne_perplexity, max(5.0, (len(embeddings) - 1) / 3.0))
    learning_rate = args.tsne_learning_rate
    if learning_rate is None:
        learning_rate = max(len(embeddings) / 12.0 / 4.0, 50.0)
    kwargs = {
        "n_components": 2,
        "init": "random",
        "learning_rate": learning_rate,
        "perplexity": perplexity,
        "random_state": args.seed,
        "verbose": 1,
        "n_jobs": args.tsne_jobs,
        "n_iter": args.tsne_iterations,
    }
    return TSNE(**kwargs).fit_transform(embeddings)


def pca_for_tsne(embeddings: np.ndarray, requested_dim: int, seed: int) -> np.ndarray:
    if requested_dim <= 0:
        return embeddings
    n_samples, n_features = embeddings.shape
    output_dim = min(requested_dim, n_features, n_samples - 1)
    if output_dim >= n_features:
        return embeddings
    print(f"running PCA before t-SNE: {n_features} -> {output_dim} dimensions")
    return (
        PCA(
            n_components=output_dim,
            svd_solver="randomized",
            random_state=seed,
        )
        .fit_transform(embeddings)
        .astype(np.float32, copy=False)
    )


def compute_tmap(
    embeddings: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    import tmap as tm

    model = tm.TMAP(
        n_neighbors=args.tmap_neighbors,
        metric=args.tmap_metric,
        seed=args.seed,
        layout_iterations=args.tmap_layout_iterations,
    )
    x, y, source, target = model.fit_transform(embeddings)
    return np.column_stack([np.asarray(x), np.asarray(y)]), (list(source), list(target))


def write_coordinates(path: Path, xy: np.ndarray) -> None:
    pd.DataFrame(
        {
            "index": np.arange(len(xy), dtype=np.int64),
            "x": xy[:, 0],
            "y": xy[:, 1],
        }
    ).to_csv(path, sep="\t", index=False)


def read_coordinates(path: Path) -> np.ndarray:
    coordinates = pd.read_csv(path, sep="\t")
    required = {"x", "y"}
    missing = required.difference(coordinates.columns)
    if missing:
        raise ValueError(f"{path} is missing coordinate columns: {', '.join(sorted(missing))}")
    return coordinates.loc[:, ["x", "y"]].to_numpy(dtype=np.float64)


def write_edges(path: Path, edges: tuple[list[int], list[int]]) -> None:
    source, target = edges
    pd.DataFrame({"source": source, "target": target}).to_csv(path, sep="\t", index=False)


def write_multilabel_scatter_png(
    path: Path,
    xy: np.ndarray,
    labels_per_point: list[list[str]],
    title: str,
    title_suffix: str,
    top_n: int,
    dpi: int,
) -> None:
    counts = Counter(label for labels in labels_per_point for label in labels)
    labels = [label for label, _ in counts.most_common(top_n)]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    shown_labels: list[str] = []
    for label in labels:
        mask = np.fromiter(
            (label in labels for labels in labels_per_point),
            dtype=bool,
            count=len(labels_per_point),
        )
        if mask.any():
            xs.append(xy[mask, 0])
            ys.append(xy[mask, 1])
            shown_labels.append(label)
    draw_scatter(path, xs, ys, shown_labels, title, title_suffix, dpi)


def write_single_label_scatter_png(
    path: Path,
    xy: np.ndarray,
    labels: list[str],
    title: str,
    title_suffix: str,
    top_n: int,
    dpi: int,
) -> None:
    counts = Counter(label for label in labels if label not in EXCLUDED_CLASSYFIRE_LABELS)
    top_labels = [label for label, _ in counts.most_common(top_n)]
    label_array = np.asarray(labels, dtype=object)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    shown_labels: list[str] = []
    for label in top_labels:
        mask = label_array == label
        if mask.any():
            xs.append(xy[mask, 0])
            ys.append(xy[mask, 1])
            shown_labels.append(label)
    draw_scatter(path, xs, ys, shown_labels, title, title_suffix, dpi)


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


def color_cycle(count: int) -> list[object]:
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors: list[object] = []
    for cmap_name in cmap_names:
        cmap = plt.get_cmap(cmap_name)
        colors.extend(cmap(index) for index in range(cmap.N))
    if count <= len(colors):
        return colors[:count]
    extra = plt.get_cmap("hsv")
    colors.extend(
        extra(index / max(1, count - len(colors))) for index in range(count - len(colors))
    )
    return colors[:count]


if __name__ == "__main__":
    main()
