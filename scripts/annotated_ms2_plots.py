#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "plotly>=5.22",
#   "scikit-learn>=1.4",
#   "tmap2==0.2.0",
# ]
# ///

from __future__ import annotations

if __name__ == "__main__":
    from annotated_ms2_png_plots import main as png_main

    raise SystemExit(png_main())

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from MulticoreTSNE import MulticoreTSNE as TSNE

from annotated_ms2_static_pngs import write_pngs

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
    title_suffix = embedding_title_suffix(len(z_columns))

    embeddings = df[z_columns].to_numpy(dtype=np.float32, copy=True)
    embeddings = normalize_embeddings(
        StandardScaler().fit_transform(embeddings).astype(np.float32),
        args.normalization,
    )

    npc_labels, npc_counts = top_labels(df, args.npc_column, args.top_labels)
    classy_labels, classy_counts = top_labels(df, args.classyfire_column, args.top_labels)
    write_counts(args.output_dir / "npc_pathway_counts.tsv", npc_counts)
    write_counts(args.output_dir / f"{args.classyfire_column}_counts.tsv", classy_counts)
    write_count_plot(
        args.output_dir / "npc_pathway_counts.html",
        npc_counts,
        "NPC pathway",
        title_suffix,
    )
    write_count_png(
        args.output_dir / "npc_pathway_counts.png",
        npc_counts,
        "NPC pathway",
        title_suffix,
    )
    write_count_plot(
        args.output_dir / f"{args.classyfire_column}_counts.html",
        classy_counts,
        args.classyfire_column.replace("_", " "),
        title_suffix,
    )
    write_count_png(
        args.output_dir / f"{args.classyfire_column}_counts.png",
        classy_counts,
        args.classyfire_column.replace("_", " "),
        title_suffix,
    )

    hover_columns = [
        column
        for column in [
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
        if column in df.columns
    ]

    if not args.skip_tsne:
        tsne_xy = compute_tsne(embeddings, args)
        write_scatter(
            args.output_dir / "tsne_npc_pathway.html",
            df,
            tsne_xy,
            npc_labels,
            hover_columns,
            "t-SNE colored by NPC pathway",
            title_suffix,
        )
        write_scatter(
            args.output_dir / f"tsne_{args.classyfire_column}.html",
            df,
            tsne_xy,
            classy_labels,
            hover_columns,
            f"t-SNE colored by {args.classyfire_column.replace('_', ' ')}",
            title_suffix,
            drop_labels=EXCLUDED_CLASSYFIRE_LABELS,
        )

    if not args.skip_tmap:
        tmap_xy, edges = compute_tmap(embeddings, args)
        write_tmap_plot(
            args.output_dir / "tmap_npc_pathway.html",
            df,
            tmap_xy,
            edges,
            npc_labels,
            hover_columns,
            "TMAP colored by NPC pathway",
            title_suffix,
        )
        write_tmap_plot(
            args.output_dir / f"tmap_{args.classyfire_column}.html",
            df,
            tmap_xy,
            edges,
            classy_labels,
            hover_columns,
            f"TMAP colored by {args.classyfire_column.replace('_', ' ')}",
            title_suffix,
            drop_labels=EXCLUDED_CLASSYFIRE_LABELS,
        )

    if not args.skip_png:
        write_pngs(
            args.output_dir,
            args.output_dir / "matplotlib_pngs",
            args.classyfire_png_top,
            args.png_dpi,
            title_suffix,
        )

    print(f"wrote plots under {args.output_dir}")


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
    parser.add_argument("--skip-png", action="store_true")
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


def top_labels(df: pd.DataFrame, column: str, top_n: int) -> tuple[pd.Series, pd.Series]:
    if column not in df.columns:
        labels = pd.Series(["Unannotated"] * len(df), index=df.index, name=column)
        return labels, labels.value_counts()

    labels = df[column].fillna("").astype(str).str.strip()
    labels = labels.mask(labels.eq("") | labels.str.lower().isin({"nan", "none", "null"}))
    counts = labels.dropna().value_counts()
    top_values = set(counts.head(top_n).index)
    grouped = labels.where(labels.isin(top_values), "Other")
    grouped = grouped.fillna("Unannotated")
    return grouped, counts


def write_counts(path: Path, counts: pd.Series) -> None:
    counts.rename_axis("label").reset_index(name="count").to_csv(path, sep="\t", index=False)


def write_count_plot(path: Path, counts: pd.Series, title: str, title_suffix: str) -> None:
    shown = counts.head(40).sort_values()
    plot_df = shown.rename_axis("label").reset_index(name="count")
    fig = px.bar(
        plot_df,
        x="count",
        y="label",
        orientation="h",
        title=f"Most common {title} annotations<br>{title_suffix}",
        labels={"label": title, "count": "Count"},
    )
    fig.write_html(path)


def write_count_png(path: Path, counts: pd.Series, title: str, title_suffix: str) -> None:
    shown = counts.head(40).sort_values()
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    ax.barh(shown.index.astype(str), shown.to_numpy())
    ax.set_title(f"Most common {title} annotations\n{title_suffix}")
    ax.set_xlabel("Count")
    ax.set_ylabel(title)
    ax.grid(axis="x", color="#d8dde3", linewidth=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_tsne(embeddings: np.ndarray, args: argparse.Namespace) -> np.ndarray:
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


def write_scatter(
    path: Path,
    df: pd.DataFrame,
    xy: np.ndarray,
    labels: pd.Series,
    hover_columns: list[str],
    title: str,
    title_suffix: str,
    drop_labels: set[str] | None = None,
) -> None:
    plot_df = df.loc[:, hover_columns].copy()
    plot_df["x"] = xy[:, 0]
    plot_df["y"] = xy[:, 1]
    plot_df["label"] = labels.to_numpy()
    if drop_labels:
        plot_df = plot_df.loc[~plot_df["label"].isin(drop_labels)]
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="label",
        hover_data=hover_columns,
        render_mode="webgl",
        title=f"{title}<br>{title_suffix}",
    )
    fig.update_traces(marker={"size": 4, "opacity": 0.75})
    fig.write_html(path)


def write_tmap_plot(
    path: Path,
    df: pd.DataFrame,
    xy: np.ndarray,
    edges: tuple[list[int], list[int]],
    labels: pd.Series,
    hover_columns: list[str],
    title: str,
    title_suffix: str,
    drop_labels: set[str] | None = None,
) -> None:
    source, target = edges
    label_values = labels.to_numpy()
    kept = np.ones(len(label_values), dtype=bool)
    if drop_labels:
        kept = ~pd.Series(label_values).isin(drop_labels).to_numpy()

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for left, right in zip(source, target, strict=False):
        if not kept[left] or not kept[right]:
            continue
        edge_x.extend([xy[left, 0], xy[right, 0], None])
        edge_y.extend([xy[left, 1], xy[right, 1], None])

    plot_df = df.loc[:, hover_columns].copy()
    plot_df["x"] = xy[:, 0]
    plot_df["y"] = xy[:, 1]
    plot_df["label"] = label_values
    if drop_labels:
        plot_df = plot_df.loc[kept]
    scatter = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="label",
        hover_data=hover_columns,
        render_mode="webgl",
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 0.4, "color": "rgba(70,70,70,0.25)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for trace in scatter.data:
        trace.marker.size = 4
        trace.marker.opacity = 0.75
        fig.add_trace(trace)
    fig.update_layout(title=f"{title}<br>{title_suffix}", xaxis_visible=False, yaxis_visible=False)
    fig.write_html(path)


if __name__ == "__main__":
    main()
