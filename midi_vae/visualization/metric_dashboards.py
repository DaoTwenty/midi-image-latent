"""Metric dashboard visualization utilities.

Provides bar charts, radar plots, and heatmaps to compare metrics across
VAEs, channel strategies, and detection methods.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_metric_bar_chart(
    results: dict[str, dict[str, float]],
    metric_key: str,
    title: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (10, 5),
    sort_by_value: bool = True,
    higher_is_better: bool = True,
) -> "matplotlib.figure.Figure":
    """Bar chart comparing a single metric across multiple conditions.

    Args:
        results: Mapping of condition_name -> {metric_key: value, ...}.
            E.g., {'sd_vae_ft_mse': {'note_f1': 0.72}, 'flux1_dev': ...}.
        metric_key: The metric to extract and plot.
        title: Figure title (defaults to metric_key if empty).
        ylabel: Y-axis label (defaults to metric_key if empty).
        figsize: Figure size.
        sort_by_value: Sort bars by value (descending if higher_is_better).
        higher_is_better: If True, colour the best bar green; if False, red.

    Returns:
        Matplotlib Figure with the bar chart.
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = [results[n].get(metric_key, float("nan")) for n in names]

    if sort_by_value:
        order = sorted(
            range(len(values)),
            key=lambda i: values[i] if not np.isnan(values[i]) else -np.inf,
            reverse=higher_is_better,
        )
        names = [names[i] for i in order]
        values = [values[i] for i in order]

    colors = ["steelblue"] * len(names)
    # Highlight best bar
    valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
    if valid:
        best_idx = valid[0][0]
        colors[best_idx] = "green" if higher_is_better else "red"

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title or metric_key)

    # Value labels on bars
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    return fig


def plot_metrics_heatmap(
    results: dict[str, dict[str, float]],
    metric_keys: Optional[list[str]] = None,
    title: str = "Metrics Heatmap",
    figsize: tuple[int, int] = (12, 6),
    normalize: bool = True,
) -> "matplotlib.figure.Figure":
    """Heatmap comparing multiple metrics across multiple conditions.

    Rows are conditions (VAEs, channel strategies, etc.), columns are
    metrics. Optionally normalises each column to [0, 1] for comparison.

    Args:
        results: Mapping of condition_name -> {metric_key: float}.
        metric_keys: Subset of metric keys to plot. Uses all if None.
        title: Figure title.
        figsize: Figure size.
        normalize: If True, normalise each column to [0, 1].

    Returns:
        Matplotlib Figure with the heatmap.
    """
    import matplotlib.pyplot as plt

    conditions = list(results.keys())
    if not conditions:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    if metric_keys is None:
        # Union of all keys
        all_keys: set[str] = set()
        for v in results.values():
            all_keys.update(v.keys())
        metric_keys = sorted(all_keys)

    # Build matrix: (n_conditions, n_metrics)
    matrix = np.full((len(conditions), len(metric_keys)), np.nan)
    for i, cond in enumerate(conditions):
        for j, key in enumerate(metric_keys):
            matrix[i, j] = results[cond].get(key, np.nan)

    if normalize:
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0 and valid.max() > valid.min():
                matrix[:, j] = (col - valid.min()) / (valid.max() - valid.min())

    fig, ax = plt.subplots(figsize=figsize)
    # Use a masked array so NaNs are shown as grey
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("YlGn")
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metric_keys)))
    ax.set_xticklabels(metric_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=8)
    ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_radar_chart(
    results: dict[str, dict[str, float]],
    metric_keys: list[str],
    title: str = "Metric Radar",
    figsize: tuple[int, int] = (7, 7),
    normalize: bool = True,
) -> "matplotlib.figure.Figure":
    """Radar (spider) chart comparing multiple metrics for multiple conditions.

    Each condition is drawn as a filled polygon. Metric values are
    optionally normalised to [0, 1] across all conditions for each axis.

    Args:
        results: Mapping of condition_name -> {metric_key: float}.
        metric_keys: List of metric keys to use as radar axes.
        title: Figure title.
        figsize: Figure size (should be roughly square).
        normalize: Normalise each axis to [0, 1] across conditions.

    Returns:
        Matplotlib Figure with the radar chart.
    """
    import matplotlib.pyplot as plt

    n_metrics = len(metric_keys)
    if n_metrics < 3:
        raise ValueError("Radar chart requires at least 3 metric axes.")

    # Angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Build values matrix
    conditions = list(results.keys())
    raw_values: list[list[float]] = []
    for cond in conditions:
        row = [results[cond].get(k, 0.0) for k in metric_keys]
        raw_values.append(row)

    matrix = np.array(raw_values, dtype=np.float64)

    if normalize:
        for j in range(n_metrics):
            col_max = matrix[:, j].max()
            col_min = matrix[:, j].min()
            if col_max > col_min:
                matrix[:, j] = (matrix[:, j] - col_min) / (col_max - col_min)
            else:
                matrix[:, j] = 0.5

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10", len(conditions))

    for i, (cond, row) in enumerate(zip(conditions, matrix)):
        vals = row.tolist() + row[:1].tolist()  # close
        color = cmap(i)
        ax.plot(angles, vals, "o-", linewidth=1.5, color=color, label=cond)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_keys, fontsize=8)
    ax.set_title(title, y=1.1)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def plot_metric_comparison_grid(
    results: dict[str, dict[str, float]],
    metric_keys: list[str],
    row_label: str = "condition",
    figsize: Optional[tuple[int, int]] = None,
) -> "matplotlib.figure.Figure":
    """Grid of bar charts — one subplot per metric.

    Args:
        results: Mapping of condition_name -> {metric_key: float}.
        metric_keys: List of metrics to plot, one per subplot.
        row_label: Label describing what the conditions represent.
        figsize: Figure size. Auto-computed if None.

    Returns:
        Matplotlib Figure with a grid of bar charts.
    """
    import matplotlib.pyplot as plt

    n = len(metric_keys)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (ncols * 5, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten() if nrows > 1 else (
        [axes] if ncols == 1 else list(axes)
    )

    conditions = list(results.keys())
    x = np.arange(len(conditions))

    for i, key in enumerate(metric_keys):
        ax = axes_flat[i]
        values = [results[c].get(key, float("nan")) for c in conditions]
        colors = ["steelblue" if not np.isnan(v) else "lightgrey" for v in values]
        ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
        ax.set_title(key, fontsize=9)
        ax.set_ylabel("value", fontsize=7)

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Metric comparison by {row_label}", fontsize=11)
    fig.tight_layout()
    return fig
