"""Latent space visualization utilities.

Provides functions to plot PCA/t-SNE/UMAP scatter plots of latent vectors,
and per-channel heatmaps of the latent space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def plot_pca_scatter(
    latents: torch.Tensor,
    labels: Optional[list] = None,
    label_name: str = "instrument",
    title: str = "PCA of Latent Space",
    n_components: int = 2,
    figsize: tuple[int, int] = (8, 6),
) -> "matplotlib.figure.Figure":
    """Plot a 2D PCA scatter of latent vectors.

    Runs PCA (via sklearn or torch.linalg.svd) on the input latents and
    plots the first two principal components. Points are coloured by label
    if provided.

    Args:
        latents: Latent vectors, shape (N, D) or (N, C, H, W) — flattened
            automatically if more than 2 dims.
        labels: Optional list of N label values for colouring.
        label_name: Human-readable name for the label in the legend.
        title: Figure title.
        n_components: Number of PCA components (2 or 3 for scatter).
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure with the scatter plot.
    """
    import matplotlib.pyplot as plt

    if isinstance(latents, torch.Tensor):
        z = latents.detach().cpu().float()
    else:
        z = torch.tensor(latents, dtype=torch.float32)

    if z.ndim > 2:
        z = z.view(z.shape[0], -1)

    # Centre
    mean = z.mean(dim=0, keepdim=True)
    z_c = z - mean

    # SVD for PCA
    _, _, Vh = torch.linalg.svd(z_c, full_matrices=False)
    components = Vh[:n_components]  # (n_components, D)
    projected = (z_c @ components.T).numpy()  # (N, n_components)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = sorted(set(labels))
        cmap = plt.get_cmap("tab10", len(unique_labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        colors = [cmap(label_to_idx[lab]) for lab in labels]
        scatter = ax.scatter(
            projected[:, 0], projected[:, 1], c=colors, s=10, alpha=0.6
        )
        # Legend
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(i),
                markersize=8,
                label=str(lab),
            )
            for i, lab in enumerate(unique_labels)
        ]
        ax.legend(handles=handles, title=label_name, bbox_to_anchor=(1.05, 1))
    else:
        ax.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.5, color="steelblue")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_umap_scatter(
    latents: torch.Tensor,
    labels: Optional[list] = None,
    label_name: str = "instrument",
    title: str = "UMAP of Latent Space",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    figsize: tuple[int, int] = (8, 6),
) -> "matplotlib.figure.Figure":
    """Plot a 2D UMAP scatter of latent vectors.

    Requires ``umap-learn`` to be installed. Falls back to PCA with a
    warning if umap-learn is not available.

    Args:
        latents: Latent vectors, shape (N, D) or (N, C, H, W).
        labels: Optional list of N labels for colouring.
        label_name: Human-readable label name for the legend.
        title: Figure title.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with the UMAP (or PCA fallback) scatter.
    """
    import matplotlib.pyplot as plt

    if isinstance(latents, torch.Tensor):
        z = latents.detach().cpu().float().numpy()
    else:
        z = np.asarray(latents, dtype=np.float32)

    if z.ndim > 2:
        z = z.reshape(z.shape[0], -1)

    try:
        import umap

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
        projected = reducer.fit_transform(z)
        method_label = "UMAP"
    except ImportError:
        import warnings
        warnings.warn(
            "umap-learn not installed; falling back to PCA for scatter plot.",
            UserWarning,
            stacklevel=2,
        )
        return plot_pca_scatter(
            torch.tensor(z),
            labels=labels,
            label_name=label_name,
            title=title.replace("UMAP", "PCA (fallback)"),
            figsize=figsize,
        )

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = sorted(set(labels))
        cmap = plt.get_cmap("tab10", len(unique_labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        colors = [cmap(label_to_idx[lab]) for lab in labels]
        ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=10, alpha=0.6)
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=cmap(i), markersize=8, label=str(lab)
            )
            for i, lab in enumerate(unique_labels)
        ]
        ax.legend(handles=handles, title=label_name, bbox_to_anchor=(1.05, 1))
    else:
        ax.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.5, color="coral")

    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_latent_channel_heatmaps(
    latent: torch.Tensor,
    vae_name: str = "",
    bar_id: str = "",
    max_channels: int = 16,
    figsize: Optional[tuple[int, int]] = None,
) -> "matplotlib.figure.Figure":
    """Plot per-channel heatmaps of a single latent encoding.

    Visualises each channel of the latent tensor (C, H, W) as a separate
    heatmap. Useful for diagnosing which channels carry musical structure.

    Args:
        latent: Single latent tensor, shape (C, H, W).
        vae_name: VAE name for the title.
        bar_id: Bar identifier for the title.
        max_channels: Maximum number of channels to display.
        figsize: Figure size. Auto-computed from n_channels if None.

    Returns:
        Matplotlib Figure with one subplot per channel.
    """
    import matplotlib.pyplot as plt

    if isinstance(latent, torch.Tensor):
        z_np = latent.detach().cpu().numpy()
    else:
        z_np = np.asarray(latent)

    C = min(z_np.shape[0], max_channels)
    ncols = 4
    nrows = (C + ncols - 1) // ncols

    if figsize is None:
        figsize = (ncols * 3, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten() if nrows > 1 else [axes] if ncols == 1 else axes

    vmax = float(np.abs(z_np[:C]).max()) + 1e-8

    for i in range(C):
        ax = axes_flat[i]
        ax.imshow(
            z_np[i],
            aspect="equal",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"Ch {i}", fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for j in range(C, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Latent channels — {vae_name} — {bar_id}", fontsize=10
    )
    fig.tight_layout()
    return fig


def plot_latent_variance(
    z_mu_batch: torch.Tensor,
    z_sigma_batch: torch.Tensor,
    vae_name: str = "",
    figsize: tuple[int, int] = (10, 4),
) -> "matplotlib.figure.Figure":
    """Plot per-channel mean and standard deviation statistics of a batch.

    Shows the spatial-mean activation and variance for each latent channel
    across a batch of encodings — helpful for diagnosing posterior collapse.

    Args:
        z_mu_batch: Batch of latent means, shape (B, C, H, W).
        z_sigma_batch: Batch of latent sigmas, shape (B, C, H, W).
        vae_name: VAE name for the title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with mean and sigma bar charts per channel.
    """
    import matplotlib.pyplot as plt

    if isinstance(z_mu_batch, torch.Tensor):
        mu = z_mu_batch.detach().cpu().float()
        sigma = z_sigma_batch.detach().cpu().float()
    else:
        mu = torch.tensor(z_mu_batch, dtype=torch.float32)
        sigma = torch.tensor(z_sigma_batch, dtype=torch.float32)

    C = mu.shape[1]
    channel_means = mu.view(mu.shape[0], C, -1).mean(dim=(0, 2)).numpy()
    channel_sigmas = sigma.view(sigma.shape[0], C, -1).mean(dim=(0, 2)).numpy()

    channels = np.arange(C)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.bar(channels, channel_means, color="steelblue")
    ax1.set_xlabel("Latent channel")
    ax1.set_ylabel("Mean activation")
    ax1.set_title("Mean per channel")
    ax1.axhline(0, color="black", linewidth=0.5)

    ax2.bar(channels, channel_sigmas, color="coral")
    ax2.set_xlabel("Latent channel")
    ax2.set_ylabel("Mean sigma")
    ax2.set_title("Sigma per channel")

    fig.suptitle(f"Latent statistics — {vae_name}", fontsize=11)
    fig.tight_layout()
    return fig
