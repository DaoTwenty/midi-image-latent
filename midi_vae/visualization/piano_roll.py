"""Piano roll visualization utilities.

Provides functions to render piano-roll images with matplotlib, including
side-by-side comparisons of ground-truth vs reconstructed bars.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def plot_piano_roll(
    image: torch.Tensor,
    title: str = "Piano Roll",
    ax=None,
    cmap: str = "hot",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> "matplotlib.axes.Axes":
    """Plot a single piano-roll image as a heatmap.

    Displays the velocity channel (R) of the image. Pitch is on the Y-axis
    and time steps on the X-axis.

    Args:
        image: Tensor of shape (3, H, W) or (H, W), values in [-1, 1].
        title: Title for the plot.
        ax: Matplotlib Axes to draw on. Creates a new figure if None.
        cmap: Matplotlib colormap name.
        vmin: Minimum value for color normalization.
        vmax: Maximum value for color normalization.

    Returns:
        The Axes object with the piano roll plotted.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)

    # Use R channel (velocity) if 3-channel, else use as-is
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        velocity = img_np[0]  # (H, W)
    elif img_np.ndim == 3 and img_np.shape[2] == 3:
        velocity = img_np[:, :, 0]  # (H, W)
    else:
        velocity = img_np  # (H, W)

    ax.imshow(
        velocity,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(title)
    return ax


def plot_gt_vs_recon(
    gt_image: torch.Tensor,
    recon_image: torch.Tensor,
    bar_id: str = "",
    vae_name: str = "",
    channel_strategy: str = "velocity_only",
    figsize: tuple[int, int] = (16, 5),
) -> "matplotlib.figure.Figure":
    """Plot ground-truth and reconstructed piano-roll images side by side.

    Displays three panels when channel_strategy is 'vos': velocity,
    onset, and sustain channels for both GT and reconstruction, plus a
    per-pixel difference heatmap.

    Args:
        gt_image: Ground-truth image tensor, shape (3, H, W).
        recon_image: Reconstructed image tensor, shape (3, H, W).
        bar_id: Bar identifier for the plot title.
        vae_name: VAE name for the subtitle.
        channel_strategy: One of 'velocity_only', 'vo_split', 'vos'.
        figsize: Figure width and height in inches.

    Returns:
        Matplotlib Figure with the comparison plot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Ground truth
    plot_piano_roll(gt_image, title=f"GT — {bar_id}", ax=axes[0])

    # Reconstruction
    plot_piano_roll(recon_image, title=f"Recon — {vae_name}", ax=axes[1])

    # Absolute difference
    if isinstance(gt_image, torch.Tensor):
        gt_np = gt_image.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_image)

    if isinstance(recon_image, torch.Tensor):
        recon_np = recon_image.detach().cpu().numpy()
    else:
        recon_np = np.asarray(recon_image)

    diff = np.abs(gt_np[0] - recon_np[0])  # velocity channel difference
    im = axes[2].imshow(
        diff,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[2].set_title("Abs Difference (velocity)")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("MIDI pitch")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Piano Roll Comparison — channel: {channel_strategy}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_channel_breakdown(
    image: torch.Tensor,
    channel_strategy: str = "vos",
    bar_id: str = "",
    figsize: tuple[int, int] = (18, 4),
) -> "matplotlib.figure.Figure":
    """Plot all three image channels side by side.

    Shows R (velocity), G (onset or velocity), and B (sustain or zeros)
    channels separately so channel assignment can be verified visually.

    Args:
        image: Image tensor of shape (3, H, W).
        channel_strategy: Strategy name to label each channel correctly.
        bar_id: Bar identifier for the title.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure with three channel subplots.
    """
    import matplotlib.pyplot as plt

    channel_labels = {
        "velocity_only": ["velocity", "velocity", "velocity"],
        "vo_split": ["velocity", "onset", "zeros"],
        "vos": ["velocity", "onset", "sustain"],
    }
    labels = channel_labels.get(channel_strategy, ["R", "G", "B"])

    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    cmaps = ["hot", "Blues", "Greens"]

    for i, (ax, label, cmap) in enumerate(zip(axes, labels, cmaps)):
        ax.imshow(
            img_np[i],
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(f"Ch {i}: {label}")
        ax.set_xlabel("Time step")
        if i == 0:
            ax.set_ylabel("MIDI pitch")

    fig.suptitle(
        f"Channel breakdown ({channel_strategy}) — {bar_id}", fontsize=11
    )
    fig.tight_layout()
    return fig


def plot_note_overlay(
    image: torch.Tensor,
    detected_notes: list,
    gt_notes: Optional[list] = None,
    title: str = "Detected Notes",
    figsize: tuple[int, int] = (12, 5),
) -> "matplotlib.figure.Figure":
    """Overlay detected (and optionally ground-truth) notes on a piano-roll image.

    Detected notes are drawn as blue rectangles; ground-truth notes (if
    provided) are drawn as green rectangles with transparency.

    Args:
        image: Image tensor, shape (3, H, W).
        detected_notes: List of MidiNote objects from detection.
        gt_notes: Optional list of ground-truth MidiNote objects.
        title: Figure title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with note overlays.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    plot_piano_roll(image, title=title, ax=ax)

    # Draw detected notes
    for note in detected_notes:
        width = note.offset_step - note.onset_step
        rect = mpatches.Rectangle(
            (note.onset_step - 0.5, note.pitch - 0.5),
            width,
            1,
            linewidth=1,
            edgecolor="cyan",
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(rect)

    # Draw ground-truth notes (if provided)
    if gt_notes:
        for note in gt_notes:
            width = note.offset_step - note.onset_step
            rect = mpatches.Rectangle(
                (note.onset_step - 0.5, note.pitch - 0.5),
                width,
                1,
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
                alpha=0.7,
                linestyle="--",
            )
            ax.add_patch(rect)

        # Legend
        detected_patch = mpatches.Patch(
            edgecolor="cyan", facecolor="none", label="Detected"
        )
        gt_patch = mpatches.Patch(
            edgecolor="lime", facecolor="none", linestyle="--", label="GT"
        )
        ax.legend(handles=[detected_patch, gt_patch], loc="upper right")
    else:
        detected_patch = mpatches.Patch(
            edgecolor="cyan", facecolor="none", label="Detected"
        )
        ax.legend(handles=[detected_patch], loc="upper right")

    fig.tight_layout()
    return fig
