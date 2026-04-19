"""Shared plotting utilities for all experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_bar_comparison(
    config_names: list[str],
    our_values: list[float],
    our_errors: list[float],
    paper_cta_values: list[float] | None = None,
    paper_hiac_values: list[float] | None = None,
    extra_values: dict[str, list[float]] | None = None,
    extra_errors: dict[str, list[float]] | None = None,
    title: str = "Pick Rate Comparison",
    ylabel: str = "Pick Rate (order-lines / hour)",
    save_path: str | Path | None = None,
) -> None:
    """Grouped bar chart comparing our results with paper reference values.

    Args:
        config_names: X-axis labels (one per warehouse config).
        our_values: Our measured pick rates.
        our_errors: Half-widths of 95% CI for our values.
        paper_cta_values: Paper's CTA baseline (orange bars).
        paper_hiac_values: Paper's best RL result (green bars).
        extra_values: Additional named series {label: values}.
        extra_errors: CI half-widths for extra series.
        title: Chart title.
        ylabel: Y-axis label.
        save_path: If given, save to file instead of displaying.
    """
    extra_values = extra_values or {}
    extra_errors = extra_errors or {}

    n_groups = len(config_names)
    n_bars = 1  # ours
    if paper_cta_values:
        n_bars += 1
    if paper_hiac_values:
        n_bars += 1
    n_bars += len(extra_values)

    width = 0.8 / n_bars
    x = np.arange(n_groups)
    bar_idx = 0

    fig, ax = plt.subplots(figsize=(max(8, 3 * n_groups), 6))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    ax.bar(
        x + offsets[bar_idx],
        our_values,
        width,
        yerr=our_errors,
        capsize=5,
        label="Our CTA",
        color=colors[0],
        alpha=0.85,
    )
    bar_idx += 1

    for label, vals in extra_values.items():
        errs = extra_errors.get(label, [0] * len(vals))
        ax.bar(
            x + offsets[bar_idx],
            vals,
            width,
            yerr=errs,
            capsize=5,
            label=label,
            color=colors[bar_idx % len(colors)],
            alpha=0.85,
        )
        bar_idx += 1

    if paper_cta_values:
        ax.bar(
            x + offsets[bar_idx],
            paper_cta_values,
            width,
            label="Paper CTA (reference)",
            color=colors[bar_idx % len(colors)],
            alpha=0.85,
        )
        bar_idx += 1

    if paper_hiac_values:
        ax.bar(
            x + offsets[bar_idx],
            paper_hiac_values,
            width,
            label="Paper HIAC — best RL (reference)",
            color=colors[bar_idx % len(colors)],
            alpha=0.85,
        )

    ax.set_xlabel("Warehouse Configuration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    _save_or_show(fig, save_path)


def plot_scalability(
    size_labels: list[str],
    pick_rates: list[float],
    errors: list[float],
    paper_points: dict[str, float] | None = None,
    title: str = "CTA Scalability Across Warehouse Sizes",
    save_path: str | Path | None = None,
) -> None:
    """Line plot of pick rate vs. warehouse size with error bands.

    Args:
        size_labels: X-axis labels (tiny → extralarge).
        pick_rates: Mean pick rate per size.
        errors: CI half-widths per size.
        paper_points: Dict mapping size_label → paper pick rate (plotted as stars).
        title: Chart title.
        save_path: If given, save to file instead of displaying.
    """
    paper_points = paper_points or {}
    x = np.arange(len(size_labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, pick_rates, marker="o", color="#2196F3", linewidth=2, label="Our CTA")
    ax.fill_between(
        x,
        [v - e for v, e in zip(pick_rates, errors)],
        [v + e for v, e in zip(pick_rates, errors)],
        alpha=0.2,
        color="#2196F3",
    )

    for label, paper_val in paper_points.items():
        if label in size_labels:
            idx = size_labels.index(label)
            ax.scatter(
                [idx],
                [paper_val],
                marker="*",
                s=200,
                color="#FF9800",
                zorder=5,
                label=f"Paper CTA — {label}",
            )

    ax.set_xlabel("Warehouse Size")
    ax.set_ylabel("Pick Rate (order-lines / hour)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()
