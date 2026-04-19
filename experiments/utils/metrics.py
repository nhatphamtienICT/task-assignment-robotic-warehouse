"""Shared metric utilities for all experiments."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def compute_pick_rate(total_deliveries: int, episode_steps: int) -> float:
    """Pick rate in order-lines per hour.

    Formula from scripts/run_heuristic.py:
        pick_rate = deliveries * 3600 / (5 * episode_length)
    The factor of 5 converts simulation time-steps to seconds.
    """
    if episode_steps == 0:
        return 0.0
    return total_deliveries * 3600 / (5 * episode_steps)


def aggregate_episode_info(
    infos: list[dict],
    global_episode_return: float,
    episode_returns: np.ndarray,
) -> dict[str, Any]:
    """Aggregate per-step info dicts into a single episode summary."""
    total_deliveries = 0
    total_clashes = 0
    total_stuck = 0
    for info in infos:
        total_deliveries += info["shelf_deliveries"]
        total_clashes += info["clashes"]
        total_stuck += info["stucks"]
    return {
        "episode_length": len(infos),
        "total_deliveries": total_deliveries,
        "total_clashes": total_clashes,
        "total_stuck": total_stuck,
        "global_episode_return": float(global_episode_return),
        "pick_rate": compute_pick_rate(total_deliveries, len(infos)),
    }


def compute_mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return (mean, half-width of confidence interval).

    Uses a normal approximation (z-score). Valid for n >= 30.
    Returns (mean, half_width) so the interval is mean ± half_width.
    """
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0
    std = float(np.std(values, ddof=1))
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    half_width = z * std / math.sqrt(n)
    return mean, half_width


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {path}")


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
