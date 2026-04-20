"""Experiment 4 — Effect of Request Queue Size on CTA Performance.

Goal
----
Fix the medium GTP warehouse (2x5 shelves, 8 AGVs, 4 pickers) and vary the
request queue size (how many shelf requests are active simultaneously).

This answers: *does performance improve when agents have more pending tasks to
choose from?*  A small queue means agents often have nothing to do (idle time).
A large queue means more parallel work is always available.

The paper fixes queue size per warehouse size (medium = 20, extralarge = 60)
but never varies it independently.  This is our second original contribution.

Configs
-------
    queue_size in [5, 10, 20, 30, 40, 50]

    Base env: 2 shelf rows, 5 shelf columns, 8 AGVs, 4 pickers
    (medium size, same agents as paper's Small GTP)

Paper reference:
    queue_size=20 → CTA = 52.7 ± 0.9  (this is one of our data points)

Outputs
-------
    experiments/results/exp4_results.json
    experiments/plots/exp4_queue_sensitivity.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tarware  # noqa: F401

from tarware.warehouse import Warehouse
from tarware.definitions import RewardType
from tarware.heuristic import heuristic_episode
from experiments.utils.metrics import (
    aggregate_episode_info,
    compute_mean_ci,
    save_results,
)
from experiments.utils.plotting import plot_scalability

QUEUE_SIZES = [5, 10, 20, 30, 40, 50]

# Paper reference: queue_size=20 matches paper's Small GTP CTA result
PAPER_CTA_AT_20 = 52.7

RESULTS_PATH = Path("experiments/results/exp4_results.json")
PLOT_PATH    = Path("experiments/plots/exp4_queue_sensitivity.png")


def make_env(queue_size: int) -> Warehouse:
    """Create a medium GTP warehouse with a custom request queue size."""
    return Warehouse(
        shelf_rows=2,
        shelf_columns=5,
        column_height=8,
        num_agvs=8,
        num_pickers=4,
        request_queue_size=queue_size,
        max_inactivity_steps=None,
        max_steps=500,
        reward_type=RewardType.INDIVIDUAL,
        observation_type="partial",
    )


def run(num_episodes: int = 500, seed: int = 42) -> dict:
    results: dict[str, dict] = {}

    for q in QUEUE_SIZES:
        label = f"queue={q}"
        ref_tag = "  [paper ref]" if q == 20 else ""
        print(f"\n[Exp 4] queue_size={q}{ref_tag} | medium-8agvs-4pickers")
        print(f"        {num_episodes} episodes | seed={seed}")

        env = make_env(q)
        pick_rates: list[float] = []

        for ep in range(num_episodes):
            t0 = time.time()
            infos, g_ret, ep_ret = heuristic_episode(env, seed=seed + ep)
            elapsed = time.time() - t0
            summary = aggregate_episode_info(infos, g_ret, ep_ret)
            pick_rates.append(summary["pick_rate"])

            if (ep + 1) % max(1, num_episodes // 5) == 0:
                mean_so_far, ci_so_far = compute_mean_ci(pick_rates)
                print(
                    f"  ep {ep+1:4d}/{num_episodes} | "
                    f"pick_rate={summary['pick_rate']:.2f} | "
                    f"mean={mean_so_far:.2f}+/-{ci_so_far:.2f} | "
                    f"fps={summary['episode_length']/elapsed:.0f}"
                )

        env.close()
        mean, ci = compute_mean_ci(pick_rates)
        paper_ref = PAPER_CTA_AT_20 if q == 20 else None

        print(f"  Result: {mean:.2f} +/- {ci:.2f}", end="")
        if paper_ref is not None:
            diff = (mean - paper_ref) / paper_ref * 100
            print(f"  (paper CTA = {paper_ref:.1f}, diff = {diff:+.1f}%)", end="")
        print()

        results[label] = {
            "queue_size": q,
            "num_episodes": num_episodes,
            "seed": seed,
            "is_paper_reference": q == 20,
            "paper_cta": paper_ref,
            "pick_rates": pick_rates,
            "mean": mean,
            "ci_half_width": ci,
        }

    return results


def main(num_episodes: int = 500, seed: int = 42) -> None:
    results = run(num_episodes=num_episodes, seed=seed)
    save_results(results, RESULTS_PATH)

    labels     = [r["label"] if "label" in r else f"queue={r['queue_size']}"
                  for r in results.values()]
    labels     = [f"queue={results[k]['queue_size']}" for k in results]
    pick_rates = [results[k]["mean"]          for k in results]
    errors     = [results[k]["ci_half_width"] for k in results]

    # Mark the paper reference point
    paper_points = {f"queue={q}": PAPER_CTA_AT_20 for q in [20]}

    plot_scalability(
        size_labels=labels,
        pick_rates=pick_rates,
        errors=errors,
        paper_points=paper_points,
        title=(
            "Experiment 4 - CTA Pick Rate vs Request Queue Size\n"
            "(medium warehouse: 2x5 shelves, 8 AGVs, 4 pickers | * = paper ref)"
        ),
        save_path=PLOT_PATH,
    )

    print("\n[Exp 4] Summary:")
    for k, r in results.items():
        ref = f"  paper CTA = {r['paper_cta']:.1f}" if r["paper_cta"] else ""
        print(f"  queue_size={r['queue_size']:2d}   {r['mean']:.2f} +/- {r['ci_half_width']:.2f}{ref}")

    print(f"\n  Results : {RESULTS_PATH}")
    print(f"  Plot    : {PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4 - Queue Size Sensitivity")
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(num_episodes=args.num_episodes, seed=args.seed)
