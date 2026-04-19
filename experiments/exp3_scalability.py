"""Experiment 3 — Scalability Analysis of CTA Across All Warehouse Sizes.

Goal
----
Run the CTA heuristic on all five warehouse sizes (tiny → extralarge) to show
how throughput scales with warehouse complexity.  This is an original
contribution — the paper only reports two GTP sizes.

The result answers: *does pick rate improve monotonically with scale?*  If so,
larger warehouses simply have more parallel work available and CTA can exploit
it.  If not, we can identify the size at which the heuristic starts to struggle.

Configs
-------
Agent counts are chosen proportionally to each warehouse's shelf count so
comparisons are fair:

    tiny        1×3 shelves  →  3 AGVs,  2 pickers
    small       2×3 shelves  →  5 AGVs,  3 pickers
    medium      2×5 shelves  →  8 AGVs,  4 pickers  ← paper Small GTP
    large       3×5 shelves  → 10 AGVs,  5 pickers
    extralarge  4×7 shelves  → 14 AGVs,  7 pickers  ← paper Large GTP

Paper reference points (starred on plot):
    medium (Small GTP):     CTA = 52.7 ± 0.9
    extralarge (Large GTP): CTA = 67.1 ± 0.8

Outputs
-------
    experiments/results/exp3_results.json
    experiments/plots/exp3_scalability.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when this script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gymnasium as gym
import tarware  # noqa: F401 — registers environments

from tarware.heuristic import heuristic_episode
from experiments.utils.metrics import (
    aggregate_episode_info,
    compute_mean_ci,
    save_results,
)
from experiments.utils.plotting import plot_scalability

# (display_label, env_id, is_paper_reference)
SCALABILITY_CONFIGS = [
    ("tiny\n(1x3)",       "tarware-tiny-3agvs-2pickers-partialobs-v1",       False),
    ("small\n(2x3)",      "tarware-small-5agvs-3pickers-partialobs-v1",      False),
    ("medium*\n(2x5)",    "tarware-medium-8agvs-4pickers-partialobs-v1",     True),
    ("large\n(3x5)",      "tarware-large-10agvs-5pickers-partialobs-v1",     False),
    ("extralarge*\n(4x7)","tarware-extralarge-14agvs-7pickers-partialobs-v1",True),
]

# Paper CTA for the two reference configs (for star annotations on plot)
PAPER_CTA_POINTS = {
    "medium*\n(2x5)":     52.7,
    "extralarge*\n(4x7)": 67.1,
}

RESULTS_PATH = Path("experiments/results/exp3_results.json")
PLOT_PATH    = Path("experiments/plots/exp3_scalability.png")


def run(num_episodes: int = 500, seed: int = 42) -> dict:
    results: dict[str, dict] = {}

    for label, env_id, is_ref in SCALABILITY_CONFIGS:
        ref_tag = " [paper ref]" if is_ref else ""
        print(f"\n[Exp 3] {label.replace(chr(10), ' ')}{ref_tag}  ({env_id})")
        print(f"        {num_episodes} episodes | seed={seed}")

        env = gym.make(env_id)
        raw_env = env.unwrapped
        pick_rates: list[float] = []

        for ep in range(num_episodes):
            t0 = time.time()
            infos, g_ret, ep_ret = heuristic_episode(raw_env, seed=seed + ep)
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
        paper_ref = PAPER_CTA_POINTS.get(label)

        print(f"  Result: {mean:.2f} +/- {ci:.2f}", end="")
        if paper_ref is not None:
            diff = (mean - paper_ref) / paper_ref * 100
            print(f"  (paper CTA = {paper_ref:.1f}, diff = {diff:+.1f}%)", end="")
        print()

        results[label] = {
            "env_id": env_id,
            "num_episodes": num_episodes,
            "seed": seed,
            "is_paper_reference": is_ref,
            "paper_cta": paper_ref,
            "pick_rates": pick_rates,
            "mean": mean,
            "ci_half_width": ci,
        }

    return results


def main(num_episodes: int = 500, seed: int = 42) -> None:
    results = run(num_episodes=num_episodes, seed=seed)
    save_results(results, RESULTS_PATH)

    size_labels = list(results.keys())
    pick_rates  = [results[l]["mean"]          for l in size_labels]
    errors      = [results[l]["ci_half_width"] for l in size_labels]

    plot_scalability(
        size_labels=size_labels,
        pick_rates=pick_rates,
        errors=errors,
        paper_points=PAPER_CTA_POINTS,
        title="Experiment 3 - CTA Pick Rate Across All Warehouse Sizes\n(* = paper-reported configs)",
        save_path=PLOT_PATH,
    )

    print("\n[Exp 3] Summary:")
    for label in size_labels:
        r = results[label]
        ref = f"  paper CTA = {r['paper_cta']:.1f}" if r["paper_cta"] else ""
        print(f"  {label.replace(chr(10), ' '):25s}  {r['mean']:.2f} +/- {r['ci_half_width']:.2f}{ref}")

    print(f"\n  Results : {RESULTS_PATH}")
    print(f"  Plot    : {PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3 — Scalability Analysis")
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(num_episodes=args.num_episodes, seed=args.seed)
