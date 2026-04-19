"""Experiment 1 — Reproduce Paper CTA Baseline (Table I, GTP section).

Goal
----
Verify that our environment setup matches the paper by running the existing
CTA/FIFO heuristic on the two GTP configurations reported in Table I.

Paper reference values (Table I):
    Small GTP  (tarware-medium-8agvs-4pickers-partialobs-v1)   CTA = 52.7 ± 0.9
    Large GTP  (tarware-extralarge-14agvs-7pickers-partialobs-v1)  CTA = 67.1 ± 0.8

Algorithm
---------
CTA (Closest Task Assignment) = the FIFO heuristic implemented in
tarware/heuristic.py.  No neural networks, no training.

Outputs
-------
    experiments/results/exp1_results.json
    experiments/plots/exp1_bar.png
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
from experiments.utils.plotting import plot_bar_comparison

# ---------------------------------------------------------------------------
# Environment configs matching Table I (GTP section) of the paper
# ---------------------------------------------------------------------------
CONFIGS = [
    ("Small GTP",  "tarware-medium-8agvs-4pickers-partialobs-v1"),
    ("Large GTP",  "tarware-extralarge-14agvs-7pickers-partialobs-v1"),
]

# Paper Table I, GTP section — for comparison only
PAPER_CTA  = {"Small GTP": 52.7, "Large GTP": 67.1}
PAPER_HIAC = {"Small GTP": 66.7, "Large GTP": 86.0}

RESULTS_PATH = Path("experiments/results/exp1_results.json")
PLOT_PATH    = Path("experiments/plots/exp1_bar.png")


def run(num_episodes: int = 500, seed: int = 42) -> dict:
    """Run the CTA heuristic on both GTP configs and return results dict."""
    results: dict[str, dict] = {}

    for config_name, env_id in CONFIGS:
        print(f"\n[Exp 1] Running CTA on {config_name} ({env_id})")
        print(f"        {num_episodes} episodes | seed={seed}")

        env = gym.make(env_id)
        raw_env = env.unwrapped
        pick_rates: list[float] = []

        for ep in range(num_episodes):
            t0 = time.time()
            infos, global_ret, ep_returns = heuristic_episode(raw_env, seed=seed + ep)
            elapsed = time.time() - t0
            summary = aggregate_episode_info(infos, global_ret, ep_returns)
            pick_rates.append(summary["pick_rate"])

            if (ep + 1) % max(1, num_episodes // 10) == 0:
                mean_so_far, ci_so_far = compute_mean_ci(pick_rates)
                print(
                    f"  ep {ep+1:4d}/{num_episodes} | "
                    f"pick_rate={summary['pick_rate']:.2f} | "
                    f"running mean={mean_so_far:.2f}+/-{ci_so_far:.2f} | "
                    f"deliveries={summary['total_deliveries']} | "
                    f"fps={summary['episode_length']/elapsed:.0f}"
                )

        env.close()
        mean, ci = compute_mean_ci(pick_rates)
        paper_cta = PAPER_CTA[config_name]
        diff_pct = (mean - paper_cta) / paper_cta * 100

        print(f"\n  === {config_name} Results ===")
        print(f"  Our CTA:    {mean:.2f} +/- {ci:.2f}")
        print(f"  Paper CTA:  {paper_cta:.1f}")
        print(f"  Difference: {diff_pct:+.1f}%")

        results[config_name] = {
            "env_id": env_id,
            "num_episodes": num_episodes,
            "seed": seed,
            "pick_rates": pick_rates,
            "mean": mean,
            "ci_half_width": ci,
            "paper_cta": paper_cta,
            "paper_hiac": PAPER_HIAC[config_name],
            "diff_pct_vs_paper_cta": diff_pct,
        }

    return results


def main(num_episodes: int = 500, seed: int = 42) -> None:
    results = run(num_episodes=num_episodes, seed=seed)
    save_results(results, RESULTS_PATH)

    config_names   = list(results.keys())
    our_values     = [results[c]["mean"]           for c in config_names]
    our_errors     = [results[c]["ci_half_width"]  for c in config_names]
    paper_cta_vals = [results[c]["paper_cta"]      for c in config_names]
    paper_hiac_vals= [results[c]["paper_hiac"]     for c in config_names]

    plot_bar_comparison(
        config_names=config_names,
        our_values=our_values,
        our_errors=our_errors,
        paper_cta_values=paper_cta_vals,
        paper_hiac_values=paper_hiac_vals,
        title="Experiment 1 — CTA Pick Rate vs Paper (Table I, GTP)",
        save_path=PLOT_PATH,
    )

    print("\n[Exp 1] Done.")
    print(f"  Results : {RESULTS_PATH}")
    print(f"  Plot    : {PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1 — CTA Baseline Reproduction")
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(num_episodes=args.num_episodes, seed=args.seed)
