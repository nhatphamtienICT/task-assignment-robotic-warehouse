"""Experiment 2 — Random Baseline vs CTA.

Goal
----
Establish a lower bound by comparing a random-action policy against the CTA
heuristic.  The story for presentation is:

    Random << Our CTA ≈ Paper CTA << Paper HIAC (RL upper bound)

This demonstrates two things:
  1. The heuristic provides substantial value over a no-policy baseline.
  2. There is still a meaningful gap between the heuristic and trained RL agents,
     motivating future RL work.

Random policy
-------------
At each step, each agent samples uniformly from its *valid* actions (determined
by compute_valid_action_masks).  Using valid actions avoids penalising the random
policy for impossible moves that the environment would anyway no-op.

Outputs
-------
    experiments/results/exp2_results.json
    experiments/plots/exp2_comparison.png
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
import numpy as np
import tarware  # noqa: F401 — registers environments

from tarware.heuristic import heuristic_episode
from experiments.utils.metrics import (
    aggregate_episode_info,
    compute_mean_ci,
    save_results,
)
from experiments.utils.plotting import plot_bar_comparison

CONFIGS = [
    ("Small GTP",  "tarware-medium-8agvs-4pickers-partialobs-v1"),
    ("Large GTP",  "tarware-extralarge-14agvs-7pickers-partialobs-v1"),
]

PAPER_CTA  = {"Small GTP": 52.7, "Large GTP": 67.1}
PAPER_HIAC = {"Small GTP": 66.7, "Large GTP": 86.0}

RESULTS_PATH = Path("experiments/results/exp2_results.json")
PLOT_PATH    = Path("experiments/plots/exp2_comparison.png")


def random_episode(
    env,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[list[dict], float, np.ndarray]:
    """Run one episode with valid-random actions.

    Args:
        env: Raw (unwrapped) Warehouse environment.
        rng: NumPy random generator for reproducible sampling.
        seed: Seed forwarded to env.reset().

    Returns:
        (all_infos, global_episode_return, episode_returns)  — same shape as
        heuristic_episode so metrics.aggregate_episode_info works unchanged.
    """
    env.reset(seed=seed)
    done = False
    all_infos: list[dict] = []
    global_return = 0.0
    episode_returns = np.zeros(env.num_agents)

    while not done:
        # Get valid action mask: shape (num_agents, action_size), 1 = valid
        masks = env.compute_valid_action_masks(
            pickers_to_agvs=False, block_conflicting_actions=False
        )
        actions = [
            int(rng.choice(np.where(row)[0]))
            for row in masks
        ]

        _, reward, terminated, truncated, info = env.step(actions)
        done = all(terminated) or all(truncated)
        episode_returns += np.array(reward, dtype=np.float64)
        global_return += float(np.sum(reward))
        all_infos.append(info)

    return all_infos, global_return, episode_returns


def _run_policy(env_id: str, policy: str, num_episodes: int, seed: int) -> list[float]:
    """Run `num_episodes` of the given policy and return per-episode pick rates."""
    env = gym.make(env_id)
    raw_env = env.unwrapped
    pick_rates: list[float] = []
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        if policy == "cta":
            infos, g_ret, ep_ret = heuristic_episode(raw_env, seed=seed + ep)
        else:
            infos, g_ret, ep_ret = random_episode(raw_env, rng=rng, seed=seed + ep)

        summary = aggregate_episode_info(infos, g_ret, ep_ret)
        pick_rates.append(summary["pick_rate"])

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            mean_so_far, _ = compute_mean_ci(pick_rates)
            print(f"    ep {ep+1:4d}/{num_episodes} | pick_rate={pick_rates[-1]:.2f} | mean={mean_so_far:.2f}")

    env.close()
    return pick_rates


def run(num_episodes: int = 500, seed: int = 42) -> dict:
    results: dict[str, dict] = {}

    for config_name, env_id in CONFIGS:
        print(f"\n[Exp 2] {config_name} ({env_id})")

        print("  -> Running Random policy...")
        t0 = time.time()
        random_rates = _run_policy(env_id, "random", num_episodes, seed)
        print(f"    Done in {time.time()-t0:.1f}s")

        print("  -> Running CTA policy...")
        t0 = time.time()
        cta_rates = _run_policy(env_id, "cta", num_episodes, seed)
        print(f"    Done in {time.time()-t0:.1f}s")

        rand_mean, rand_ci = compute_mean_ci(random_rates)
        cta_mean,  cta_ci  = compute_mean_ci(cta_rates)
        improvement = (cta_mean - rand_mean) / max(rand_mean, 1e-9) * 100

        print(f"\n  === {config_name} Summary ===")
        print(f"  Random:     {rand_mean:.2f} +/- {rand_ci:.2f}")
        print(f"  Our CTA:    {cta_mean:.2f} +/- {cta_ci:.2f}  (+{improvement:.1f}% vs random)")
        print(f"  Paper CTA:  {PAPER_CTA[config_name]:.1f}")
        print(f"  Paper HIAC: {PAPER_HIAC[config_name]:.1f}  (RL upper bound)")

        results[config_name] = {
            "env_id": env_id,
            "num_episodes": num_episodes,
            "seed": seed,
            "random_pick_rates": random_rates,
            "cta_pick_rates": cta_rates,
            "random_mean": rand_mean,
            "random_ci": rand_ci,
            "cta_mean": cta_mean,
            "cta_ci": cta_ci,
            "paper_cta": PAPER_CTA[config_name],
            "paper_hiac": PAPER_HIAC[config_name],
            "pct_improvement_cta_over_random": improvement,
        }

    return results


def main(num_episodes: int = 500, seed: int = 42) -> None:
    results = run(num_episodes=num_episodes, seed=seed)
    save_results(results, RESULTS_PATH)

    config_names = list(results.keys())

    our_cta_values  = [results[c]["cta_mean"]    for c in config_names]
    our_cta_errors  = [results[c]["cta_ci"]      for c in config_names]
    rand_values     = [results[c]["random_mean"] for c in config_names]
    rand_errors     = [results[c]["random_ci"]   for c in config_names]
    paper_cta_vals  = [results[c]["paper_cta"]   for c in config_names]
    paper_hiac_vals = [results[c]["paper_hiac"]  for c in config_names]

    plot_bar_comparison(
        config_names=config_names,
        our_values=our_cta_values,
        our_errors=our_cta_errors,
        extra_values={"Random Baseline": rand_values},
        extra_errors={"Random Baseline": rand_errors},
        paper_cta_values=paper_cta_vals,
        paper_hiac_values=paper_hiac_vals,
        title="Experiment 2 — Random vs CTA vs Paper (GTP Configs)",
        save_path=PLOT_PATH,
    )

    print("\n[Exp 2] Done.")
    print(f"  Results : {RESULTS_PATH}")
    print(f"  Plot    : {PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2 — Random vs CTA")
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(num_episodes=args.num_episodes, seed=args.seed)
