"""Master script — run all three experiments sequentially.

Usage
-----
    # Quick smoke test (10 episodes each):
    python experiments/run_all.py --num_episodes 10 --seed 42

    # Full run (500 episodes each, ~20-30 min total):
    python experiments/run_all.py --num_episodes 500 --seed 42

    # Run a single experiment:
    python experiments/run_all.py --only 1
    python experiments/run_all.py --only 2
    python experiments/run_all.py --only 3

Output files
------------
    experiments/results/exp1_results.json
    experiments/results/exp2_results.json
    experiments/results/exp3_results.json
    experiments/plots/exp1_bar.png
    experiments/plots/exp2_comparison.png
    experiments/plots/exp3_scalability.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `import experiments` works
# regardless of where this script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _check_dependencies() -> None:
    missing = []
    for pkg in ("matplotlib", "numpy", "gymnasium"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[run_all] Missing dependencies: {', '.join(missing)}")
        print("Install with:  pip install " + " ".join(missing))
        sys.exit(1)


def main() -> None:
    _check_dependencies()

    parser = argparse.ArgumentParser(description="Run all TA-RWARE experiments")
    parser.add_argument(
        "--num_episodes", type=int, default=500,
        help="Episodes per environment config (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--only", type=int, choices=[1, 2, 3], default=None,
        help="Run only one experiment by number",
    )
    args = parser.parse_args()

    # Import after dependency check so errors are clear
    import experiments.exp1_cta_baseline as exp1
    import experiments.exp2_random_vs_cta as exp2
    import experiments.exp3_scalability   as exp3

    run_flags = {1: True, 2: True, 3: True}
    if args.only:
        run_flags = {k: (k == args.only) for k in run_flags}

    wall_start = time.time()
    print("=" * 60)
    print("  TA-RWARE Experiment Suite")
    print(f"  episodes={args.num_episodes}  seed={args.seed}")
    print("=" * 60)

    if run_flags[1]:
        t0 = time.time()
        print("\n>>> EXPERIMENT 1 - Reproduce Paper CTA Baseline")
        exp1.main(num_episodes=args.num_episodes, seed=args.seed)
        print(f"    Elapsed: {time.time()-t0:.1f}s")

    if run_flags[2]:
        t0 = time.time()
        print("\n>>> EXPERIMENT 2 - Random Baseline vs CTA")
        exp2.main(num_episodes=args.num_episodes, seed=args.seed)
        print(f"    Elapsed: {time.time()-t0:.1f}s")

    if run_flags[3]:
        t0 = time.time()
        print("\n>>> EXPERIMENT 3 - Scalability Analysis")
        exp3.main(num_episodes=args.num_episodes, seed=args.seed)
        print(f"    Elapsed: {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print(f"  All experiments done in {time.time()-wall_start:.1f}s")
    print("  Results in: experiments/results/")
    print("  Plots   in: experiments/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
