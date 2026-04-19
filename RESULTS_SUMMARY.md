# Results Summary — TA-RWARE Experiment Reproduction

**Paper:** Krnjaic et al. (2023). *Scalable Multi-Agent Reinforcement Learning for Warehouse
Logistics with Robotic and Human Co-Workers*. arXiv: 2212.11498v3

**Group:** [Your group name]  
**Date:** [Date of final run]

---

## 1. Paper Overview

The paper proposes **hierarchical MARL** for warehouse automation where two heterogeneous
robot types must cooperate: **AGVs** (autonomous guided vehicles that transport shelves) and
**Pickers** (stationary robots that load/unload items). The key contributions are:

- A new GTP (Goods-to-Person) simulation environment called **TA-RWARE** — the repository
  we are using.
- A **3-level manager/worker hierarchy**: a single manager assigns shelf goals to AGVs/Pickers;
  each agent has its own low-level actor that uses A* pathfinding to execute the assigned goal.
- Showing that hierarchical decomposition significantly outperforms flat MARL and classical
  heuristics across multiple warehouse sizes.

**Key findings from the paper (Table I, GTP section):**

| Algorithm | Small GTP | Large GTP |
|-----------|-----------|-----------|
| Random    | ~10       | ~11       |
| CTA (heuristic) | 52.7 ± 0.9 | 67.1 ± 0.8 |
| IAC (flat RL) | 56.4 ± 0.9 | 74.3 ± 1.0 |
| HIAC (hierarchical RL) | **66.7 ± 0.3** | **86.0 ± 0.5** |

Pick rate = order-lines delivered per hour. Higher is better.

> **Note:** The paper also covers PTG (Person-to-Goods) experiments using a commercial
> Dematic simulator that is not publicly available. We reproduce only the GTP experiments,
> which use this open-source TA-RWARE repository.

---

## 2. Scope: Which Experiments from the Paper We Reproduce

The paper (Section V-C) evaluates algorithms across **6 experiment configurations in total**:

| # | Environment | Paradigm | Simulator |
|---|-------------|----------|-----------|
| 1 | PTG Small   | Person-to-Goods | Dematic (commercial, not public) |
| 2 | PTG Medium  | Person-to-Goods | Dematic (commercial, not public) |
| 3 | PTG Large   | Person-to-Goods | Dematic (commercial, not public) |
| 4 | PTG Disjoint | Person-to-Goods | Dematic (commercial, not public) |
| 5 | **GTP Small** | Goods-to-Person | TA-RWARE (this repo, open-source) |
| 6 | **GTP Large** | Goods-to-Person | TA-RWARE (this repo, open-source) |

**We reproduce experiments 5 and 6 (2 out of 6).** Experiments 1–4 use Dematic's
proprietary PTG simulator which is not publicly available and cannot be reproduced.

Within the 2 reproducible GTP configurations, we run **3 analyses** (our "experiments"):

## 3. Experiments We Chose

We selected three analyses that are:
- **Easy to implement** (no neural network training, no GPU required)
- **Easy to explain** (classical algorithms, deterministic)
- **Academically defensible** (reproduce reported baselines, add novel scalability analysis)

### Experiment 1 — Reproduce Paper CTA Baseline (Table I validation)

Reproduce the paper's Closest Task Assignment (CTA) heuristic on both GTP configurations
reported in Table I. CTA is the FIFO heuristic already implemented in `tarware/heuristic.py`:
AGVs are assigned to the closest requested shelf; pickers follow AGVs in their zone.

**Academic value:** Validates that our environment setup matches the paper's. Without this,
no comparison is meaningful.

### Experiment 2 — Random Baseline vs CTA

Run a random-action agent (uniformly samples from valid actions at each step) on the same two
configurations, then compare against CTA. Include the paper's HIAC result as an upper bound.

**Academic value:** Establishes the floor (random) and ceiling (paper's best RL). This frames
the story: *how much does the heuristic help, and how much room remains for RL?*

### Experiment 3 — CTA Scalability Analysis (Novel Contribution)

Run CTA across all five warehouse sizes (tiny → extralarge) with agent counts proportional to
warehouse area. The paper only reports two GTP sizes; this analysis is our original contribution.

**Academic value:** Shows whether pick rate scales monotonically with warehouse size, revealing
whether CTA degrades or improves at scale — a question the paper does not address for GTP.

---

## 4. Implementation

### Environment Configurations

| Name | Env ID | Shelves | AGVs | Pickers |
|------|--------|---------|------|---------|
| Small GTP | `tarware-medium-8agvs-4pickers-partialobs-v1` | 2x5 | 8 | 4 |
| Large GTP | `tarware-extralarge-14agvs-7pickers-partialobs-v1` | 4x7 | 14 | 7 |

### Key Implementation Choices

**CTA heuristic (Exp 1 & 3):** Reuses `tarware.heuristic.heuristic_episode()` unchanged.
Each episode runs until `max_steps=500` or all agents are done.

**Random policy (Exp 2):** At each timestep, calls
`env.compute_valid_action_masks(pickers_to_agvs=False, block_conflicting_actions=False)`
to get the binary mask of legal actions per agent, then samples uniformly from valid indices.
This avoids penalising the random agent for physically impossible moves.

**Pick rate formula** (from paper + `scripts/run_heuristic.py`):
```
pick_rate = total_deliveries * 3600 / (5 * episode_steps)
```
The factor of 5 converts simulation steps to seconds (1 step = 5 s).

**Statistics:** Mean ± 95% CI using normal approximation: `1.96 * std / sqrt(n)`, valid for n=500.

### Code Structure

```
experiments/
├── utils/
│   ├── metrics.py          compute_pick_rate(), compute_mean_ci(), save/load results
│   └── plotting.py         bar charts, scalability line plots
├── exp1_cta_baseline.py    Experiment 1
├── exp2_random_vs_cta.py   Experiment 2
├── exp3_scalability.py     Experiment 3
└── run_all.py              Master script
```

---

## 5. How to Run

### Prerequisites

```bash
# Install dependencies (Python 3.11 recommended)
pip install gymnasium numpy networkx six pyglet matplotlib pyastar2d
```

### Quick Smoke Test (verify setup, ~1 min)

```bash
python experiments/run_all.py --num_episodes 10 --seed 42
```

### Full Run (500 episodes each, ~20-30 min)

```bash
python experiments/run_all.py --num_episodes 500 --seed 42
```

### Run Individual Experiments

```bash
python experiments/run_all.py --only 1   # Exp 1 only
python experiments/run_all.py --only 2   # Exp 2 only
python experiments/run_all.py --only 3   # Exp 3 only
```

Or run each script directly:

```bash
python experiments/exp1_cta_baseline.py --num_episodes 500 --seed 42
python experiments/exp2_random_vs_cta.py --num_episodes 500 --seed 42
python experiments/exp3_scalability.py --num_episodes 500 --seed 42
```

### Outputs

| File | Description |
|------|-------------|
| `experiments/results/exp1_results.json` | Exp 1 raw pick rates + stats |
| `experiments/results/exp2_results.json` | Exp 2 random vs CTA comparison |
| `experiments/results/exp3_results.json` | Exp 3 scalability data |
| `experiments/plots/exp1_bar.png` | Bar chart: our CTA vs paper CTA vs paper HIAC |
| `experiments/plots/exp2_comparison.png` | Bar chart: Random vs CTA vs paper reference |
| `experiments/plots/exp3_scalability.png` | Line plot: pick rate across all 5 sizes |

---

## 6. Results

> **[Fill in after running `python experiments/run_all.py --num_episodes 500`]**

### Experiment 1 — CTA Reproduction

| Configuration | Our CTA | Paper CTA | Difference |
|---------------|---------|-----------|------------|
| Small GTP | _____ ± _____ | 52.7 ± 0.9 | ______% |
| Large GTP | _____ ± _____ | 67.1 ± 0.8 | ______% |

![Experiment 1 Bar Chart](experiments/plots/exp1_bar.png)

### Experiment 2 — Random vs CTA

| Configuration | Random | Our CTA | Paper CTA | Paper HIAC |
|---------------|--------|---------|-----------|------------|
| Small GTP | _____ ± _____ | _____ ± _____ | 52.7 | 66.7 |
| Large GTP | _____ ± _____ | _____ ± _____ | 67.1 | 86.0 |

![Experiment 2 Comparison](experiments/plots/exp2_comparison.png)

### Experiment 3 — Scalability

| Size | Env Config | Our CTA | Paper CTA |
|------|-----------|---------|-----------|
| tiny (1x3) | 3 AGVs, 2 pickers | _____ ± _____ | — |
| small (2x3) | 5 AGVs, 3 pickers | _____ ± _____ | — |
| medium (2x5) | 8 AGVs, 4 pickers | _____ ± _____ | 52.7 ± 0.9 |
| large (3x5) | 10 AGVs, 5 pickers | _____ ± _____ | — |
| extralarge (4x7) | 14 AGVs, 7 pickers | _____ ± _____ | 67.1 ± 0.8 |

![Experiment 3 Scalability](experiments/plots/exp3_scalability.png)

---

## 7. Analysis

### Do Our CTA Results Match the Paper?

**[Fill in after running experiments]**

If our values are close to the paper's (within ~5%), this confirms our environment setup is
correct and the implementation matches. Minor differences can arise from:
- Different random seeds between our runs and the paper's
- The paper used 1,000+ evaluation episodes; we use 500
- The paper may have used a specific pyastar2d build with minor path differences

If our values are systematically lower:
- Check that the environment ID matches the paper's configuration (shelf rows/columns, agent counts)
- Verify the pick rate formula: `deliveries * 3600 / (5 * steps)`

If our values are systematically higher:
- This is less likely but could indicate a bug in the original implementation or a version difference
  in the gymnasium API (the repo migrated from OpenAI Gym to Gymnasium)

### What Does Experiment 2 Tell Us?

The random baseline is expected to score ~8-15 pick rate (confirmed by our smoke test: ~10-11).
The gap between random (~10) and CTA (~52-67) is ~5-6x — a massive improvement from a purely
rule-based heuristic. The further gap from CTA to HIAC (paper's RL result) is ~20-30% — the
reward that trained RL earns on top of human-designed heuristics.

### What Does Experiment 3 Tell Us?

The scalability analysis reveals whether pick rate grows, plateaus, or degrades with warehouse
size. Based on the paper's two data points (52.7 → 67.1 from medium to extralarge), pick rate
increases with size. This is because:
- Larger warehouses have more shelves and more concurrent requests
- With proportionally more agents, the heuristic can exploit parallelism
- The pick rate metric measures **throughput** (total deliveries/hour), not **efficiency**
  (deliveries per agent)

If our results show pick rate decreasing at some size, this would indicate the heuristic hits a
coordination ceiling — an interesting finding that directly motivates the RL approach.

---

## 8. Future Work

To further extend this project:

1. **Run the paper's RL algorithms** using EPyMARL
   (`pip install git+https://github.com/uoe-agents/epymarl`). The paper's HIAC algorithm can
   likely be reproduced by extending EPyMARL's IAC algorithm with the hierarchical manager network.

2. **Tune the heuristic**: Modify the FIFO assignment to use a weighted distance+priority metric
   instead of pure closest-distance. This might close some of the gap between CTA and HIAC.

3. **Benchmark with globalobs vs partialobs**: Our experiments use `partialobs`. Rerunning with
   `globalobs` would show whether full state information changes heuristic performance (it
   shouldn't, since the heuristic uses direct env state, but it would verify this).

4. **Agent ratio ablation**: Fix warehouse size (medium) and vary AGV:Picker ratio from 1:1 to
   4:1 to find the optimal configuration for this task structure.
