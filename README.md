# AI-Powered Multi-Robot Coordination in Dynamic Environments

A university project for the **Design and Analysis of Algorithms** course. The project studies how a team of robots can allocate delivery tasks, navigate a warehouse-like environment, respond to changing obstacles and tasks, and balance travel efficiency with urgent-delivery priorities.

## Overview

The simulator models multiple robots on a configurable 2D grid with shelves, moving obstacles, and delivery tasks introduced during execution. Robots navigate using **A\*** with replanning and collision-avoidance safeguards. Task assignment is evaluated with **Auction**, **Hungarian** variants, and a **parameter-sharing Double Deep Q-Network (Double DQN)** policy.

The reinforcement-learning component learns **task-allocation decisions**, not robot motor controls or individual pathfinding steps. Movement and collision avoidance are handled separately by the simulation and A* planner.

### Main capabilities

- A* route planning and replanning as routes become blocked
- Multi-robot movement with collision-prevention and yielding mechanisms
- Dynamic task generation, moving obstacles, and normal/urgent priorities
- Auction-based, Hungarian V3/V4, and learned Double DQN task assignment
- Reproducible, seed-based comparison of allocation strategies
- Best-found solution, per-seed delta/gap, and retrospective best-so-far tracking
- Tkinter GUI with routes, robot status, live metrics, algorithm comparison, simulation summary, and CSV export
- Experiments on team-feature sensitivity, controlled fine-tuning ablation, scalability, and zero-shot generalization

## Reference scenario

The primary benchmark uses a **20×20 grid**, **5 robots**, shelves/static obstacles, **5 moving obstacles**, and **40 dynamically generated delivery tasks**. Tasks can be normal (priority 1) or urgent (priority 3). The task generator introduces new work over time. Experiments use fixed random seeds for reproducible comparisons.

Scalability tests additionally consider **20×20 with 8 robots**, **30×30 with 5 robots**, and **30×30 with 10 robots**. These are not identical difficulty levels: changing the grid can change path lengths and obstacle layouts.

## Algorithms and design

| Component | Approach | Role |
| --- | --- | --- |
| Route planning | A* and online replanning | Find and update routes to pickup/delivery locations |
| Safe coordination | Collision prevention and yielding | Resolve occupied cells and robot interactions |
| Assignment baseline | Auction | Market-style assignment based on task bids/costs |
| Assignment baseline | Hungarian V3 | Cost-based matching of robots to tasks |
| Priority-focused assignment | Hungarian V4 | Modified assignment behavior favoring urgent-task performance |
| Learned assignment | Parameter-sharing Double DQN (MARL V4 Dynamic 20) | Score candidate robot–task assignments using 17 features |

The learned task-assignment network uses **12 basic robot/task features plus 5 team-aware features**: nearest-robot proximity, competing robot's pickup distance, relative assignment advantage, pending urgent-task fraction, and idle-robot fraction.

### Training and reward caveat

The dynamic Double DQN training script (`experiments/train_marl_v4_dynamic.py`) uses a decision-level reward based on elapsed steps/movement, a completion bonus (**+12 normal**, **+24 urgent**), and a penalty for assignments that terminate without completion. The separate `src/marl_environment_v2.py` defines a shared team reward with explicit collision and waiting penalties, but **that environment reward is not directly consumed by the final dynamic training loop**. Collision avoidance is primarily enforced by simulation logic rather than learned from a collision penalty in that final loop.

## Results

### Main benchmark: 100 held-out scenarios (20×20, 5 robots)

All six tested strategies completed all 100 scenarios, with no recorded collisions. Lower values are better.

| Algorithm | Success | Mean finish steps | Mean robot distance | Mean normal-task completion | Mean urgent-task completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Auction | 100/100 | 213.690 | 907.860 | 109.410 | 41.922 |
| Hungarian V3 | 100/100 | 213.940 | 907.890 | 109.413 | 41.824 |
| Hungarian V4 | 100/100 | **211.940** | 904.660 | 112.433 | **38.337** |
| MARL Original | 100/100 | 216.410 | 917.870 | 108.632 | 43.076 |
| MARL Dynamic 10 | 100/100 | 214.350 | 909.790 | 106.904 | 43.094 |
| MARL Dynamic 20 | 100/100 | 212.440 | **903.230** | **103.946** | 44.366 |

**Interpretation:** Hungarian V4 has the lowest mean finish time and best urgent-task completion time, while Dynamic 20 has the lowest mean travel distance and normal-task completion time. No one policy wins every metric. Numerical advantages should not automatically be interpreted as statistically significant.

### Best-found, delta, and best-so-far analysis

For **each scenario and each metric separately**, the best-found reference is the lowest observed cost among the compared strategies. The per-scenario difference is `delta = candidate_cost - best_found`, and relative gap is `100 * delta / best_found` for nonzero positive costs.

A separate retrospective tracking experiment compares Auction → Hungarian V3 → Hungarian V4 → MARL Original → Dynamic 10 → Dynamic 20, updating the **best-so-far** cost whenever a candidate improves it. This ordering is **an analysis convention, not a verified development timeline**. The mean per-scenario best-so-far distance improves from **907.860** (initial Auction baseline) to **876.770** (best among six strategies chosen separately for each seed). This is an **empirical best-known reference, not a proof of a global optimum**, nor the performance of any one deployable strategy.

### Team-information experiments

- **Inference sensitivity (100 scenarios):** full 17-feature model: 212.440 finish steps / 903.230 distance; neutralized team features: 216.840 steps / 922.690 distance.
- **Controlled fine-tuning information ablation (30 scenarios):** full team information: 213.033 steps / 915.200 distance; neutralized team information during train and test: 223.500 steps / 958.233 distance.

Both ablation variants use the same initial pretrained 17-feature architecture. This is a **matched fine-tuning information ablation**, *not* independent training from scratch. A single training seed is insufficient for robust significance claims. Neutralizing pretrained inputs at inference alone is not equivalent to a controlled training ablation.

### Scalability and zero-shot generalization (30 seeds per configuration)

All algorithm/configuration combinations below completed **30/30** runs. The learned model was applied **without retraining** outside its original 20×20/5-robot setup.

| Grid / robots | Auction mean steps | Hungarian V3 | Hungarian V4 | MARL Dynamic 20 |
| --- | ---: | ---: | ---: | ---: |
| 20×20 / 5 | 214.67 | **212.10** | 213.03 | 214.03 |
| 20×20 / 8 | 156.03 | 154.97 | 155.03 | **151.13** |
| 30×30 / 5 | 303.20 | **303.03** | 307.70 | 307.37 |
| 30×30 / 10 | 184.80 | 181.60 | 183.40 | **181.17** |

These are experimental observations over a limited range of robot counts and grid sizes. They do not establish asymptotic scalability or a formal optimality guarantee.

## Getting started (Windows / PowerShell)

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/elda-ferati/multi-robot-coordination.git
cd multi-robot-coordination
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch
```

The GUI uses Tkinter (typically included with Windows Python). The Python code also uses standard-library modules. If another package import is reported missing, install the package in `.venv` before continuing.

Launch the GUI:

```powershell
.\.venv\Scripts\python.exe gui.py
```

Run the built-in project check:

```powershell
.\.venv\Scripts\python.exe final_project_check.py
```

Display the saved main-algorithm comparison through the GUI's **Open Algorithm Comparison** option or, where supported, by running `algorithm_comparison.py`.

### Reproducing the additional analyses

The following commands assume their corresponding scripts have been added to the project root. They are research/analysis utilities; some were developed after the initial GitHub version, so check that the files and input CSVs exist locally before running.

```powershell
.\.venv\Scripts\python.exe best_found_delta_analysis.py
.\.venv\Scripts\python.exe best_so_far_tracking.py
.\.venv\Scripts\python.exe check_team_feature_sensitivity.py
.\.venv\Scripts\python.exe controlled_team_ablation.py --episodes 20 --eval-seeds 30
.\.venv\Scripts\python.exe scalability_benchmark.py --seeds 30
.\.venv\Scripts\python.exe marl_generalization_benchmark.py --seeds 30
```

Training and benchmarking are computational experiments and may require a functioning pretrained checkpoint in `models_v4/` or `models_v4_dynamic/`. Checkpoint files may be too large for ordinary GitHub pushes; verify repository contents and file-size limits before attempting to upload them.

## Repository organization

```text
src/                       Simulation, robots, tasks, planning, allocation, MARL components
experiments/               Training, benchmark, evaluation and diagnostics scripts
models_v4/                 Pretrained V4 model checkpoints (when present)
models_v4_dynamic/         Dynamic fine-tuned checkpoints (when present)
results/                   CSV benchmark outputs and analysis reports
gui.py                     Interactive simulation
algorithm_comparison.py    Benchmark comparison visualizations
final_project_check.py     Project smoke checks
README.md                  Project description and reproduction guide
```

Additional local experiment scripts produce separate outputs under `results/best_found_analysis/`, `results/best_so_far_tracking/`, `results/team_feature_sensitivity/`, `results/controlled_team_ablation/`, `results/scalability/`, and `results/marl_generalization/`.

## Limitations and future work

- Results describe the tested scenarios; optimality of full dynamic multi-robot schedules is **not formally proven**.
- The Double DQN policy learns **task allocation**, while A* and simulation rules handle paths and collision prevention.
- The final dynamic training reward does **not** directly integrate the explicit collision penalty defined in `MARLEnvironmentV2`.
- Some learning experiments use a shared pretrained checkpoint and one fine-tuning seed; more independent training runs would strengthen statistical conclusions.
- Larger maps and teams require more extensive testing; grid sizes also change the workload geometry.

## Academic context

Developed as a **Design and Analysis of Algorithms** university project, focusing on dynamic path planning, assignment strategies, multi-agent reinforcement learning, reproducible simulation, and empirical evaluation.
