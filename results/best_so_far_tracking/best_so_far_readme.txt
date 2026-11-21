BEST-SO-FAR ITERATIVE TRACKING — MULTI-ROBOT COORDINATION
========================================================================
Input: final_marl_dynamic_benchmark.csv
Comparison order: Auction -> Hungarian V3 -> Hungarian V4 -> MARL Original -> Dynamic 10 -> Dynamic 20
The order is chosen for analysis, NOT a verified development timeline.
Each seed and metric has its OWN evolving best (minimum cost).
Delta = candidate cost - previous best; negative Delta means improvement.
First candidate initializes best and has no Delta.
This is retrospective tracking over existing runs, NOT a new search optimizer.
Best-so-far is empirical; it does not prove a global mathematical optimum.

TOTAL COMPLETION STEPS (finish) — 100 comparable seeds
Step  Algorithm            Mean candidate  Mean best so far  New best
   1  Auction                     213.690           213.690       100
   2  Hungarian V3                213.940           211.310        18
   3  Hungarian V4                211.940           206.650        41
   4  MARL Original               216.410           205.220        23
   5  Dynamic 10                  214.350           203.720        25
   6  Dynamic 20                  212.440           202.120        17

TOTAL ROBOT DISTANCE (distance) — 100 comparable seeds
Step  Algorithm            Mean candidate  Mean best so far  New best
   1  Auction                     907.860           907.860       100
   2  Hungarian V3                907.890           901.760        22
   3  Hungarian V4                904.660           889.220        39
   4  MARL Original               917.870           885.060        22
   5  Dynamic 10                  909.790           880.710        22
   6  Dynamic 20                  903.230           876.770        25

MEAN NORMAL-TASK COMPLETION STEPS (p1_avg) — 100 comparable seeds
Step  Algorithm            Mean candidate  Mean best so far  New best
   1  Auction                     109.410           109.410       100
   2  Hungarian V3                109.413           108.591        24
   3  Hungarian V4                112.433           107.402        25
   4  MARL Original               108.632           105.289        41
   5  Dynamic 10                  106.904           103.951        44
   6  Dynamic 20                  103.946           101.708        55

MEAN URGENT-TASK COMPLETION STEPS (p3_avg) — 100 comparable seeds
Step  Algorithm            Mean candidate  Mean best so far  New best
   1  Auction                      41.922            41.922       100
   2  Hungarian V3                 41.824            41.464        25
   3  Hungarian V4                 38.337            38.129        73
   4  MARL Original                43.076            37.985         9
   5  Dynamic 10                   43.094            37.913         5
   6  Dynamic 20                   44.366            37.856         3

Generated: best_so_far_history_by_seed.csv, best_so_far_progress_summary.csv
The original benchmark and previous Best Found / Delta reports remain unchanged.
