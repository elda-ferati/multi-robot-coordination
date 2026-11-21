MULTI-ROBOT COORDINATION — BEST-FOUND + DELTA ANALYSIS
==================================================================
Input: final_marl_dynamic_benchmark.csv
Algorithms compared: 6 (Auction, Dynamic 10, Dynamic 20, Hungarian V3, Hungarian V4, MARL Original)
Unique seeds in input: 100
Runs that did not complete successfully / had collisions: 0

DEFINITIONS (each seed, each metric, separately)
best_found = minimum observed cost over compared algorithms
delta = algorithm_cost - best_found  [lower is better]
gap_percent = 100 * delta / abs(best_found) (when best_found != 0)
Best/tied count credits ALL algorithms tied at minimum; sums may exceed seed count.
Only seeds where every algorithm succeeded and has a usable value are included.
Best-found is an empirical reference, NOT a proven global optimum.
Metrics are independent: do not combine winners across metrics.

TOTAL COMPLETION STEPS (finish)
Comparable seeds: 100; excluded: 0
Algorithm                 Mean cost    Best/ties   Mean delta   Mean gap %
Hungarian V4                211.940           28        9.820        4.909
Dynamic 20                  212.440           19       10.320        5.098
Auction                     213.690           23       11.570        5.794
Hungarian V3                213.940           15       11.820        5.920
Dynamic 10                  214.350           20       12.230        6.117
MARL Original               216.410           14       14.290        7.128

TOTAL ROBOT DISTANCE (distance)
Comparable seeds: 100; excluded: 0
Algorithm                 Mean cost    Best/ties   Mean delta   Mean gap %
Dynamic 20                  903.230           27       26.460        3.040
Hungarian V4                904.660           26       27.890        3.214
Auction                     907.860           19       31.090        3.609
Hungarian V3                907.890           18       31.120        3.585
Dynamic 10                  909.790           14       33.020        3.775
MARL Original               917.870           13       41.100        4.744

MEAN NORMAL-TASK COMPLETION STEPS (p1_avg)
Comparable seeds: 100; excluded: 0
Algorithm                 Mean cost    Best/ties   Mean delta   Mean gap %
Dynamic 20                  103.946           55        2.238        2.147
Dynamic 10                  106.904           16        5.196        5.162
MARL Original               108.632            5        6.924        7.005
Auction                     109.410           13        7.703        7.787
Hungarian V3                109.413           14        7.705        7.783
Hungarian V4                112.433            8       10.725       10.721

MEAN URGENT-TASK COMPLETION STEPS (p3_avg)
Comparable seeds: 100; excluded: 0
Algorithm                 Mean cost    Best/ties   Mean delta   Mean gap %
Hungarian V4                 38.337           76        0.481        1.373
Hungarian V3                 41.824           17        3.968       10.795
Auction                      41.922           12        4.066       11.222
MARL Original                43.076            9        5.219       14.098
Dynamic 10                   43.094            3        5.238       14.104
Dynamic 20                   44.366            4        6.509       17.518

INTERPRETATION:
Mean gap is the average per-seed relative deviation from best found.
This differs from comparing overall averages across seeds.
Gap and delta are not statistical significance tests.
Because workloads are dynamic, a best-known assignment may not be globally optimal.
For an optimality proof, an exact solver or a mathematically justified lower bound is required.
