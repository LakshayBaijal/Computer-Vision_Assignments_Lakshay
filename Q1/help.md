Run 1 — AABB baseline HP1
use_angle: false
epochs: 10
Purpose: required for Q1.1 + comparison baseline.
Run 2 — AABB baseline HP2
use_angle: false
epochs: 10
Change ONE hyperparam vs Run 1 (e.g. LR).
Purpose: Q1.1 says visualizations for at least 2 hyperparameter settings.
Run 3 — OBB direct regression
use_angle: true
epochs: 15 (already done, reuse checkpoint)
Purpose: first angle approach in Q1.2.
Run 4 — OBB multi-bin A
use_angle: true + your multi-bin mode enabled
bin size example: 30°
epochs: 10
Purpose: second angle approach required in Q1.2.
Run 5 — OBB multi-bin B
same as Run 4 but bin size example: 60°
epochs: 10
Purpose: required comparison of bin sizes.