"""ST-EoS: Sharpness / curvature and weight-trajectory co-analysis.

Core research target (Gap A): jointly visualize the sharpness time-series
(Hessian max eigenvalue lambda_max or critical sharpness lambda_c) with the
low-dimensional projection of the weight trajectory, revealing whether the
Edge-of-Stability phase moves along a narrow flat corridor and how that
corridor changes with learning rate / depth / data difficulty.

Subpackages:
  models    - architecture registry (MLP first; Transformer/CNN later)
  data      - synthetic + real dataset factories (MNIST via domestic mirror)
  optim     - custom SGD/AdamW exposing _compute_update (Meta lambda_c contract)
  monitor   - cheap per-step metrics + periodic weight snapshots
  sharpness - lambda_max (power iteration / HVP) and lambda_c (bracketing)
  analysis  - trajectory PCA/UMAP, delta-theta geometry, joint visualization
"""

__version__ = "0.0.1"
