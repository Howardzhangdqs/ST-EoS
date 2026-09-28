# ST-EoS — Sharpness / Curvature × Weight-Trajectory Co-analysis

Research code for studying the **Edge of Stability (EoS)** phenomenon by jointly
analyzing **loss-landscape curvature** (Hessian top eigenvalue λ_max and the
scalable critical sharpness λ_c) and the **low-dimensional geometry of the GD
weight trajectory**. Core research target (Gap A): visualize and quantify how
the sharpness time-series and the weight trajectory co-evolve, and how the EoS
"corridor" changes with learning rate, data difficulty, and architecture.

Single-GPU friendly (developed on one RTX 4080 / 16 GB). All experiments run in
seconds-to-minutes on small MLPs — the regime where EoS is cleanest.

## Headline results (see `FINDINGS_zh.md` for the full log)

1. **λ_max reproduces EoS** — climbs to 2/η and oscillates there (Cohen 2021).
2. **λ_c validated** as the gradient-direction curvature gᵀHg/gᵀg to <1 % on
   pure quadratics; equals λ_max only at gradient/top-eigenvector alignment.
3. **The GD trajectory is low-dimensional** — PC1 captures 84–98 % of variance.
4. **|cos(Δθ, v_top)| → 0.998 at EoS** — the Damian et al. (2023)
   self-stabilization signature (must use the *single-step* Δθ; multi-step
   snapshot diffs cancel the period-2 EoS oscillation and mislead).
5. **2/η-threshold tracking is universal**: λ_max_final/(2/η) = 1.0004
   (mean ratio over 11 stable runs; max deviation 0.48 %) across a 20× LR
   range, 12× data-difficulty range, and 8× width range. ARPACK
   cross-validation (F32): estimator bias ≤ 3e-7 (negligible); calibrated
   ratio 0.9993 ± 0.0027, max deviation 0.46 % — the residual spread is
   λ_max(t) oscillation about 2/η, not measurement error.
6. **Corridor dimensionality scales with capacity**: PC1 drops as η, width, or
   difficulty grows.
7. **Divergence at η ≥ 0.1** (2/η ≤ 10) — route-to-instability.
8. **Output-norm proxy (Gap B)**: every layer's ‖W‖ tracks λ_max (Pearson
   0.74–0.82), but the *output* layer is **not** uniquely diagnostic — the
   **last hidden** layer is the best single-layer proxy (refines Li et al.).
9. **No common multi-seed skeleton (Gap E)**: each EoS trajectory is ~94 % 1-D,
   but its direction is init-specific (near-orthogonal across seeds; pooled
   PCA shows ~4 equal PCs). Song & Yun alignment does **not** extend to this
   deeper tanh MLP — a clean negative result.
10. **Mechanism C optimizer (‖W⟩ proxy) — weak control authority**: not
    competitive vs SGD/AdamW/SAM; the proxy is a good predictor but a weak
    actuator (Finding 12).
11. **Generalization to CIFAR-10 + PreAct-ResNet(BN)**: trajectory
    low-dimensionality *survives* (PC1 76 %), but the ‖W⟩ sharpness proxy
    *breaks* under BatchNorm — corr(‖W⟩, λ_max) flips from +0.8 (MLP) to −0.8
    (BN convnet), explaining why the norm-cap optimizer can't work on BN nets.
12. **Under BatchNorm, the BN scale γ is the sharpness proxy** (detrended
    +0.56); under LayerNorm convnets (ConvNeXt) the LN-γ proxy generalizes
    (+0.57). For attention (ViT) no single proxy dominates.
13. **Generalizes to timm ViT + ConvNeXt on CIFAR-10/100**: low-dim trajectories
    hold across all 6 runs (PC1 77–81 %); ViT shows sustained EoS-like sharpness
    while ConvNeXt's λ_max collapses to ~1.

## Quick start

```bash
# venv that INHERITS the system torch (no 2.5 GB re-download).
# The .pth file points at an existing torch install; adjust the path if needed.
python3 -m venv .venv
echo "/path/to/existing/torch/site-packages" > \
  "$(./.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')/zzz_inherit.pth"
.venv/bin/python -c "import torch, numpy, matplotlib, sklearn"   # sanity

# Milestone-0: train MLP + record everything
.venv/bin/python scripts/experiments/train_mlp_synthetic.py --config configs/mlp_synthetic.yaml

# Phase-2: reproduce EoS with curvature measurement
.venv/bin/python scripts/experiments/train_mlp_synthetic.py --config configs/mlp_synthetic_eos.yaml
.venv/bin/python -m steos.analysis.plot_eos --run mlp_ts_eos

# Phase-3 (Gap A): trajectory PCA + joint visualization
.venv/bin/python scripts/analysis/analyze_trajectory.py --run mlp_ts_eos --color-by lambda_max

# Curvature correctness gate (must pass before trusting any number)
.venv/bin/python scripts/analysis/verify_curvature.py

# Sweeps
.venv/bin/python scripts/experiments/sweep.py --config configs/mlp_synthetic_eos.yaml \
    --axis optimizer.lr --values 0.01,0.02,0.05,0.1,0.2 --steps 5000
.venv/bin/python scripts/analysis/compare_sweep.py --sweep-dir runs/sweep_optimizer_lr
```

Formal dependencies are declared in `pyproject.toml` (`uv.lock` is tracked);
the `.pth` inheritance above is a local disk-saving shortcut.

## Package layout

```
steos/          core package
  models/       MLP (top-level named layers for per-layer attribution), ResNet/ViT/ConvNeXt wrappers + registry
  data/         synthetic factories (teacher-student regression, Gaussian cls) + CIFAR
  optim/        custom SGD exposing _compute_update (Meta λ_c contract), SAM/LinSAM/xSAM variants, rank-k plants
  monitor/      per-step metrics (loss/grad/‖Δθ‖/per-layer norms) + snapshots
  sharpness/    λ_max (power iteration, Pearlmutter HVP) + λ_c (bracketing)
  analysis/     trajectory PCA/UMAP, joint visualization (Gap A), sweep comparison
  train.py      architecture-agnostic trainer; curvature measured at θ_t (pre-step)
configs/        YAML configs (plain, EoS)
scripts/
  experiments/  training & experiment entry points (train_*, *_experiment, sweeps)
  analysis/     analyze_*/verify_*/validate_* — run-data analysis & correctness gates
  plots/        plot_* figure generation
  queues/       run_*.sh experiment queues (resume-safe) + wait_*.sh progress watchers
  notify_report.py  email progress reports (local ops tooling)
tests/          pytest suite (curvature correctness gate)
paper/          ICLR 2027 manuscript (main.tex) + 中文精读 interpretation_zh/
figures/        curated result figures (tracked); 200-dpi PNG mirrors in paper_figures_png/
docs/reviews/   revision-round review / verdict working documents (process archive)
third_party/    Meta scalable-curvature reference (measure_critical_lr, optimizers)
FINDINGS_zh.md       dated log of empirical discoveries
REVISION2_STATUS.md  revision-2 experiment queue status
```

Experiment outputs land in `runs/` (gitignored, regenerable) and are written
only at run end (`metrics.csv` + `manifest.json`).

## How curvature is measured

- **λ_max** — power iteration on the loss Hessian; Hessian-vector products via
  Pearlmutter's trick `Hv = ∇(g·v)` (double-backprop). ~100–300 HVPs.
- **λ_c** — critical sharpness (Kalra et al.): η_c = min{η : L(θ−ηΔθ) > L(θ)},
  λ_c = 2/η_c, found by exponential bracketing + bisection along the update
  direction using only forward passes (~5–8 each). Requires the custom SGD's
  `_compute_update` so a "virtual step" at arbitrary LR can be evaluated
  without mutating optimizer state.

For GD on a local quadratic, η_c = 2·gᵀg / gᵀHg, so **λ_c = gᵀHg/gᵀg** — the
Rayleigh quotient of H along the *gradient*, equal to λ_max only when g is
aligned with v_top. See `scripts/analysis/verify_curvature.py`.

## Literature anchors

- Cohen et al., *Gradient Descent on Neural Networks Typically Occurs at the
  Edge of Stability*, ICLR 2021.
- Li et al., *Analyzing Sharpness along GD Trajectory*, NeurIPS 2022
  (output-layer weight norm as sharpness proxy).
- Damian et al., *Self-Stabilization*, ICLR 2023 (cubic-Taylor EoS dynamics).
- Song & Yun, *Trajectory Alignment / Bifurcation Theory*, NeurIPS 2023.
- Kalra et al., *Universal Sharpness Dynamics*, ICLR 2025 (route-to-chaos).
- Yoo et al., *Understanding Sharpness Dynamics — Minimalist Example*, ICML 2025
  (data-difficulty / depth / stochasticity axis).
- Kalra et al., *A Scalable Measure of Loss Landscape Curvature* (critical
  sharpness λ_c), arXiv:2601.16979 — reference code in `third_party/`.

## Status / next

Done: harness, EoS reproduction, λ_max/λ_c validation, Gap-A trajectory
analysis, LR/difficulty/width sweeps, Gap B (output-norm proxy), Gap E
(multi-seed alignment), and scale-out to CIFAR-10/100 with ResNet-18 and timm
ViT / ConvNeXt (including deterministic ARPACK re-measurement) — 31 findings
in `FINDINGS_zh.md`. Planned: multi-seed CIs for the headline numbers, a
rank-k `n_sweeps` sensitivity ablation; see the top of `FINDINGS_zh.md` for
the latest findings.
