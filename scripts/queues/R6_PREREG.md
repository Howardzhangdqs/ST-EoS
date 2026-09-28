# R6 pre-registration — batch-size × momentum regime-transition sweep

Written 2026-09-05, **before any sweep cell completed** (only the anchor
cell, reusing round-3/5 runs, had been observed). Purpose: fix the analysis
conventions and the falsifiable predictions before data lands, so the
eventual Finding 46 / paper paragraph is a pre-specified test, not a
post-hoc reading.

## Design (fixed)

- Grid: B ∈ {2048, 512, 128, 32} × μ ∈ {0, 0.9}; arms {sgd-sensing
  (reference+measurement), rankkx d1 (rank-1 damping, k=1), amplify
  α=0.5 (rank-1 amplification)}; n=3 seeds (0,1,2).
- Protocol: canonical 5000 steps, lr=0.1, wd=5e-4, deterministic CUDA,
  ARPACK outcome (fixed **512-sample** eval batch, tol 1e-6, k=3) every
  200 steps, `--multibatch` (m=4 fresh **training-batch-size** batches for
  the batch sharpness), plant sensing on the same fixed eval batch,
  refresh N=20. lr is NOT rescaled with B (single-knob isolation of
  stochasticity); the plant's λ_targ stays 2/η=20 at every μ (same physical
  controller; the μ-dependence of the operative boundary is part of what is
  measured, not retuned away).
- Anchor cell (128, 0.9) = the published round-3/5 runs, NOT rerun.

## Analysis conventions (fixed)

- Late window: step ≥ 2000 (same as rounds 3–5).
- Levels: mean ± 95% t-CI over seeds. Paired arm-vs-SGD deltas on shared
  seeds: mean ± 90% t-CI (same convention as analyze_bsharp_arms.py).
- Language calibration: n=3 cells support "no measurable reduction at this
  power", never "equivalent/inert proved"; "significant" only for CIs
  excluding zero. TOST only if a formal equivalence claim is needed
  (margin 0.25·|SGD−SAM| at the anchor; not expected to be central here).
- Gradient-energy share: from plant jsonl, mean over late window of
  Σᵢcos²(g, v̂ᵢ) (d1: cos_g_v; amp: (gdotv/gnorm)²).
- The four phase quantities (reviewer's figure): (a) gradient share vs B;
  (b) paired Δλ (d1−sgd, amp−sgd) vs B; (c) batch sharpness vs B with the
  EoSS plateaus 2(1∓μ)/η; (d) reference λ vs B.

## Pre-specified predictions (falsifiable)

- **P1 (authority)**: the gradient-energy share of the edited top-1
  subspace rises monotonically with B (from ~2×10⁻⁴ at B=128 toward the
  full-batch GD alignment regime); stochasticity, not μ, is the dominant
  axis (μ=0 and μ=0.9 shares similar at fixed B).
- **P2 (sign-control collapse)**: paired Δλ of both d1 and amp stay within
  noise at B ≤ 512; IF authority grows enough at B=2048, d1 moves λ
  sharp-ward (toward 2d/η) and amp flat-ward — the transition the round-6
  review asked us to locate. Absence of any transition by B=2048 is itself
  the headline (authority needs near-full batch).
- **P3 (EoSS family)**: at μ=0.9 the batch sharpness interpolates between
  the plateaus — rising toward 2(1+μ)/η=38 as B grows (2048 closest),
  falling toward 2(1−μ)/η=2 as B shrinks; at μ=0 BOTH plateaus coincide at
  2/η=20, so the batch sharpness stays ≈20 at every B. This is the sharpest
  falsifiable sub-prediction (flat line at μ=0).
- **P4 (reference λ)**: at μ=0 the fixed-batch reference λ sits near 2/η=20
  across B (GD-like boundary); at μ=0.9 it stays elevated (~29, the
  published anchor level) and does not approach the small-batch plateau —
  the reference-λ and batch-sharpness quantities decouple differently
  across B.

## Known limitations (disclosed up front)

- n=3 per cell; B=32 at lr=0.1 is off the linear-scaling rule (kept
  deliberately; divergence would be reported as such, not retuned).
- The plant λ_targ=20 is below the heavy-ball boundary 38 at μ=0.9, so the
  d1 arm fires harder at μ=0.9 than at μ=0 by construction (fired_frac is
  logged per cell to quantify this).
- Full-batch GD anchor on ResNet-18/CIFAR-100 is computationally out of
  reach (bs=50000); the deterministic anchor remains the MLP testbed, and
  B=2048 is the largest quasi-deterministic point.

## Addendum (2026-09-05, after the FIRST bs=2048 sensing run's trajectory
shape was visible, BEFORE any actuator-arm outcome at any B): window
sensitivity for the fixed-step confound

Fixed steps across B confounds epochs (5000 steps = 3.2 ep at B=32, 12.8 ep
at B=128, 51.2 ep at B=512, 204.8 ep at B=2048 — the original draft of this
addendum wrongly called B=512 epoch-matched; only B=128 is). The first
bs=2048/μ=0.9 sensing run memorizes its training batches by step ~1600
(fixed-batch train loss ~0.001) and the reference λ collapses to ~8.
Pre-specified handling: for every cell report the preregistered late window
(step ≥ 2000) AND a mid window (200 ≤ step < 2000); B=2048 figure points
use the mid window with the late-window values shown alongside; the
memorization caveat is disclosed in the paper.

## Addendum 2 (2026-09-06, after the 512 cells landed, BEFORE any 2048
actuator outcome): phase-matched comparison across (B, μ)

Partial-data correction and refinement of Addendum 1:
1. B=512/μ=0.9 ALSO memorizes by the late window (51.2 ep; late ref λ
   1.99±0.31, batch sharpness 0.95 — a converged state, gradients ~0). The
   mid window (epochs 2–20) is the training regime there too. Figure points
   for ALL B ≥ 512 use the mid window; B ≤ 128 keeps the late window
   (12.8/3.2 ep — still training at 5000 steps).
2. Momentum shifts convergence speed by ~(1−μ)⁻¹: at equal steps, μ=0 runs
   are ~10× "younger" in optimization progress than μ=0.9. Cross-μ
   comparisons are therefore made WITHIN matched phase (training vs
   converged), never at matched step count alone; per-cell trajectories are
   shown so readers can see each cell's phase.
3. Pre-data observation registered for the decisive upcoming test: through
   B=512/μ=0.9 the gradient share of the edited top-1 subspace stays flat
   (~1.9e-4 late / 7e-4 mid at B=512 vs 2.1e-4 / 1.25e-3 at B=128 — no
   growth with B), and both actuator signs show no measurable mid-window
   effect (d1 +12.1 [−15.5,+39.7]; amp −2.0 [−42,+38]). If the same holds
   at B=2048, the conclusion is that per-step top-eigenspace authority does
   NOT return anywhere in the reachable stochastic range — the
   deterministic↔stochastic transition in sign-control is complete already
   at B≤512, i.e. "authority needs near-full batch" is the conservative
   reading, and the phase figure's panel (a) is flat rather than
   monotone-in-B. This is registered BEFORE the 2048 actuator arms ran.

## Addendum 3 (2026-09-06 23:20, after d1@2048 s0/s1 landed, BEFORE any
## amp05@2048 outcome): the damping direction at the quasi-deterministic end

Interim decisive readout (mid window, paired vs per-seed SGD): rank-1
damping at bs=2048/μ=0.9 moves the reference λ SHARP-WARD by +42.0 (s0) and
+45.4 (s1) — versus ±3 (inert, cross-seed noise) at bs=128 — while the
edited-subspace gradient share stays near-flat (2.1e-4 @128 → 1.9e-4 @512
→ 4.3–5.7e-4 @2048, mid windows). Note this partially UPDATES Addendum 2's
flat-authority reading: authority, measured by OUTCOME, returns at 2048
even though the share barely moves. Registered interpretation fork for the
amp05@2048 arms (running next):
- if amp05@2048 moves flat-ward (paired Δ < 0): FULL sign-revival — both
  signs move λ at 2048 in the directions the 2-D reduction predicts
  (damping ↑, amplification ↓): the deterministic sign-controlled mechanism
  re-emerges between B=512 and B=2048;
- if amp05@2048 is ~0 or sharp-ward: ASYMMETRIC revival — damping-only
  sensitivity at large B, which would argue the return of authority is
  partial and sign-dependent.
Either outcome completes the review's requested regime-transition
characterization. (d1_s2 was lost to a transient cudaErrorLaunchTimeout —
non-scientific, sibling seeds passed immediately before — and will be
re-run; conclusions rest on the completed seeds with n disclosed.)

## Addendum 4 (2026-09-07 05:30, fork resolution AFTER amp05@2048 landed)

The Addendum-3 fork resolves as ASYMMETRIC REVIVAL: d1 paired deltas
+42.0/+45.4 (mean +43.7, n=2) — consistently sharp-ward, the 2-D
reduction's damping direction; amp05 paired deltas +14.5/+23.6/−26.8
(mean +3.8, n=3) — scattered around zero, no measurable flat-ward effect
at this power, against per-seed scatter of ±25. Gradient shares at 2048:
d1 4.3–5.7e-4, amp 3.2–3.7e-4 (mid windows). Reading (post-hoc,
disclosed as such): at the quasi-deterministic end the local edit regains
real per-step purchase on the reference λ, but only in the sharp-ward
direction; the flat-ward direction remains unavailable to single-direction
edits at every tested batch size. Combined with the earlier cells, the
review's regime-transition question has a two-part answer: (i) per-step
top-eigenspace authority in the GRADIENT-ENERGY sense stays ~2–6e-4 across
the whole B range (no monotone return), yet (ii) OUTCOME authority revives
asymmetrically at B=2048. The 2048/μ=0 cell (running) will show whether
the asymmetry is momentum-dependent.
