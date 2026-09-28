"""Cross-validate the in-run power-iteration λ_max estimator against ARPACK.

For each MLP-path run behind the "λ_max_final ≈ 2/η (0.48%)" headline, reload
the final snapshot θ, rebuild model+data from manifest.json, pass the
full-batch loss sanity gate against metrics.csv, then measure λ_max(θ) three
ways:

  1. lam_recorded — the in-run warm-started power-iter value from
                    sharpness.csv (the number the headline is built on);
  2. lam_ws_rerun — a faithful re-run of the SAME estimator at θ, warm-started
                    from the SECOND-TO-LAST v_top in vtops.npz (the exact
                    starting point of the last in-run measurement);
  3. lam_arpack   — ARPACK ground truth (scipy eigsh on the Pearlmutter HVP,
                    steos.sharpness.hessian.max_eigenvalue_arpack).

Timeline semantics (steos/train.py): sharpness@step s and metrics@step s are
measured at the PRE-update θ_s, while snapshots@step s stores the POST-update
θ_{s+1}. We therefore load the snapshot whose step is closest to the LAST
sharpness step (sweep runs: snapshot@3950 — nominally aligned, actually one GD
update past the measurement point; fine runs: 5-step mismatch, snapshot@2990
vs sharp@2995). The loss gate compares the rebuilt θ against the metrics row
at step = snapshot_step + 1 (the same θ); if that row is missing it falls back
to step = snapshot_step and flags it.

Data seed: the manifest data block carries no explicit seed; steos.train also
calls make_dataset without one, so the default seed=0 is used (recorded per
row as data_seed=0).

Usage:
    .venv/bin/python scripts/analysis/verify_lambda_max_arpack.py            # all runs
    .venv/bin/python scripts/analysis/verify_lambda_max_arpack.py --runs fine --runs width_32
        [--device cuda] [--output runs/crosscheck/lambda_max_crosscheck.csv]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from steos.data import loss_fn, make_dataset
from steos.models import build_model
from steos.sharpness.hessian import (
    _param_list,
    hvp,
    max_eigenvalue,
    max_eigenvalue_arpack,
)

RUN_ROOT = Path("runs")

# NOTE: the task brief says "11 runs" but the brace expansion lists 12
# directories (4 difficulty + 4 width + 4 fine); all exist and are complete,
# so all 12 are processed by default and the discrepancy is reported.
DEFAULT_RUNS = [
    "sweep_difficulty/sweep_difficulty_0.1",
    "sweep_difficulty/sweep_difficulty_0.4",
    "sweep_difficulty/sweep_difficulty_0.8",
    "sweep_difficulty/sweep_difficulty_1.2",
    "sweep_width/sweep_width_32",
    "sweep_width/sweep_width_64",
    "sweep_width/sweep_width_128",
    "sweep_width/sweep_width_256",
    "mlp_ts_eos_fine",
    "mlp_ts_eos_fine_0.01",
    "mlp_ts_eos_fine_0.02",
    "mlp_ts_eos_fine_0.05",
]

COLS = [
    "run", "snapshot_step", "sharp_step", "step_mismatch", "gate_metrics_step",
    "loss_gate_relerr", "data_seed", "lr", "momentum", "n_params",
    "lam_recorded", "lam_ws_rerun", "lam_arpack", "arpack_ritz_residual",
    "ws_ritz_residual", "vtop_init_step",
    "relbias_recorded_vs_arpack", "relbias_ws_vs_arpack",
    "ratio_2_over_eta_recorded", "ratio_2_over_eta_arpack",
    "skipped_reason",
]


def _unflatten(flat: np.ndarray, params: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split a flat vector into per-param tensors following _param_list order."""
    out, s = [], 0
    for p in params:
        e = s + p.numel()
        out.append(torch.from_numpy(np.ascontiguousarray(flat[s:e])).to(
            p.device, p.dtype).view(p.shape))
        s = e
    assert s == flat.shape[0], f"flat size {flat.shape[0]} != param total {s}"
    return out


def _ritz_residual(model, loss_f, X, y, params, v_list, lam) -> float:
    """Relative Ritz residual ‖Hv − λ̂v‖ / max(|λ̂|, 1e-12) for a unit-norm v."""
    Hv = hvp(model, loss_f, X, y, v_list, params)
    num2 = sum(float(((h - lam * vi) ** 2).sum().item()) for h, vi in zip(Hv, v_list))
    return math.sqrt(num2) / max(abs(lam), 1e-12)


def process_run(rel: str, device: str, gate_tol: float) -> dict:
    """Run the full cross-check for one run directory; returns a CSV row."""
    row: dict = {c: float("nan") for c in COLS}
    row["run"] = rel
    row["skipped_reason"] = ""
    run_dir = RUN_ROOT / rel

    # ---- 1. manifest ------------------------------------------------------
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
    except FileNotFoundError:
        row["skipped_reason"] = "manifest.json missing"
        return row
    cfg = manifest["config"]

    # LayerNorm MLPs have different HVP semantics — protocol says skip.
    if cfg["model"].get("layer_norm", False):
        row["skipped_reason"] = "layer_norm=true (HVP semantics differ)"
        return row
    ocfg = cfg["optimizer"]
    row["lr"] = float(ocfg["lr"])
    row["momentum"] = float(ocfg.get("momentum", 0.0))

    # ---- 2. rebuild model + data ------------------------------------------
    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name"), **mcfg).to(device)
    model.eval()  # MLP: no BN/dropout, eval == train; eval per ARPACK contract
    params = _param_list(model)
    row["n_params"] = int(sum(p.numel() for p in params))

    dcfg = dict(cfg["data"])
    data_seed = int(dcfg.pop("seed", 0))  # manifest carries none → make_dataset default
    row["data_seed"] = data_seed
    X, y, task = make_dataset(dcfg.pop("name"), device=device, seed=data_seed, **dcfg)
    loss_f = loss_fn(task)

    # ---- 3. snapshot / sharpness alignment --------------------------------
    try:
        snap = np.load(run_dir / "snapshots.npz", allow_pickle=True)
        sharp = pd.read_csv(run_dir / "sharpness.csv")
        metrics = pd.read_csv(run_dir / "metrics.csv")
        vt = np.load(run_dir / "vtops.npz")
    except FileNotFoundError as exc:
        row["skipped_reason"] = f"missing artifact: {exc.filename}"
        return row

    sharp_step = int(sharp["step"].iloc[-1])
    row["sharp_step"] = sharp_step
    row["lam_recorded"] = float(sharp["lambda_max"].iloc[-1])

    snap_steps = snap["steps"].astype(int)
    idx = int(np.argmin(np.abs(snap_steps - sharp_step)))
    snapshot_step = int(snap_steps[idx])
    row["snapshot_step"] = snapshot_step
    row["step_mismatch"] = snapshot_step - sharp_step  # signed

    # load θ; verify the npz param order matches this model's parameters()
    names = [str(n) for n in snap["param_names"]]
    cur = [n for n, _ in model.named_parameters()]
    if names != cur:
        row["skipped_reason"] = f"param_names mismatch: {names} != {cur}"
        return row
    with torch.no_grad():
        # W is ref-centered (θ − θ_ref, θ_ref = first snapshot frame) — add it back
        theta_flat = snap["W"][idx].astype(np.float32)
        if "ref" in snap and snap["ref"].size:
            theta_flat = theta_flat + snap["ref"].astype(np.float32)
        v_list = _unflatten(theta_flat, params)
        for p, vi in zip(params, v_list):
            p.copy_(vi)

    # ---- 4. sanity gate: rebuilt full-batch loss vs metrics.csv -----------
    # snapshots@s is the POST-update θ_{s+1} ⇒ compare with metrics step s+1.
    gate_step = snapshot_step + 1
    mrow = metrics.loc[metrics["step"] == gate_step]
    if mrow.empty:  # fall back to the pre-update row (same step number)
        gate_step = snapshot_step
        mrow = metrics.loc[metrics["step"] == gate_step]
    row["gate_metrics_step"] = gate_step
    if mrow.empty:
        row["skipped_reason"] = "metrics.csv has no row at the gate step"
        return row
    loss_ref = float(mrow["loss"].iloc[0])
    with torch.no_grad():
        loss_re = float(loss_f(model(X), y))
    relerr = abs(loss_re - loss_ref) / max(abs(loss_ref), 1e-12)
    row["loss_gate_relerr"] = relerr
    if relerr > gate_tol:
        row["skipped_reason"] = (
            f"loss gate failed: relerr {relerr:.2e} > {gate_tol:.0e} "
            "(likely data-seed mismatch)"
        )
        return row

    # ---- 5. ARPACK reference ----------------------------------------------
    try:
        evals, evecs = max_eigenvalue_arpack(
            model, loss_f, X, y, params, k=3, tol=1e-6, v0=None, seed=0,
        )
        lam_arpack = float(evals[0])  # eigsh columns are already unit-norm
        v_arpack = _unflatten(evecs[:, 0].astype(np.float32), params)
        row["lam_arpack"] = lam_arpack
        row["arpack_ritz_residual"] = _ritz_residual(
            model, loss_f, X, y, params, v_arpack, lam_arpack)
    except Exception as exc:  # noqa: BLE001 — record and continue with NaN
        row["skipped_reason"] = f"ARPACK failed: {type(exc).__name__}: {exc}"

    # ---- 6. faithful re-run of the in-run warm-started power-iter ---------
    shcfg = cfg.get("sharpness", {})
    n_iter = int(shcfg.get("n_iter", 200))
    tol = float(shcfg.get("tol", 1e-4))
    vt_steps = vt["steps"].astype(int)
    # second-to-last v_top = warm-start point of the LAST in-run measurement
    v0_flat = vt["V"][-2].astype(np.float32)
    row["vtop_init_step"] = int(vt_steps[-2])
    v_init = _unflatten(v0_flat, params)
    lam_ws, v_ws = max_eigenvalue(
        model, loss_f, X, y, n_iter=n_iter, tol=tol,
        return_vec=True, v_init=v_init,
    )
    row["lam_ws_rerun"] = float(lam_ws)
    row["ws_ritz_residual"] = _ritz_residual(
        model, loss_f, X, y, params, v_ws, float(lam_ws))

    # ---- 7. derived quantities ---------------------------------------------
    if not math.isnan(row["lam_arpack"]):
        lam_ref = float(row["lam_arpack"])
        row["relbias_recorded_vs_arpack"] = row["lam_recorded"] / lam_ref - 1.0
        row["relbias_ws_vs_arpack"] = float(lam_ws) / lam_ref - 1.0
        row["ratio_2_over_eta_arpack"] = lam_ref * row["lr"] / 2.0
    row["ratio_2_over_eta_recorded"] = row["lam_recorded"] * row["lr"] / 2.0
    return row


def summarize(df: pd.DataFrame) -> None:
    ok = df[df["skipped_reason"] == ""].copy()
    pct = lambda s: (100 * s).map(lambda v: f"{v:+.3f}%")

    print("\n=== per-run results ===")
    show = df[["run", "snapshot_step", "sharp_step", "step_mismatch",
               "loss_gate_relerr", "lam_recorded", "lam_ws_rerun", "lam_arpack",
               "relbias_recorded_vs_arpack", "relbias_ws_vs_arpack",
               "ratio_2_over_eta_recorded", "ratio_2_over_eta_arpack",
               "skipped_reason"]].copy()
    for c in ["relbias_recorded_vs_arpack", "relbias_ws_vs_arpack"]:
        show[c] = pct(show[c].astype(float))
    for c in ["ratio_2_over_eta_recorded", "ratio_2_over_eta_arpack"]:
        show[c] = show[c].astype(float).map(lambda v: f"{v:.5f}")
    print(show.to_string(index=False))

    if ok.empty:
        print("\n[warn] no runs passed the gates — no summary statistics")
        return

    def stats(col, center):
        v = ok[col].astype(float)
        dev = (v - center).abs().max()
        return (v.mean(), v.std(ddof=1), v.min(), v.max(), dev)

    for col, center, label in [
        ("relbias_recorded_vs_arpack", 0.0, "power-iter bias (recorded vs ARPACK)"),
        ("relbias_ws_vs_arpack", 0.0, "warm-start rerun bias (ws_rerun vs ARPACK)"),
        ("ratio_2_over_eta_recorded", 1.0, "λ_max·η/2 recorded"),
        ("ratio_2_over_eta_arpack", 1.0, "λ_max·η/2 ARPACK-corrected"),
    ]:
        m, s, lo, hi, am = stats(col, center)
        print(f"\n{label}: mean {m:+.5f}  std {s:.5f}  min {lo:+.5f}  max {hi:+.5f}"
              f"  max|dev| {am:.5f}   (n={len(ok)})")

    print("\nRitz residuals (relative ‖Hv−λ̂v‖/λ̂): "
          f"ARPACK max {ok['arpack_ritz_residual'].astype(float).max():.2e}, "
          f"warm-start max {ok['ws_ritz_residual'].astype(float).max():.2e}")


def main():
    ap = argparse.ArgumentParser(
        description="ARPACK cross-check of the in-run λ_max power-iter estimator")
    ap.add_argument("--runs", action="append", default=None,
                    help="substring filter on run ids (repeatable); default: all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="runs/crosscheck/lambda_max_crosscheck.csv")
    ap.add_argument("--gate-tol", type=float, default=1e-5,
                    help="loss-gate relative-error threshold (default 1e-5)")
    args = ap.parse_args()

    runs = DEFAULT_RUNS
    if args.runs:
        runs = [r for r in DEFAULT_RUNS if any(f in r for f in args.runs)]
        missing = [f for f in args.runs if not any(f in r for r in runs)]
        if missing:
            print(f"[warn] filters matched nothing: {missing}")

    torch.manual_seed(0)
    rows = []
    for rel in runs:
        print(f"\n--- {rel} ---", flush=True)
        rows.append(process_run(rel, args.device, args.gate_tol))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(out, index=False)
    print(f"\n[done] wrote {out}")
    summarize(df)


if __name__ == "__main__":
    main()
