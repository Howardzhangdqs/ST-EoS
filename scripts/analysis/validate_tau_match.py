"""Round-9 τ-interpolation endpoint contracts (sam_interp ≡ endpoints).

sam_interp_step blends g ← (1−τ)·g_lin + τ·g_SAM. The two endpoint
identities must hold on a deterministic 60-step run of the REAL training
script (identical flags, seed 0, ResNet-18):

  (i)  τ=1.0 vs the plain `sam` arm        — metrics numerically identical
  (ii) τ=0.0 vs the `linsam_bnmatched` arm — metrics numerically identical

The implementation is built so both should be BIT-identical (see
steos/optim/sam_interp.py): identical ε (sam's own accumulation order),
identical forward/BN order, exact IEEE-754 endpoint blend. Tolerance
here is generous (default 1e-6 relative) to catch semantic drift, not
float noise; bit-identity shows up as max|Δrel| = 0.

The linsam pair is additionally cross-checked on the per-step
plant_per_step.jsonl diagnostics (gnorm / lam_step / cos_g_hg — a 60-step
trajectory fingerprint), and both pairs on the manifest finals.

Usage:
    .venv/bin/python scripts/analysis/validate_tau_match.py --pair sam     # check (i)
    .venv/bin/python scripts/analysis/validate_tau_match.py --pair linsam  # check (ii)
    .venv/bin/python scripts/analysis/validate_tau_match.py                # both

Runs are resume-safe: an existing manifest.json skips re-training.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
DET = ["--deterministic", "--outcome-arpack", "--outcome-tol", "1e-6",
       "--outcome-k", "3", "--eval-batch", "512", "--chunk-size", "1024"]

PAIRS = {
    "sam": [("valtau_sam_s0", ["--opt", "sam"]),
            ("valtau_interp_t1_s0", ["--opt", "sam_interp", "--tau", "1.0"])],
    "linsam": [("valtau_linsambn_s0", ["--opt", "linsam_bnmatched"]),
               ("valtau_interp_t0_s0", ["--opt", "sam_interp", "--tau", "0.0"])],
}


def train(name: str, extra: list[str], steps: int) -> None:
    run_dir = ROOT / "runs" / name
    if (run_dir / "manifest.json").exists():
        print(f"[skip] {name} (manifest exists)")
        return
    cmd = [PY, "scripts/experiments/train_cifar100_resnet18.py", "--run-name", name,
           "--rho", "0.05", "--seed", "0", "--steps", str(steps),
           "--outcome-every", "20", *extra, *DET]
    print(f"[train] {name}: {' '.join(extra)} ({steps} steps)", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout[-3000:])
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit(f"training {name} failed rc={r.returncode}")
    tail = [l for l in r.stdout.splitlines() if l.startswith("step")][-1:]
    print(f"[done ] {name} rc=0 {time.time() - t0:.0f}s  {tail}")


def rel_diff(a: pd.Series, b: pd.Series) -> float:
    """Elementwise max relative diff; NaN==NaN counts as equal."""
    x, y = a.to_numpy(float), b.to_numpy(float)
    nan_ok = bool((np.isnan(x) == np.isnan(y)).all())
    m = ~np.isnan(x) & ~np.isnan(y)
    if not m.any():
        return 0.0 if nan_ok else float("inf")
    d = np.abs(x[m] - y[m]) / np.maximum(np.maximum(np.abs(x[m]), np.abs(y[m])), 1e-12)
    return float(d.max()) if nan_ok else float("inf")


def compare_csv(runa: str, runb: str) -> tuple[bool, float, list, int]:
    A = pd.read_csv(ROOT / "runs" / runa / "metrics.csv")
    B = pd.read_csv(ROOT / "runs" / runb / "metrics.csv")
    num = [c for c in A.columns
           if pd.api.types.is_numeric_dtype(A[c]) and c in B.columns]
    worst, bad = 0.0, []
    for c in num:
        d = rel_diff(A[c], B[c])
        worst = max(worst, d)
        if d > 0:
            bad.append((c, d))
    return worst <= 1e-6, worst, bad, len(num)


def compare_jsonl(runa: str, runb: str) -> tuple[bool, float]:
    fa = ROOT / "runs" / runa / "plant_per_step.jsonl"
    fb = ROOT / "runs" / runb / "plant_per_step.jsonl"
    if not (fa.exists() and fb.exists()):
        return True, 0.0  # nothing to compare (sam pair has no jsonl)
    A = pd.DataFrame([json.loads(l) for l in fa.read_text().splitlines()])
    B = pd.DataFrame([json.loads(l) for l in fb.read_text().splitlines()])
    worst = 0.0
    for c in A.columns:
        if c in B.columns and pd.api.types.is_numeric_dtype(A[c]):
            worst = max(worst, rel_diff(A[c], B[c]))
    return worst <= 1e-6, worst


def compare_manifest(runa: str, runb: str, keys) -> list:
    ma = json.loads((ROOT / "runs" / runa / "manifest.json").read_text())
    mb = json.loads((ROOT / "runs" / runb / "manifest.json").read_text())
    out = []
    for k in keys:
        va, vb = ma.get(k), mb.get(k)
        same = (va == vb) if not isinstance(va, float) else abs(va - vb) <= 1e-9 * max(1, abs(vb))
        out.append((k, va, vb, same))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=["sam", "linsam", "both"], default="both")
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    pairs = list(PAIRS) if args.pair == "both" else [args.pair]

    all_ok = True
    for p in pairs:
        print(f"=== pair {p}: endpoint τ={'1.0 vs sam' if p == 'sam' else '0.0 vs linsam_bnmatched'} ===")
        for name, extra in PAIRS[p]:
            train(name, extra, args.steps)
        (runa, _), (runb, _) = PAIRS[p]
        ok, worst, bad, ncols = compare_csv(runa, runb)
        print(f"metrics.csv  max rel diff = {worst:.3e}  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}  ({ncols} numeric cols"
              f"{'; ' + ', '.join(f'{c}:{d:.2e}' for c, d in bad[:6]) if bad else ''})")
        ok_j, worst_j = compare_jsonl(runa, runb)
        print(f"plant jsonl  max rel diff = {worst_j:.3e}  "
              f"{'PASS ✓' if ok_j else 'FAIL ✗'}")
        mism = [(k, va, vb) for k, va, vb, s in
                compare_manifest(runa, runb,
                                 ["final_loss", "final_acc", "lambda_max_final",
                                  "lambda_max_max"]) if not s]
        print(f"manifest finals {'MATCH ✓' if not mism else 'MISMATCH ✗ ' + str(mism)}")
        # bn_protocol strings differ BY DESIGN (arm labels); compare the
        # update-count class (leading token) instead, and show both.
        ma = json.loads((ROOT / "runs" / runa / "manifest.json").read_text())
        mb = json.loads((ROOT / "runs" / runb / "manifest.json").read_text())
        proto_ok = (ma.get("bn_protocol", "").split()[0]
                    == mb.get("bn_protocol", "").split()[0])
        print(f"bn_protocol class {'MATCH ✓' if proto_ok else 'MISMATCH ✗'} "
              f"[{ma.get('bn_protocol')}] vs [{mb.get('bn_protocol')}]")
        all_ok &= ok and ok_j and not mism and proto_ok
    print("\n" + ("ALL ENDPOINT CONTRACTS PASS" if all_ok else "CONTRACT FAILURE"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
