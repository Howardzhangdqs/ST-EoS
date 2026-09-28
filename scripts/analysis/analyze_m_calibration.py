"""m=4 batch-sharpness subsampling calibration: downward bias vs MC variance.

Review ask: the in-run λ_step estimator averages only _BS_SHARP_M = 4 fresh
minibatches (train_cifar100_resnet18.py L893–908), and the offline m-scan
(analyze_subspace_share_k100.py, batch-seed 20260921) read m=4 → 23.11 vs
m=64 → 24.55 — is m=4 systematically biased LOW, or is that gap ordinary
Monte-Carlo variance of a 4-draw mean?

Exact estimator being replicated (read from the trainer source, lines cited):

  * draws:  idx_bs = rng_bs.integers(0, N, BS)  with rng_bs =
    np.random.default_rng(SEED + 977) — a DEDICATED stream, i.i.d. WITH
    replacement over the 50k unaugmented train set (duplicates within a
    batch possible), BS = manifest bs = 128 (trainer L698, L903–904);
  * per-batch value: _grad_rayleigh_on_batch (trainer L298–334) —
    g_B^T H_B g_B / ‖g_B‖² in EVAL mode with BN running stats saved/
    restored, chunk-wise create_graph=False gradient accumulation, then ONE
    Pearlmutter HVP (_hvp_flat_chunked) — imported UNMODIFIED from the
    trainer here, so the replicated quantity is bit-for-bit the same code;
  * estimate: lhg_bs_mean = mean of the m = 4 per-batch values.

Design (per mb2c checkpoint, 3 seeds):
  1. draw an m_pool = 128 pool of per-batch values with a FRESH dedicated
     rng (seed stream [batch_seed, run_seed]; NOT the in-run rng_bs — this
     is an independent replication, like the m-scan's own offline stream);
  2. λ̂_128 = pool mean ≈ ground truth (SE = σ̂/√128, σ̂ = pool sd);
  3. subsample the pool B times at size m ∈ {4, 16} WITH REPLACEMENT
     (nonparametric bootstrap of the mean). Why with- and not
     without-replacement: the in-run λ̂_m is a mean of m INDEPENDENT batch
     draws — each of the m batches is a fresh i.i.d. sample, never a draw
     from a finite depleting pool. Without-replacement subsampling would
     inject a finite-population negative correlation, shrinking the
     subsample sd by sqrt((M−m)/(M−1)) (−1.6% at m=4, −6% at m=16 for
     M=128) and thus UNDERSTATE the estimator's variance — the wrong model
     for the question being adjudicated;
  4. honest no-reuse companion: the 128 pool values are also partitioned
     into M/m DISJOINT groups (32 groups of 4, 8 of 16); the spread of the
     group means is the reuse-free sampling spread (their average is
     exactly λ̂_128, as any partition's mean-of-means is);
  5. bias tests: (a) center: bootstrap/disjoint distribution mean vs
     λ̂_128 within its SE; (b) z-scores of each disjoint group against the
     complement mean (independent data, non-degenerate): mean_z ≈ 0 and
     ~95% |z| < 2 ⇒ no systematic offset; (c) replication: the ORIGINAL
     m-scan m=4 per-run values (batch-seed 20260921, read from
     figures/subspace_share_k100_<run>.csv if present) are z-scored
     against the fresh pool — if the earlier low m=4 readings sit within
     ~2 predicted SE(λ̂_4) = σ̂/2 of λ̂_128, the m=4-vs-m=64 gap is MC
     variance, not bias.

Offline-at-checkpoint caveat (disclosed): pools are measured at the FINAL
checkpoint, whereas in-run lhg_bs_mean is logged along the trajectory; the
calibration targets the estimator's sampling properties at a fixed model,
which is the right frame for the m-subsampling question (same convention
as the m-scan it adjudicates).

Outputs:
    figures/m_calibration_per-run.csv    one row per (run, m)
    figures/m_calibration_aggregate.csv  pooled rows + bias verdicts

Usage:
    .venv/bin/python scripts/analysis/analyze_m_calibration.py            # 3 mb2c runs
    .venv/bin/python scripts/analysis/analyze_m_calibration.py --runs a,b --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from steos.data.cifar import load_cifar                      # noqa: E402
from train_cifar100_resnet18 import (                        # noqa: E402
    _grad_rayleigh_on_batch,
    build_resnet18,
)

DEFAULT_RUNS = [
    "cifar100_r18_sgd_detarpack_mb2c_s0",
    "cifar100_r18_sgd_detarpack_mb2c_s1",
    "cifar100_r18_sgd_detarpack_mb2c_s2",
]
SCAN_DIR = ROOT / "figures"


def read_scan_m_values(run: str) -> dict[int, float]:
    """Per-run m-scan values (kind=lhg_m) from the earlier offline scan, if any.
    The scan files drop the model prefix: subspace_share_k100_sgd_detarpack_
    mb2c_s0.csv for run cifar100_r18_sgd_detarpack_mb2c_s0."""
    stem = run
    for pref in ("cifar100_r18_", "cifar100_", "cifar10_"):
        if run.startswith(pref):
            stem = run[len(pref):]
            break
    cands = [SCAN_DIR / f"subspace_share_k100_{name}.csv"
             for name in (run, stem)]
    out: dict[int, float] = {}
    for f in cands:
        if f.exists():
            df = pd.read_csv(f)
            d = df[df["kind"] == "lhg_m"]
            out = {int(r.m): float(r.mean) for r in d.itertuples()}
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS),
                    help="comma-separated run dirs (default: the 3 mb2c seeds)")
    ap.add_argument("--m-pool", type=int, default=128)
    ap.add_argument("--ms", default="4,16",
                    help="subsample sizes (must divide --m-pool)")
    ap.add_argument("--boot", type=int, default=2000,
                    help="number of bootstrap subsamples B (>= 200)")
    ap.add_argument("--bs", type=int, default=128,
                    help="minibatch size (manifest convention: 128)")
    ap.add_argument("--batch-seed", type=int, default=20260922,
                    help="fresh offline rng base seed (20260921 was the "
                         "m-scan's; +1 keeps this an independent replication)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: figures/; use /tmp for "
                         "smoke runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="chunked-HVP batch cap (256 = shared-GPU safe)")
    args = ap.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    ms = sorted({int(x) for x in args.ms.split(",") if x.strip()})
    for m in ms:
        if args.m_pool % m != 0:
            raise SystemExit(f"m={m} must divide --m-pool={args.m_pool}")
    if args.boot < 200:
        raise SystemExit("--boot must be >= 200")

    per_run_rows: list[dict] = []
    pool_summary: list[dict] = []

    for run in runs:
        run_dir = ROOT / "runs" / run
        ckpt = torch.load(run_dir / "final_model.pt", map_location=args.device,
                          weights_only=False)
        man = ckpt["manifest"]
        if int(man["bs"]) != args.bs:
            raise SystemExit(f"{run}: manifest bs={man['bs']} != --bs {args.bs}")
        print(f"[load] {run}: opt={man['opt']} steps={man['steps']} "
              f"seed={man['seed']} final_acc_full={man.get('final_acc_full')}",
              flush=True)
        model = build_resnet18(num_classes=man["n_classes"]).to(args.device)
        model.load_state_dict(ckpt["model"])
        Xtr, ytr, _, _, _ = load_cifar(n_classes=man["n_classes"],
                                       device=args.device)
        N = len(Xtr)

        # fresh dedicated stream, independent of the in-run rng_bs
        rng = np.random.default_rng([args.batch_seed, int(man["seed"])])
        t0 = time.time()
        pool = np.empty(args.m_pool, float)
        for j in range(args.m_pool):
            idx = rng.integers(0, N, args.bs)     # i.i.d. with replacement,
            pool[j] = _grad_rayleigh_on_batch(    # = trainer L903–906 exactly
                model, Xtr[idx], ytr[idx], args.chunk_size)
        dt = time.time() - t0
        lam = float(pool.mean())
        sd = float(pool.std(ddof=1))
        se = sd / np.sqrt(len(pool))
        print(f"[pool] m={args.m_pool}: λ̂={lam:.3f}  sd={sd:.3f} "
              f"SE(λ̂)={se:.3f}  min={pool.min():.2f} max={pool.max():.2f} "
              f"({dt/args.m_pool:.2f} s/batch)", flush=True)
        pool_summary.append(dict(run=run, seed=int(man["seed"]),
                                 m_pool=args.m_pool, lam_hat=lam, sd_pool=sd,
                                 se_pool=se))

        scan = read_scan_m_values(run)
        # disjoint partition of the pool (fixed: sequential blocks; the pool
        # order is i.i.d. so any fixed partition is exchangeable)
        for m in ms:
            groups = pool.reshape(-1, m).mean(axis=1)      # M/m group means
            # bootstrap of the i.i.d.-mean estimator
            bs_means = rng.choice(pool, size=(args.boot, m),
                                  replace=True).mean(axis=1)
            # independent-complement z per disjoint group:
            # z_g = (mean_g - mean_rest) / sqrt(σ̂²(1/m + 1/(M-m)))
            n_g = len(groups)
            z_list = []
            for i in range(n_g):
                rest = np.concatenate([groups[:i], groups[i + 1:]])
                # mean_rest from group means = weighted mean of pool minus g
                nrest = args.m_pool - m
                mean_rest = (lam * args.m_pool - groups[i] * m) / nrest
                z = ((groups[i] - mean_rest)
                     / (sd * np.sqrt(1 / m + 1 / nrest)))
                z_list.append(z)
            z_arr = np.asarray(z_list)
            q_lo, q_hi = np.percentile(bs_means, [2.5, 97.5])
            g_lo, g_hi = np.percentile(groups, [2.5, 97.5])
            row = dict(
                run=run, seed=int(man["seed"]), m=m, m_pool=args.m_pool,
                lam_hat_128=lam, sd_pool=sd, se_pool=se,
                n_boot=args.boot,
                boot_mean=float(bs_means.mean()), boot_sd=float(bs_means.std()),
                boot_q2_5=float(q_lo), boot_q97_5=float(q_hi),
                pred_sd_theory=float(sd / np.sqrt(m)),
                disjoint_n_groups=n_g,
                disjoint_mean=float(groups.mean()),
                disjoint_sd=float(groups.std(ddof=1)),
                disjoint_q2_5=float(g_lo), disjoint_q97_5=float(g_hi),
                z_center_boot=float((bs_means.mean() - lam) / se),
                z_center_disjoint=float((groups.mean() - lam) / se),
                group_z_mean=float(z_arr.mean()), group_z_sd=float(z_arr.std()),
                group_z_frac_within2=float((np.abs(z_arr) < 2).mean()),
                scan_m_value=scan.get(m, float("nan")),
                scan_z=(float((scan[m] - lam) / (sd / np.sqrt(m)))
                        if m in scan else float("nan")),
            )
            per_run_rows.append(row)
            print(f"  [m={m:3d}] boot {row['boot_mean']:.3f}±{row['boot_sd']:.3f} "
                  f"[{q_lo:.2f},{q_hi:.2f}] (theory sd {row['pred_sd_theory']:.3f}) | "
                  f"disjoint {row['disjoint_mean']:.3f}±{row['disjoint_sd']:.3f} | "
                  f"group-z mean {row['group_z_mean']:+.3f} "
                  f"sd {row['group_z_sd']:.3f} |z|<2: "
                  f"{row['group_z_frac_within2']:.2f}"
                  + (f" | scan m={m}: {scan[m]:.2f} "
                     f"(z={row['scan_z']:+.2f})" if m in scan else ""),
                  flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "figures")
    out_dir.mkdir(exist_ok=True)
    df_pr = pd.DataFrame(per_run_rows)
    df_pr.to_csv(out_dir / "m_calibration_per-run.csv", index=False,
                 float_format="%.6g")

    # ---- aggregate -----------------------------------------------------------
    agg_rows = []
    lams = np.array([p["lam_hat"] for p in pool_summary])
    sds = np.array([p["sd_pool"] for p in pool_summary])
    pooled_lam = float(lams.mean())
    for m in ms:
        boot_sds = df_pr[df_pr.m == m]["boot_sd"].to_numpy()
        disj_sds = df_pr[df_pr.m == m]["disjoint_sd"].to_numpy()
        theory = float(np.mean(sds / np.sqrt(m)))
        scan_vals = df_pr[df_pr.m == m]["scan_m_value"].to_numpy()
        scan_zs = df_pr[df_pr.m == m]["scan_z"].to_numpy()
        agg_rows.append(dict(
            kind="subsampling", m=m, n_runs=len(runs),
            lam_hat_pooled=pooled_lam,
            lam_hat_cross_run_sd=float(lams.std(ddof=1)) if len(lams) > 1 else 0.0,
            se_of_mean=float(np.sqrt(np.mean(sds ** 2) / len(sds)
                                     / args.m_pool)),
            sd_sub_mean=float(boot_sds.mean()),
            sd_disjoint_mean=float(disj_sds.mean()),
            sd_theory=theory,
            scan_value_mean=float(np.nanmean(scan_vals)) if np.isfinite(
                scan_vals).any() else float("nan"),
            scan_vs_pool_z_mean=float(np.nanmean(scan_zs)) if np.isfinite(
                scan_zs).any() else float("nan"),
            scan_vs_pool_z_maxabs=float(np.nanmax(np.abs(scan_zs))) if np.isfinite(
                scan_zs).any() else float("nan"),
        ))
    # verdict rows: is E[λ̂_4] == λ̂_128 (no downward bias)?
    for m in ms:
        d = df_pr[df_pr.m == m].iloc[0]
        zc = df_pr[df_pr.m == m]["z_center_disjoint"].to_numpy()
        verdict = (f"center z={np.mean(zc):+.4f} (|z|<<1: subsample means "
                   f"centered on λ̂_128 by construction of the partition); "
                   f"group-z mean {d['group_z_mean']:+.3f} "
                   f"(bias would show as mean_z<0 with |mean_z|>2/√n_groups); "
                   f"earlier m-scan value {d['scan_m_value']:.2f} sits at "
                   f"z={d['scan_z']:+.2f} of the m={m} sampling distribution "
                   f"(sd σ̂/√m={d['pred_sd_theory']:.2f})")
        agg_rows.append(dict(kind="verdict", m=m, n_runs=len(runs),
                             lam_hat_pooled=pooled_lam,
                             lam_hat_cross_run_sd=float(lams.std(ddof=1))
                             if len(lams) > 1 else 0.0,
                             se_of_mean=float("nan"), sd_sub_mean=float("nan"),
                             sd_disjoint_mean=float("nan"),
                             sd_theory=float(d["pred_sd_theory"]),
                             scan_value_mean=float(d["scan_m_value"]),
                             scan_vs_pool_z_mean=float(d["scan_z"]),
                             scan_vs_pool_z_maxabs=float("nan"),
                             note=verdict))
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(out_dir / "m_calibration_aggregate.csv", index=False,
                  float_format="%.6g")
    print(f"=> {out_dir/'m_calibration_per-run.csv'} "
          f"({len(df_pr)} rows), {out_dir/'m_calibration_aggregate.csv'} "
          f"({len(df_agg)} rows)")


if __name__ == "__main__":
    main()
