"""Cross-batch sensing robustness study — review round 2 (fixed-batch confound).

Question: the at-ResNet-scale reference λ_max and the controller's sensed
v̂_top come from a FIXED 512-example eval batch. Is that direction actually
the sharp direction of other batches / of the current training minibatch?
If cross-batch top-eigenspace overlap were low, "rank-1 is inert" could not
be attributed to rank. This script quantifies:

  * λ_max agreement across 3 fixed batches (seeds 123/124/125) and the
    current training minibatch (bs=128),
  * |cos(v1_ref, v1_other)| top-eigenvector overlaps,
  * top-3 subspace overlaps (min principal-angle cosine).

Reads the `--multibatch` runs (cifar100_r18_sgd_detarpack_mb_s*).
Outputs: printed summary + figures/crossbatch_summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PFX = "cifar100_r18_sgd_detarpack_mb_s"
LATE_STEP = 2000


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))
    return float(xs.mean()), h


def main():
    dfs = []
    for d in sorted(Path("runs").glob(PFX + "*")):
        s = d.name.rsplit("_s", 1)[-1]
        if s.isdigit() and (d / "metrics.csv").exists():
            df = pd.read_csv(d / "metrics.csv")
            if "lam1_b124" in df.columns:
                dfs.append((int(s), df))
    if not dfs:
        raise SystemExit("no multibatch runs found (queue stage 1)")
    print(f"multibatch runs: seeds {[s for s, _ in dfs]}")

    lam_cols = {"ref (seed123)": "lambda_max", "fixed b124": "lam1_b124",
                "fixed b125": "lam1_b125", "current minibatch": "lam1_curmb"}
    cos_cols = {"b124": "cos_v1_ref_b124", "b125": "cos_v1_ref_b125",
                "curmb": "cos_v1_ref_curmb"}
    sub_cols = {"b124": "sub_min_ref_b124", "b125": "sub_min_ref_b125",
                "curmb": "sub_min_ref_curmb"}

    rows = []
    late_all = []
    for name, col in lam_cols.items():
        per_seed = []
        for _, df in dfs:
            m = df[df["step"] >= LATE_STEP]
            per_seed.append(float(m[col].mean()))
            late_all.append(df[df["step"] >= LATE_STEP])
        m, h = ci95(per_seed)
        rows.append(dict(quantity=f"late λ {name}", mean=m, ci95=h, n=len(per_seed)))
        print(f"late λ {name:18s} = {m:6.2f} ± {h:4.2f}")
    for name, col in cos_cols.items():
        vals = np.concatenate([df.loc[df["step"] >= LATE_STEP, col].abs().dropna().to_numpy()
                               for _, df in dfs])
        rows.append(dict(quantity=f"|cos(v1_ref, v1_{name})| median",
                         mean=float(np.median(vals)), ci95=float(np.percentile(vals, 75)),
                         n=len(vals)))
        print(f"|cos(v1_ref, v1_{name})|: median {np.median(vals):.3f}  "
              f"IQR [{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]")
    for name, col in sub_cols.items():
        vals = np.concatenate([df.loc[df["step"] >= LATE_STEP, col].dropna().to_numpy()
                               for _, df in dfs])
        rows.append(dict(quantity=f"top-3 subspace min-cos vs {name} median",
                         mean=float(np.median(vals)), ci95=float(np.percentile(vals, 75)),
                         n=len(vals)))
        print(f"top-3 subspace min-cos vs {name}: median {np.median(vals):.3f}")

    pd.DataFrame(rows).to_csv("figures/crossbatch_summary.csv", index=False)
    print("=> figures/crossbatch_summary.csv")


if __name__ == "__main__":
    main()
