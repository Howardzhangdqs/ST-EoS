"""At-scale rank-1 amplification arm aggregation (Finding 35c).

Reads runs/cifar100_r18_amp0.5_detarpack_s{seeds}/metrics.csv (deterministic
CUDA + ARPACK reference λ_max, the detarpack three-way protocol) and the
matching sgd/rank1/sam reference runs, and reports late-window statistics on
the same footing as Finding 31a:

  late  = steps >= 2000 (the Fig-30 / paper late window)
  lastq = steps >= 3750 (last quarter — the amplification arm is still
          descending at this budget, so the window mean understates it)

Usage: python scripts/analysis/analyze_amplify_at_scale.py [seeds ...]   (default: all found)
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("runs")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # t(0.975, df)


def load(opt: str, seeds):
    out = {}
    for s in seeds:
        f = RUNS / f"cifar100_r18_{opt}_detarpack_s{s}" / "metrics.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        lam = m.lambda_max.to_numpy(float)
        st = m.step.to_numpy()
        acc = float(m.eval_acc.iloc[-1])
        late = st >= 2000
        lastq = st >= 3750
        out[s] = dict(late=float(lam[late].mean()), lastq=float(lam[lastq].mean()),
                      frac38=float((lam[late] < 38).mean()), acc=acc)
    return out


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]] or list(range(5))
    print(f"{'arm':10s} {'n':>2s} {'late mean':>18s} {'last-q mean':>18s} "
          f"{'frac<38':>8s} {'final acc':>12s}")
    for opt in ("sgd", "rank1", "sam", "amp0.5", "amp1.0"):
        d = load(opt, seeds)
        if not d:
            continue
        n = len(d)
        t = T975.get(n - 1, 1.96)
        cols = {k: np.array([v[k] for v in d.values()]) for k in
                ("late", "lastq", "frac38", "acc")}

        def ms(k):
            v = cols[k]
            return f"{v.mean():7.2f}±{t*v.std(ddof=1)/sqrt(n):5.2f}"

        print(f"{opt:10s} {n:2d} {ms('late'):>18s} {ms('lastq'):>18s} "
              f"{cols['frac38'].mean():8.2f} {cols['acc'].mean():12.3f}")
        for s, v in d.items():
            print(f"  s{s}: late {v['late']:6.2f}  lastq {v['lastq']:6.2f}  acc {v['acc']:.3f}")


if __name__ == "__main__":
    main()
