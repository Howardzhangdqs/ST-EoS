"""Run the EoS config under multiple seeds (Gap E setup). Same data, full-batch GD,
only the initialization differs — the clean test of Song & Yun trajectory alignment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# reuse the sweep machinery: axis "seed" is a top-level scalar
import scripts.sweep as sweep  # noqa: E402


def _run_one(value, config, steps, snapshot_every, sharp_every, sweep_dir, sweep_name):
    import copy, pandas as pd
    base = config
    if steps:
        base["train"]["steps"] = steps
    base["monitor"]["snapshot_every"] = snapshot_every
    if base.get("sharpness", {}).get("enabled"):
        base["sharpness"]["every"] = sharp_every
    cfg = copy.deepcopy(base)
    sweep.set_nested(cfg, "seed", int(value))
    run_name = f"{sweep_name}_{int(value)}"
    cfg["run_name"] = run_name
    run_dir = sweep_dir / run_name
    if (run_dir / "snapshots.npz").exists() and (run_dir / "sharpness.csv").exists():
        print(f"SKIP {run_name} (exists)")
    else:
        from steos.train import Trainer
        Trainer(cfg, run_dir).train()
    lr_used = float(cfg["optimizer"]["lr"])
    return sweep.summarize(run_dir, lr_used)


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mlp_synthetic_eos.yaml")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--sharp-every", type=int, default=50)
    ap.add_argument("--sweep-name", default="multiseed")
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    sweep_dir = Path(args.run_root) / args.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for s in seeds:
        row = _run_one(s, base, args.steps, args.snapshot_every, args.sharp_every, sweep_dir, args.sweep_name)
        rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).to_csv(sweep_dir / "summary.csv", index=False)
    print("\n=== multiseed sweep summary ===")
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
