"""Entry point: train an MLP on synthetic data with full EoS monitoring.

Usage:
    .venv/bin/python scripts/experiments/train_mlp_synthetic.py --config configs/mlp_synthetic.yaml
    .venv/bin/python -m steos.train --config configs/mlp_synthetic.yaml   # equivalent
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steos.train import run_from_config


def main():
    ap = argparse.ArgumentParser(description="ST-EoS milestone-0: MLP + synthetic + monitoring")
    ap.add_argument("--config", default="configs/mlp_synthetic.yaml")
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()
    run_from_config(args.config, args.run_root)


if __name__ == "__main__":
    main()
