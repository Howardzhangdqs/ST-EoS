"""Phase-1 screen: find/confirm a MULTI-SHARP EoS landscape.

Trains plain full-batch SGD on candidate multi-output teacher-student setups and
measures the top-12 Hessian spectrum at the EoS regime. A setup PASSES the
multi-sharp acceptance criterion when, at EoS:

    λ_2 >= 0.2 · (2/η)   AND   λ_1 / λ_2 < 3            (and λ_1 ≈ 2/η)

i.e. at least two comparably-sharp directions coexist (so spectral migration has
somewhere to flow), unlike the standard 1-D-sharp teacher-student
(λ_1≈269, λ_2≈0.6). The first setup that passes is the Phase-2 testbed.

Rationale (Li et al. NeurIPS 2022): the output layer dominates the top Hessian
eigenvalues; k isotropic (equal-norm-column) outputs ⇒ k comparable eigenvalues.

Usage:
    python scripts/experiments/multisharp_screen.py
    python scripts/experiments/multisharp_screen.py --configs wide       # wider/flat-spectrum sweep
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steos.data import loss_fn, make_dataset
from steos.models import build_model
from steos.optim import SGD, register_param_names
from steos.sharpness import top_k_eigenvalues

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.05
TWO_OVER_ETA = 2.0 / LR          # = 40
SPEC_K = 12
SPEC_ITER = 60
LAM2_MIN = 0.2 * TWO_OVER_ETA    # = 8.0   acceptance: λ_2 floor
LAM_RATIO_MAX = 3.0              # acceptance: λ_1/λ_2 ceiling

# Candidate setups ranked by likelihood (spec Phase-1 Step 1b). Each builds a
# multisharp_teacher_student dataset + a matching tanh MLP, trains plain SGD to
# EoS, and checks the multi-sharp criterion on the final top-12 spectrum.
CONFIGS = {
    # candidate 1 (MOST LIKELY): k isotropic outputs, moderate width
    "k8_h256":   dict(k=8,  teacher_hidden=[256, 256], out_norm=1.0, steps=1500),
    "k4_h256":   dict(k=4,  teacher_hidden=[256, 256], out_norm=1.0, steps=1500),
    "k8_h128":   dict(k=8,  teacher_hidden=[128, 128], out_norm=1.0, steps=1500),
    "k16_h256":  dict(k=16, teacher_hidden=[256, 256], out_norm=1.0, steps=1500),
    # candidate 2: wider hidden (more room for multiple sharp directions)
    "k8_wide":   dict(k=8,  teacher_hidden=[512, 512, 512], out_norm=1.0, steps=1500),
}
CONFIGS_WIDE = {  # the extended sweep (--configs wide), if the first pass misses
    "k4_h256":   dict(k=4,  teacher_hidden=[256, 256], out_norm=1.0, steps=1500),
    "k8_wide":   dict(k=8,  teacher_hidden=[512, 512, 512], out_norm=1.0, steps=1500),
    "k8_out2":   dict(k=8,  teacher_hidden=[256, 256], out_norm=2.0, steps=1500),
    "k8_out05":  dict(k=8,  teacher_hidden=[256, 256], out_norm=0.5, steps=1500),
    "k6_h256":   dict(k=6,  teacher_hidden=[256, 256], out_norm=1.0, steps=1500),
    "k8_lr02":   dict(k=8,  teacher_hidden=[256, 256], out_norm=1.0, steps=1500, lr=0.02),
}


def _build(k, teacher_hidden, out_norm, dim=20, n=1024, difficulty=0.3, seed=1):
    torch.manual_seed(seed)  # data seed fixed (matches rankk_sweep convention)
    X, y, task = make_dataset(
        "multisharp_teacher_student", n=n, dim=dim, difficulty=difficulty,
        teacher_hidden=teacher_hidden, out_dim=k, out_norm=out_norm,
        device=DEV, seed=seed,
    )
    return X, y, task


def _new_model(task, seed=0, hidden=(256, 256)):
    torch.manual_seed(seed)
    return build_model("mlp", in_dim=task["in_dim"], hidden_dims=list(hidden),
                       out_dim=task["out_dim"], activation="tanh").to(DEV)


def screen_one(name, cfg, data_seed=1):
    lr = cfg.get("lr", LR)
    two_over_eta = 2.0 / lr
    hidden = cfg["teacher_hidden"]
    X, y, task = _build(cfg["k"], hidden, cfg["out_norm"], seed=data_seed)
    model = _new_model(task, seed=0, hidden=hidden)
    opt = SGD(model.parameters(), lr=lr)
    register_param_names(opt, model)
    loss = loss_fn(task)
    steps = cfg["steps"]
    meas_every = max(50, steps // 15)
    rows = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(X)
        l = float(loss(out, y).item())
        loss(out, y).backward()
        opt.step()
        if step % meas_every == 0 or step == steps - 1:
            lam = float(top_k_eigenvalues(model, loss, X, y, k=1, n_iter=80)[0])
            rows.append((step, l, lam))
    # final top-12 spectrum at EoS
    spec = top_k_eigenvalues(model, loss, X, y, k=SPEC_K, n_iter=SPEC_ITER)
    spec = [float(x) for x in spec]
    l1, l2 = spec[0], spec[1]
    ratio = l1 / max(l2, 1e-9)
    passes = (l2 >= 0.2 * two_over_eta) and (ratio < LAM_RATIO_MAX)
    print(f"\n##### {name}: k={cfg['k']} hidden={hidden} out_norm={cfg['out_norm']} "
          f"lr={lr} (2/η={two_over_eta:g}) steps={steps} #####")
    print(f"  loss: init={rows[0][1]:.4f}  final={rows[-1][1]:.4f}")
    print(f"  λ_max trajectory: " + " ".join(f"{r[2]:.1f}" for r in rows[1:]))
    print(f"  top-{SPEC_K} spectrum at EoS: " + "  ".join(f"λ{i+1}={v:6.2f}" for i, v in enumerate(spec)))
    print(f"  λ_1={l1:.2f}  λ_2={l2:.2f}  λ_1/λ_2={ratio:.2f}  "
          f"(criterion: λ_2≥{0.2*two_over_eta:.1f} and ratio<{LAM_RATIO_MAX})")
    print(f"  => {'PASS (multi-sharp)' if passes else 'FAIL (1-D sharp or sub-EoS)'}")
    return dict(name=name, k=cfg["k"], hidden=str(hidden), out_norm=cfg["out_norm"],
                lr=lr, steps=steps, loss_final=rows[-1][1], lam_max_final=rows[-1][2],
                **{f"lam{i+1}": spec[i] for i in range(SPEC_K)},
                lam_ratio=ratio, passes=passes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", choices=["default", "wide"], default="default")
    args = ap.parse_args()
    cfgs = CONFIGS if args.configs == "default" else CONFIGS_WIDE
    t0 = time.time()
    print(f"Multi-sharp screen: DEV={DEV}, 2/η={TWO_OVER_ETA:g}, "
          f"criterion λ_2≥{LAM2_MIN:.1f} & λ_1/λ_2<{LAM_RATIO_MAX}")
    results = []
    for name, cfg in cfgs.items():
        try:
            results.append(screen_one(name, cfg))
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: ERROR {e}")
        if any(r["passes"] for r in results):
            print(f"\n*** first PASS found ({results[-1]['name']}); skipping remaining ***")
            break

    df = pd.DataFrame(results)
    pd.set_option("display.width", 240)
    print("\n=== screen summary ===")
    if not len(df):
        print("  (no candidate completed — all errored)")
    else:
        print(df[["name", "k", "hidden", "out_norm", "lr", "loss_final", "lam_max_final",
                  "lam1", "lam2", "lam3", "lam4", "lam5", "lam_ratio", "passes"]].to_string(
            index=False, float_format=lambda x: f"{x:.3g}"))
    passed = df[df.passes] if len(df) else df
    if len(passed):
        print(f"\n*** MULTI-SHARP CONFIRMED: {list(passed['name'])} ***")
        Path("figures").mkdir(parents=True, exist_ok=True)
        df.to_csv("figures/multisharp_screen.csv", index=False)
    else:
        print("\n*** NO candidate passed — see λ_1/λ_2 ratios above (informative negative) ***")
    print(f"\nwall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
