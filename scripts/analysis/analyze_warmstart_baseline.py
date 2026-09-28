"""One-HVP warm-started power-iteration baseline vs the gradient Rayleigh
sharpness monitor λ̂_g = gᵀHg/gᵀg (reviewer-requested comparator).

Family 1 — OFFLINE, exact (MLP teacher–student, runs/mlp_rayleigh{,_s1..s4}):
  Rebuilds θ at every sharpness step from saved artifacts and runs the
  warm-start chain: v_0 = seeded random unit vector (torch.manual_seed(0));
  for s = 1..S−1: w = H(θ_s) v_{s−1} (single HVP);
  λ̂_ws[s] = v_{s−1}·w / ‖v_{s−1}‖²; v_s = w/‖w‖.
  Timeline note (steos/train.py): sharpness@step s is measured at the
  PRE-update θ_s, while snapshots@step s store the POST-update θ_{s+1}. The
  exact pre-update θ_s is recovered as snapshot@s − dthetas@s (dthetas.npz
  holds the single-step Δθ at the same step grid) — validated by a full-batch
  loss gate against metrics.csv and by recomputing λ̂_g offline and matching
  it against the in-run sharpness.csv lambda_hat_grad.
  The model has no BatchNorm/LayerNorm → offline HVPs at saved θ are exact.

Family 2 — ONLINE (toy PreAct-ResNet 0.175M / CIFAR-10, runs/cifar_ws_s{0..4}):
  Recorded during training by scripts/experiments/train_cifar_resnet.py --warm-start-track
  (new CSV columns lambda_hat_ws, cos_ws_vtop, t_hvp_g_s, t_hvp_ws_s). BN
  running stats are not in snapshots.npz → online-only.

For both families, λ̂_ws and λ̂_g are compared against ground-truth λ_max
(warm-started power iteration, 4–11 HVPs): Pearson + Spearman correlation,
relative level error |λ̂−λ_max|/λ_max (mean/median), early-vs-EoS regime split
by |cos(g, v_top)| > 0.7, and wall-clock cost per measurement.

Usage:
    .venv/bin/python scripts/analysis/analyze_warmstart_baseline.py [--device cuda]
        [--skip-offline] [--skip-online] [--skip-missing]
        [--out-json analysis_out/warmstart_baseline.json]
        [--out-md analysis_out/warmstart_baseline.md]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from steos.data import loss_fn, make_dataset
from steos.models import build_model
from steos.sharpness.hessian import _param_list, hvp

RUN_ROOT = Path("runs")
MLP_RUNS = ["mlp_rayleigh"] + [f"mlp_rayleigh_s{i}" for i in range(1, 5)]
CIFAR_RUNS = [f"cifar_ws_s{i}" for i in range(5)]
GATE = 0.7          # |cos(g, v_top)| > 0.7 ⇒ EoS regime (analyze_rayleigh_proxy L71)
NEG_TOL = 1e-6      # λ̂_ws may dip this far below 0 (numerical) without failing
LOSS_GATE_TOL = 1e-4
LAMG_XCHECK_TOL = 5e-3   # |λ̂_g offline − λ̂_g in-run| / |λ̂_g in-run| tolerance


# ---------------------------------------------------------------- statistics
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float(a @ b / d) if d > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    r = float(spearmanr(a, b)[0])  # type: ignore[arg-type]  # [0] = correlation
    return r if np.isfinite(r) else float("nan")


def est_metrics(name: str, est: np.ndarray, ref: np.ndarray, gate: np.ndarray) -> dict:
    """corr + relative-level-error stats of one estimator vs λ_max, split by
    the |cos(g, v_top)| > GATE regime gate (bool array; NaN cos ⇒ not EoS)."""
    ok = np.isfinite(est) & np.isfinite(ref) & (ref > 1e-6)
    out = {
        "estimator": name,
        "n_points": int(ok.sum()),
        "min_value": float(np.min(est[ok])) if ok.any() else float("nan"),
        "n_below_neg_tol": int((est[ok] < -NEG_TOL).sum()) if ok.any() else 0,
    }
    if ok.sum() >= 3:
        e, r = est[ok], ref[ok]
        rel = np.abs(e - r) / r
        out.update(
            pearson=pearson(e, r),
            spearman=spearman(e, r),
            rel_err_mean=float(rel.mean()),
            rel_err_median=float(np.median(rel)),
        )
        for regime, m in (("early", ok & ~gate), ("eos", ok & gate)):
            if m.sum() >= 3:
                ee, rr = est[m], ref[m]
                out[f"{regime}_pearson"] = pearson(ee, rr)
                out[f"{regime}_spearman"] = spearman(ee, rr)
                out[f"{regime}_rel_err_mean"] = float((np.abs(ee - rr) / rr).mean())
                out[f"n_{regime}"] = int(m.sum())
            else:
                out[f"{regime}_pearson"] = float("nan")
                out[f"{regime}_spearman"] = float("nan")
                out[f"{regime}_rel_err_mean"] = float("nan")
                out[f"n_{regime}"] = int(m.sum())
    else:
        for k in ("pearson", "spearman", "rel_err_mean", "rel_err_median",
                  "early_pearson", "eos_pearson", "early_spearman", "eos_spearman",
                  "early_rel_err_mean", "eos_rel_err_mean"):
            out[k] = float("nan")
        out["n_early"] = int((ok & ~gate).sum())
        out["n_eos"] = int((ok & gate).sum())
    return out


def aggregate(per_seed: list[dict], estimator: str) -> dict:
    """mean±sd (ddof=1) across seeds of the per-seed scalar metrics."""
    keys = ["pearson", "spearman", "rel_err_mean", "rel_err_median",
            "early_pearson", "eos_pearson", "n_early", "n_eos", "n_points"]
    agg: dict = {"estimator": estimator, "n_seeds": len(per_seed)}
    for k in keys:
        v = np.array([r[k] for r in per_seed], float)
        v = v[np.isfinite(v)]
        agg[f"{k}_mean"] = float(v.mean()) if len(v) else float("nan")
        agg[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return agg


def _corr_sanity(m: dict) -> bool:
    for k in ("pearson", "spearman", "early_pearson", "eos_pearson"):
        v = m.get(k, float("nan"))
        if np.isfinite(v) and abs(v) > 1.0 + 1e-9:
            return False
    return True


# ------------------------------------------------------------ family 1: MLP
def offline_run(run_name: str, device: str) -> dict:
    """Rebuild θ_s, run the one-HVP warm-start chain, compare vs sharpness.csv."""
    run_dir = RUN_ROOT / run_name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config"]
    assert not cfg["model"].get("layer_norm", False), "layer_norm HVP semantics differ"

    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name"), **mcfg).to(device)
    model.eval()  # no BN/dropout → eval == train; matches ARPACK-script convention
    params = _param_list(model)

    dcfg = dict(cfg["data"])
    data_seed = int(dcfg.pop("seed", 0))  # manifest carries none → make_dataset default
    X, y, task = make_dataset(dcfg.pop("name"), device=device, seed=data_seed, **dcfg)
    loss_f = loss_fn(task)

    sharp = pd.read_csv(run_dir / "sharpness.csv")
    snap = np.load(run_dir / "snapshots.npz", allow_pickle=True)
    dth = np.load(run_dir / "dthetas.npz")
    vtop = np.load(run_dir / "vtops.npz")
    metrics = pd.read_csv(run_dir / "metrics.csv")

    S = len(sharp)
    steps = sharp["step"].to_numpy(int)
    assert (steps == snap["steps"].astype(int)).all(), "snapshot/sharpness grid mismatch"
    assert (steps == dth["steps"].astype(int)).all(), "dtheta/sharpness grid mismatch"
    names = [str(n) for n in snap["param_names"]]
    cur = [n for n, _ in model.named_parameters()]
    assert names == cur, f"param_names mismatch in {run_name}"

    # exact pre-update θ_s = snapshot@s (post-update θ_{s+1}) − Δθ@s
    theta = snap["W"].astype(np.float32) + snap["ref"].astype(np.float32) - dth["D"].astype(np.float32)

    def load_theta(i: int) -> None:
        with torch.no_grad():
            s = 0
            for p in params:
                e = s + p.numel()
                p.copy_(torch.from_numpy(np.ascontiguousarray(theta[i, s:e])).to(
                    p.device, p.dtype).view(p.shape))
                s = e

    # --- gates: full-batch loss at θ_s vs metrics.csv (loss@s is pre-update) --
    loss_ref = metrics.set_index("step")["loss"]
    gate_errs = []
    for i in range(0, S, 10):  # spot-check every 10th frame
        load_theta(i)
        with torch.no_grad():
            lre = float(loss_f(model(X), y))
        gate_errs.append(abs(lre - float(loss_ref.loc[steps[i]]))
                         / max(abs(float(loss_ref.loc[steps[i]])), 1e-12))
    loss_gate_max = float(max(gate_errs))

    # --- the warm-start chain + offline λ̂_g recomputation (cost accounting) --
    n_flat = int(theta.shape[1])
    torch.manual_seed(0)                       # fixed random v_0 (task spec)
    v0 = torch.randn(n_flat, device=device, dtype=torch.float32)
    v0 = v0 / float(v0.norm())
    v_prev = [v0[s:e].view(p.shape) for s, e, p in _slices(params)]

    lam_ws = np.full(S, np.nan)
    cos_used = np.full(S, np.nan)   # |cos| of the vector USED at step i vs in-run v_top@i
    lam_g_off = np.full(S, np.nan)
    t_grad, t_hvp_g, t_hvp_ws = [], [], []
    Vtop = vtop["V"].astype(np.float32)

    for i in range(1, S):                      # s = 1..S−1 (v_0 has no predecessor)
        load_theta(i)
        # λ̂_g cost = gradient (fwd+bwd) + 1 HVP; also cross-checks the rebuild
        t0 = time.perf_counter()
        out = model(X)
        loss = loss_f(out, y)
        g = torch.autograd.grad(loss, params, create_graph=False)
        g = [gi.detach() for gi in g]
        t_grad.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        Hg = hvp(model, loss_f, X, y, g, params)
        t_hvp_g.append(time.perf_counter() - t0)
        gg = sum(float((a * a).sum().item()) for a in g)
        lam_g_off[i] = sum(float((a * b).sum().item()) for a, b in zip(g, Hg)) / max(gg, 1e-30)
        # the baseline: exactly ONE HVP with the persistent state
        v_used = v_prev
        v_used_flat = torch.cat([a.reshape(-1) for a in v_used])
        vt = torch.from_numpy(Vtop[i]).to(device)
        cos_used[i] = float(v_used_flat @ vt / (float(v_used_flat.norm()) * float(vt.norm())))
        t0 = time.perf_counter()
        w = hvp(model, loss_f, X, y, v_used, params)
        t_hvp_ws.append(time.perf_counter() - t0)
        vv = float(v_used_flat @ v_used_flat)
        vw = sum(float((a * b).sum().item()) for a, b in zip(v_used, w))
        lam_ws[i] = vw / max(vv, 1e-30)
        wn = float(np.sqrt(sum(float((b * b).sum().item()) for b in w)))
        v_prev = [b / wn for b in w] if wn > 0 else [a.clone() for a in v_used]

    # λ̂_g cross-check vs the in-run value (validates θ rebuild + data + surface)
    lam_g_csv = sharp["lambda_hat_grad"].to_numpy(float)
    xerr = np.abs(lam_g_off[1:] - lam_g_csv[1:]) / np.maximum(np.abs(lam_g_csv[1:]), 1e-12)
    lamg_xcheck_max = float(np.nanmax(xerr[np.isfinite(lam_g_csv[1:]) & (np.abs(lam_g_csv[1:]) > 1e-8)]))

    gate_mask = np.abs(sharp["cos_grad_vtop"].to_numpy(float)) > GATE
    lam_ref = sharp["lambda_max"].to_numpy(float)
    res = {
        "run": run_name,
        "seed": int(cfg["seed"]),
        "n_sharp_rows": int(S),
        "expected_n_points": int(S - 1),
        "loss_gate_max_relerr": loss_gate_max,
        "lam_g_offline_xcheck_max_relerr": lamg_xcheck_max,
        "estimators": {
            "lambda_hat_ws": est_metrics("lambda_hat_ws", lam_ws[1:], lam_ref[1:], gate_mask[1:]),
            "lambda_hat_grad": est_metrics("lambda_hat_grad", lam_g_csv[1:], lam_ref[1:], gate_mask[1:]),
        },
        "timing": {
            "ms_grad_mean": float(np.mean(t_grad) * 1e3),
            "ms_hvp_g_mean": float(np.mean(t_hvp_g) * 1e3),
            "ms_hvp_ws_mean": float(np.mean(t_hvp_ws) * 1e3),
            "lambda_max_n_hvp_mean": float(sharp["lambda_max_n_hvp"].mean()),
        },
        "diagnostics": {
            "mean_abs_cos_grad_vtop": float(np.abs(sharp["cos_grad_vtop"].to_numpy(float)[1:]).mean()),
            "mean_abs_cos_ws_used_vtop": float(np.nanmean(np.abs(cos_used[1:]))),
        },
        "series": {
            "step": steps[1:].tolist(),
            "lambda_max": lam_ref[1:].tolist(),
            "lambda_hat_ws": lam_ws[1:].tolist(),
            "lambda_hat_grad": lam_g_csv[1:].tolist(),
            "eos_gate": gate_mask[1:].tolist(),
            "cos_ws_used_vtop": cos_used[1:].tolist(),
        },
    }
    return res


def _slices(params):
    s = 0
    out = []
    for p in params:
        e = s + p.numel()
        out.append((s, e, p))
        s = e
    return out


# ------------------------------------------------------- family 2: CIFAR ws
def online_run(run_name: str) -> dict:
    run_dir = RUN_ROOT / run_name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    m = pd.read_csv(run_dir / "metrics.csv")
    lam_ref = m["lambda_max"].to_numpy(float)
    gate_mask = np.abs(m["cos_grad_vtop"].to_numpy(float)) > GATE
    ok_ref = np.isfinite(lam_ref) & (lam_ref > 1e-6)
    res = {
        "run": run_name,
        "seed": int(manifest.get("seed", -1)),
        "ws_init_seed": int(manifest.get("ws_init_seed", -1)),
        "n_rows": int(len(m)),
        "n_points": int(ok_ref.sum()),
        "estimators": {
            "lambda_hat_ws": est_metrics("lambda_hat_ws",
                                         m["lambda_hat_ws"].to_numpy(float),
                                         lam_ref, gate_mask),
            "lambda_hat_grad": est_metrics("lambda_hat_grad",
                                           m["lambda_hat_grad"].to_numpy(float),
                                           lam_ref, gate_mask),
        },
        "timing": {
            "ms_hvp_g_mean": float(m["t_hvp_g_s"].mean() * 1e3),
            "ms_hvp_ws_mean": float(m["t_hvp_ws_s"].mean() * 1e3),
            "lambda_max_n_hvp_mean": float(m["lambda_max_n_hvp"].mean()),
        },
        "diagnostics": {
            "mean_abs_cos_grad_vtop": float(np.abs(m["cos_grad_vtop"].to_numpy(float)[ok_ref]).mean()),
            "mean_abs_cos_ws_vtop": float(np.abs(m["cos_ws_vtop"].to_numpy(float)[ok_ref]).mean()),
        },
        "series": {
            "step": m["step"].tolist(),
            "lambda_max": lam_ref.tolist(),
            "lambda_hat_ws": m["lambda_hat_ws"].tolist(),
            "lambda_hat_grad": m["lambda_hat_grad"].tolist(),
            "eos_gate": gate_mask.tolist(),
            "cos_ws_vtop": m["cos_ws_vtop"].tolist(),
            "t_hvp_g_s": m["t_hvp_g_s"].tolist(),
            "t_hvp_ws_s": m["t_hvp_ws_s"].tolist(),
        },
    }
    return res


# ----------------------------------------------------------------- reporting
AGG_KEYS_EST = ["lambda_hat_ws", "lambda_hat_grad"]


def family_report(fam: dict, label: str, offline: bool) -> list[str]:
    lines = [f"## {label}", ""]
    seeds_txt = ", ".join(f"{r['run']}(seed {r['seed']})" for r in fam["runs"])
    lines += [f"Runs: {seeds_txt}", ""]
    hdr = ("| estimator | Pearson | Spearman | rel-err mean | rel-err median "
           "| early-Pearson | EoS-Pearson | n pts/seed |")
    lines += [hdr, "|---|---|---|---|---|---|---|---|"]
    for est in AGG_KEYS_EST:
        a = fam["aggregate"][est]
        def f(k): 
            v = a[f"{k}_mean"]; s = a[f"{k}_sd"]
            return "nan" if not np.isfinite(v) else (f"{v:+.3f}" if "pearson" in k or "spearman" in k else f"{v:.3f}") + (f"±{s:.3f}" if np.isfinite(s) else "")
        lines.append(
            f"| {est.replace('lambda_hat_', 'λ̂_')} | {f('pearson')} | {f('spearman')} | "
            f"{f('rel_err_mean')} | {f('rel_err_median')} | {f('early_pearson')} | "
            f"{f('eos_pearson')} | {a['n_points_mean']:.0f}±{a['n_points_sd']:.0f} |")
    lines.append("")
    t_keys = (["ms_grad_mean", "ms_hvp_g_mean", "ms_hvp_ws_mean", "lambda_max_n_hvp_mean"]
              if offline else ["ms_hvp_g_mean", "ms_hvp_ws_mean", "lambda_max_n_hvp_mean"])
    tm = {k: float(np.mean([r["timing"][k] for r in fam["runs"]])) for k in t_keys}
    if offline:
        lines.append(
            f"Cost per measurement: λ̂_g = grad fwd+bwd {tm['ms_grad_mean']:.1f} ms + "
            f"1 HVP {tm['ms_hvp_g_mean']:.1f} ms; λ̂_ws = 1 HVP {tm['ms_hvp_ws_mean']:.1f} ms; "
            f"λ_max ground truth = {tm['lambda_max_n_hvp_mean']:.1f} HVPs (in-run, warm-started).")
    else:
        lines.append(
            f"Cost per measurement: λ̂_g = 1 HVP {tm['ms_hvp_g_mean']:.1f} ms (HVP only; "
            f"its eval-batch gradient adds a fwd+bwd); λ̂_ws = 1 HVP {tm['ms_hvp_ws_mean']:.1f} ms; "
            f"λ_max ground truth = {tm['lambda_max_n_hvp_mean']:.1f} HVPs (in-run, warm-started).")
    n_e = fam["aggregate"]["lambda_hat_ws"]
    lines.append(
        f"Regime split (|cos(g,v_top)|>{GATE}): early {n_e['n_early_mean']:.0f}±{n_e['n_early_sd']:.0f} "
        f"pts, EoS {n_e['n_eos_mean']:.0f}±{n_e['n_eos_sd']:.0f} pts per seed "
        f"(estimator-independent: same gate for both).")
    d = fam["runs"][0]["diagnostics"]
    lines.append(
        f"Mean |cos(g,v_top)| = {d['mean_abs_cos_grad_vtop']:.3f}; "
        f"mean |cos(v_ws used, v_top)| = {d['mean_abs_cos_ws_used_vtop'] if offline else d['mean_abs_cos_ws_vtop']:.3f}.")
    return lines


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-offline", action="store_true")
    ap.add_argument("--skip-online", action="store_true")
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip online runs whose manifest is absent instead of exiting")
    ap.add_argument("--out-json", default="analysis_out/warmstart_baseline.json")
    ap.add_argument("--out-md", default="analysis_out/warmstart_baseline.md")
    args = ap.parse_args()

    report: dict = {"meta": {
        "gate": GATE, "neg_tol": NEG_TOL, "device": args.device,
        "mlp_runs": MLP_RUNS, "cifar_runs": CIFAR_RUNS,
        "v0_seed": 0,
        "sanity": {},
        "deviations": [
            "snapshots.npz frames are POST-update θ_{s+1} (steos/train.py saves after "
            "opt.step()); exact pre-update θ_s = snapshot@s − dthetas@s. Validated by "
            "the loss gate and by matching the offline-recomputed λ̂_g against the "
            "in-run sharpness.csv lambda_hat_grad.",
            "steos/sharpness/rayleigh.py gained an optional hvp_timer kwarg "
            "(default None, no behavior change) so t_hvp_g_s isolates the λ̂_g HVP.",
            "MLP family seed enters via the config field `seed`; configs/"
            "mlp_rayleigh_s{1..4}.yaml differ from mlp_rayleigh.yaml only in "
            "run_name+seed. CIFAR family uses the existing --seed CLI arg; v_ws is "
            "initialized at the first measurement from a dedicated torch.Generator "
            "seeded with --seed (online λ̂_ws is therefore defined at step 0 too).",
        ],
    }}
    md: list[str] = ["# One-HVP warm-started power-iteration baseline vs λ̂_g", ""]

    # ---- family 1 ----------------------------------------------------------
    if not args.skip_offline:
        runs = []
        for r in MLP_RUNS:
            print(f"[offline] {r} ...", flush=True)
            runs.append(offline_run(r, args.device))
        fam = {"runs": runs, "aggregate": {
            est: aggregate([r["estimators"][est] for r in runs], est)
            for est in AGG_KEYS_EST}}
        report["family1_mlp_offline"] = fam
        # sanity gates
        bad_corr = [r["run"] for r in runs for e in r["estimators"].values()
                    if not _corr_sanity(e)]
        min_ws = min(e["min_value"] for r in runs
                     for k, e in r["estimators"].items() if k == "lambda_hat_ws")
        n_below = sum(e["n_below_neg_tol"] for r in runs
                      for k, e in r["estimators"].items() if k == "lambda_hat_ws")
        npoints_ok = all(r["estimators"]["lambda_hat_ws"]["n_points"] == r["expected_n_points"]
                         for r in runs)
        report["meta"]["sanity"]["family1"] = {
            "corr_in_range": not bad_corr, "corr_violations": bad_corr,
            "lambda_hat_ws_min": min_ws, "lambda_hat_ws_n_below_neg_tol": n_below,
            "n_points_equals_rows_minus_1": npoints_ok,
            "loss_gate_max_relerr": max(r["loss_gate_max_relerr"] for r in runs),
            "lam_g_xcheck_max_relerr": max(r["lam_g_offline_xcheck_max_relerr"] for r in runs),
        }
        print(f"  gates: corr_ok={not bad_corr} min λ̂_ws={min_ws:.2e} "
              f"n_points==S−1: {npoints_ok} loss_gate≤{max(r['loss_gate_max_relerr'] for r in runs):.1e} "
              f"λ̂_g xcheck≤{max(r['lam_g_offline_xcheck_max_relerr'] for r in runs):.1e}")
        md += family_report(fam, "Family 1 — MLP teacher–student (offline exact HVPs, "
                                 "full-batch GD; 5 seeds)", offline=True)
        md.append("")

    # ---- family 2 ----------------------------------------------------------
    if not args.skip_online:
        runs, missing = [], []
        for r in CIFAR_RUNS:
            if not (RUN_ROOT / r / "manifest.json").exists():
                missing.append(r)
                if args.skip_missing:
                    print(f"[online] {r}: manifest missing — skipped")
                    continue
                raise SystemExit(f"[online] {r}: manifest missing (run the queue first "
                                 "or pass --skip-missing)")
            print(f"[online] {r} ...", flush=True)
            runs.append(online_run(r))
        fam = {"runs": runs, "aggregate": {
            est: aggregate([r["estimators"][est] for r in runs], est)
            for est in AGG_KEYS_EST}}
        report["family2_cifar_online"] = fam
        if runs:
            bad_corr = [r["run"] for r in runs for e in r["estimators"].values()
                        if not _corr_sanity(e)]
            min_ws = min(e["min_value"] for r in runs
                         for k, e in r["estimators"].items() if k == "lambda_hat_ws")
            n_below = sum(e["n_below_neg_tol"] for r in runs
                          for k, e in r["estimators"].items() if k == "lambda_hat_ws")
            report["meta"]["sanity"]["family2"] = {
                "corr_in_range": not bad_corr, "corr_violations": bad_corr,
                "lambda_hat_ws_min": min_ws, "lambda_hat_ws_n_below_neg_tol": n_below,
                "missing_runs": missing,
            }
            print(f"  gates: corr_ok={not bad_corr} min λ̂_ws={min_ws:.2e} missing={missing}")
            md += family_report(fam, "Family 2 — toy PreAct-ResNet 0.175M / CIFAR-10 "
                                     "(online tracker; 5 seeds)", offline=False)
            md.append("")

    md += ["## Deviations / notes", ""]
    md += [f"- {d}" for d in report["meta"]["deviations"]]
    md += ["", f"Sanity gates: λ̂_ws ≥ −{NEG_TOL:g} (violations reported in JSON "
               f"meta.sanity); corr ∈ [−1,1]; offline n_points = sharpness rows − 1."]

    out_json, out_md = Path(args.out_json), Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1))
    out_md.write_text("\n".join(md) + "\n")
    print(f"\n[done] wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
