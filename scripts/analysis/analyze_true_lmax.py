"""Analyze the true-λ_max plant main experiment (ST-EoS-PR-01).

Reads:
    runs/cifar100_r18_truelmax_s{0,1,2}/  (plant: plant_per_step.jsonl + metrics.csv)
    runs/cifar100_r18_sgd_arm_s{0,1,2}/    (vanilla reference with ARPACK outcome)

Computes (per preregistration §4):
  §4.1 eligibility gate: divisor median ≥1.5, fire rate ≥95%, damping mean >0.3
  §4.2 outcome classification: escape / pin(a) / pin(b) / break vs vanilla
  §4.3 Ritz residual: effective-force rate (fraction steps ‖Hv−λv‖/λ < 0.1)

Writes:
    runs/main/ANALYSIS.md  (human briefing)
    runs/main/analysis.json (machine-readable)

Usage:
    python scripts/analysis/analyze_true_lmax.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd


RUN_ROOT = Path("runs")
LAM_TARG = 20.0
OUT_DIR = Path("runs/main")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_plant(seed):
    d = RUN_ROOT / f"cifar100_r18_truelmax_s{seed}"
    if not d.exists():
        return None
    per_step = []
    jpath = d / "plant_per_step.jsonl"
    if jpath.exists():
        with open(jpath) as f:
            for line in f:
                per_step.append(json.loads(line))
    mpath = d / "metrics.csv"
    metrics = pd.read_csv(mpath) if mpath.exists() else None
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {"per_step": per_step, "metrics": metrics, "manifest": manifest, "dir": d}


def load_vanilla(seed):
    d = RUN_ROOT / f"cifar100_r18_sgd_arm_s{seed}"
    if not d.exists():
        return None
    mpath = d / "metrics.csv"
    metrics = pd.read_csv(mpath) if mpath.exists() else None
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {"metrics": metrics, "manifest": manifest, "dir": d}


def _safe(f, xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return f(xs) if xs else float("nan")


def eligibility_gate(plant):
    """Preregistration §4.1: actuator fired with real force.
    Primary: divisor median ≥1.5. Aux: fire rate ≥95%, damping mean >0.3."""
    ps = plant["per_step"]
    if not ps:
        return {"gate_pass": False, "reason": "no per-step data"}
    # skip step 0 (init transient, λ_max huge at random init — preregistration
    # spirit: gate on the "actuation regime", steps >0)
    divisors = [r["divisor"] for r in ps if r.get("step", 0) > 0
                and not np.isnan(r.get("divisor", float("nan")))]
    dampings = [r["damping"] for r in ps if r.get("step", 0) > 0
                and not np.isnan(r.get("damping", float("nan")))]
    fires = [r["fire"] for r in ps if r.get("step", 0) > 0]
    div_med = st.median(divisors) if divisors else float("nan")
    fire_rate = (sum(fires) / len(fires)) if fires else float("nan")
    damp_mean = st.mean(dampings) if dampings else float("nan")
    primary_pass = div_med >= 1.5
    aux_pass = (fire_rate >= 0.95) and (damp_mean > 0.3)
    return {
        "divisor_median": div_med,
        "divisor_mean": st.mean(divisors) if divisors else float("nan"),
        "fire_rate": fire_rate,
        "damping_mean": damp_mean,
        "damping_median": st.median(dampings) if dampings else float("nan"),
        "primary_pass": bool(primary_pass),
        "aux_pass": bool(aux_pass),
        "gate_pass": bool(primary_pass and aux_pass),
        "n_steps_evaluated": len(divisors),
    }


def effective_force_rate(plant):
    """Preregistration §4.3: fraction of REFRESH steps with Ritz < 0.1.
    If <80% → staleness confound flagged (becomes binding control)."""
    ps = plant["per_step"]
    ritz_at_refresh = [r["ritz"] for r in ps
                       if r.get("is_refresh_step") and not np.isnan(r.get("ritz", float("nan")))]
    if not ritz_at_refresh:
        return {"effective_force_rate": float("nan"), "n_refresh": 0, "flag": "no_refresh_data"}
    n_under = sum(1 for r in ritz_at_refresh if r < 0.1)
    rate = n_under / len(ritz_at_refresh)
    return {
        "effective_force_rate": rate,
        "n_refresh": len(ritz_at_refresh),
        "ritz_median": st.median(ritz_at_refresh),
        "ritz_p90": sorted(ritz_at_refresh)[int(0.9 * len(ritz_at_refresh)) - 1],
        "staleness_confound_flag": rate < 0.80,
    }


def trajectory_summary(metrics):
    """λ_max trajectory stats from outcome measurements (every 200 steps).
    Skip step 0 (init transient) when computing 'steady' stats."""
    if metrics is None or len(metrics) == 0:
        return {}
    df = metrics
    lams = df["lambda_max"].dropna().tolist()
    steady = df[df["step"] > 0]["lambda_max"].dropna().tolist()
    out = {
        "n_measurements": len(lams),
        "lambda_max_init": float(df["lambda_max"].iloc[0]) if len(df) else float("nan"),
        "lambda_max_final": float(df["lambda_max"].iloc[-1]) if len(df) else float("nan"),
        "lambda_max_max": float(df["lambda_max"].max()) if len(df) else float("nan"),
    }
    if steady:
        out.update({
            "lambda_max_steady_mean": st.mean(steady),
            "lambda_max_steady_median": st.median(steady),
            "lambda_max_steady_min": min(steady),
            "lambda_max_steady_max": max(steady),
            "lambda_max_steady_p25": sorted(steady)[int(0.25 * len(steady))],
            "lambda_max_steady_p75": sorted(steady)[int(0.75 * len(steady)) - 1],
            "lambda_max_above_ltarg_frac": sum(1 for v in steady if v > LAM_TARG) / len(steady),
            "final_acc": float(df["eval_acc"].iloc[-1]) if "eval_acc" in df else float("nan"),
            "final_loss": float(df["train_loss"].iloc[-1]) if "train_loss" in df else float("nan"),
        })
    return out


def classify_outcome(plant_trajs, vanilla_trajs, final_acc_plant, final_acc_vanilla):
    """Preregistration §4.2 outcome classification.

    Escape:  plant crosses λ_targ, persists, exceeds vanilla trajectory, stays
             above λ_targ for ≥80% of post-crossing steps.
    Pin(a):  plant λ_max overlaps vanilla (both > λ_targ).
    Pin(b):  plant λ_max ≪ vanilla, plant ≈ λ_targ (±20%).
    Break:   train loss diverges or final acc < 0.5× vanilla."""
    if not plant_trajs or not vanilla_trajs:
        return {"classification": "inconclusive", "reason": "missing data"}

    # Break check first (preregistration: final acc < 0.5× vanilla)
    if (final_acc_vanilla and final_acc_plant is not None
            and final_acc_plant < 0.5 * final_acc_vanilla):
        return {"classification": "break",
                "reason": f"plant acc {final_acc_plant:.3f} < 0.5×vanilla {final_acc_vanilla:.3f}"}

    # Aggregate vanilla steady median (the comparator)
    vanilla_steady_medians = [t["lambda_max_steady_median"] for t in vanilla_trajs
                              if "lambda_max_steady_median" in t]
    if not vanilla_steady_medians:
        return {"classification": "inconclusive", "reason": "no vanilla steady data"}
    vanilla_med = st.mean(vanilla_steady_medians)

    # Per-seed classification (preregistration: "all three" for Escape).
    # Order matters: Pin(b) (most specific: plant pinned near λ_targ, ≪ vanilla)
    # → Escape (plant clearly exceeds vanilla) → Pin(a) (plant ≈ vanilla) →
    # ambiguous.
    seed_classes = []
    for pt in plant_trajs:
        if "lambda_max_steady_median" not in pt:
            seed_classes.append("inconclusive"); continue
        pm = pt["lambda_max_steady_median"]
        above_frac = pt.get("lambda_max_above_ltarg_frac", 0.0)
        # Pin(b): plant ≈ λ_targ (within ±20%) AND substantially below vanilla
        if abs(pm - LAM_TARG) / LAM_TARG <= 0.20 and pm <= 0.7 * vanilla_med:
            seed_classes.append("pin(b)")
        # Escape: plant clearly EXCEEDS vanilla (≥1.2×) and stays above λ_targ
        elif pm >= 1.2 * vanilla_med and above_frac >= 0.5:
            seed_classes.append("escape")
        # Pin(a): plant OVERLAPS vanilla (within ±30%, both > λ_targ)
        elif abs(pm - vanilla_med) / max(vanilla_med, 1e-9) <= 0.30 and pm > LAM_TARG:
            seed_classes.append("pin(a)")
        else:
            seed_classes.append("ambiguous")

    # Aggregate
    if all(c == "escape" for c in seed_classes):
        overall = "escape"
    elif all(c == "pin(b)" for c in seed_classes):
        overall = "pin(b)"
    elif all(c == "pin(a)" for c in seed_classes):
        overall = "pin(a)"
    else:
        # mixed → use the majority, flag
        from collections import Counter
        cnt = Counter(seed_classes)
        overall = cnt.most_common(1)[0][0] + " (mixed)"

    return {
        "classification": overall,
        "seed_classes": seed_classes,
        "vanilla_steady_median_mean": vanilla_med,
        "vanilla_steady_medians": vanilla_steady_medians,
        "plant_steady_medians": [pt.get("lambda_max_steady_median", float("nan"))
                                  for pt in plant_trajs],
    }


def main():
    report = {"seeds": [], "LAM_TARG": LAM_TARG}

    plants = []
    vanillas = []
    # Auto-discover seeds: try 0..9, keep whichever have plant or vanilla data.
    for seed in range(10):
        p = load_plant(seed)
        v = load_vanilla(seed)
        if p is None and v is None:
            continue
        seed_rec = {"seed": seed}
        if p is not None:
            plants.append(p)
            seed_rec["plant_gate"] = eligibility_gate(p)
            seed_rec["plant_ritz"] = effective_force_rate(p)
            seed_rec["plant_traj"] = trajectory_summary(p["metrics"])
        if v is not None:
            vanillas.append(v)
            seed_rec["vanilla_traj"] = trajectory_summary(v["metrics"])
        report["seeds"].append(seed_rec)

    # Aggregate outcome classification (use whatever seeds are available)
    plant_trajs = [s["plant_traj"] for s in report["seeds"] if "plant_traj" in s]
    vanilla_trajs = [s["vanilla_traj"] for s in report["seeds"] if "vanilla_traj" in s]
    final_acc_p = _safe(lambda xs: xs[-1],
                        [s["plant_traj"].get("final_acc") for s in report["seeds"]
                         if "plant_traj" in s and "final_acc" in s["plant_traj"]])
    final_acc_v = _safe(lambda xs: xs[-1],
                        [s["vanilla_traj"].get("final_acc") for s in report["seeds"]
                         if "vanilla_traj" in s and "final_acc" in s["vanilla_traj"]])
    report["outcome"] = classify_outcome(plant_trajs, vanilla_trajs, final_acc_p, final_acc_v)

    # Aggregate gate across seeds (only seed 0 in interim)
    if plant_trajs:
        divisors = [s["plant_gate"]["divisor_median"] for s in report["seeds"]
                    if "plant_gate" in s and not np.isnan(s["plant_gate"].get("divisor_median", float("nan")))]
        report["aggregate_divisor_median_across_seeds"] = (
            {"mean": st.mean(divisors), "range": [min(divisors), max(divisors)]} if divisors else {}
        )

    (OUT_DIR / "analysis.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- human briefing ----
    lines = ["# True-λ_max Plant — Main Experiment Analysis (ST-EoS-PR-01)", "",
             f"λ_targ = {LAM_TARG}", ""]
    for s in report["seeds"]:
        lines.append(f"## Seed {s['seed']}")
        if "plant_gate" in s:
            g = s["plant_gate"]
            lines.append(f"### Plant §4.1 eligibility gate")
            lines.append(f"- divisor median: **{g['divisor_median']:.3f}**  "
                         f"(mean {g['divisor_mean']:.3f})  — primary pass: "
                         f"**{g['primary_pass']}** (≥1.5)")
            lines.append(f"- fire rate: **{g['fire_rate']*100:.1f}%**  (≥95% required)")
            lines.append(f"- damping mean: **{g['damping_mean']:+.3f}**  "
                         f"(median {g['damping_median']:+.3f})  (>0.3 required)")
            lines.append(f"- **GATE PASS: {g['gate_pass']}**")
        if "plant_ritz" in s:
            r = s["plant_ritz"]
            lines.append(f"### Plant §4.3 Ritz residual (staleness guard)")
            lines.append(f"- effective-force rate: **{r.get('effective_force_rate', float('nan'))*100:.1f}%**  "
                         f"(n_refresh={r.get('n_refresh')})")
            lines.append(f"- ritz median: {r.get('ritz_median', float('nan')):.4f}  "
                         f"p90: {r.get('ritz_p90', float('nan')):.4f}")
            if r.get("staleness_confound_flag"):
                lines.append(f"- **STALENESS CONFOUND FLAGGED** — Round-1 staleness scan becomes binding")
        if "plant_traj" in s:
            t = s["plant_traj"]
            lines.append(f"### Plant outcome trajectory")
            lines.append(f"- λ_max init={t.get('lambda_max_init', float('nan')):.1f}  "
                         f"max={t.get('lambda_max_max', float('nan')):.1f}  "
                         f"final={t.get('lambda_max_final', float('nan')):.1f}")
            if "lambda_max_steady_median" in t:
                lines.append(f"- steady (step>0): median **{t['lambda_max_steady_median']:.2f}**  "
                             f"mean {t['lambda_max_steady_mean']:.2f}  "
                             f"range [{t['lambda_max_steady_min']:.1f}, {t['lambda_max_steady_max']:.1f}]")
                lines.append(f"- fraction steps with λ_max > λ_targ: "
                             f"{t['lambda_max_above_ltarg_frac']*100:.0f}%")
            lines.append(f"- final train_loss={t.get('final_loss', float('nan')):.3f}  "
                         f"final acc={t.get('final_acc', float('nan')):.3f}")
        if "vanilla_traj" in s:
            t = s["vanilla_traj"]
            lines.append(f"### Vanilla SGD reference trajectory")
            lines.append(f"- λ_max init={t.get('lambda_max_init', float('nan')):.1f}  "
                         f"max={t.get('lambda_max_max', float('nan')):.1f}  "
                         f"final={t.get('lambda_max_final', float('nan')):.1f}")
            if "lambda_max_steady_median" in t:
                lines.append(f"- steady (step>0): median **{t['lambda_max_steady_median']:.2f}**  "
                             f"mean {t['lambda_max_steady_mean']:.2f}  "
                             f"range [{t['lambda_max_steady_min']:.1f}, {t['lambda_max_steady_max']:.1f}]")
            lines.append(f"- final train_loss={t.get('final_loss', float('nan')):.3f}  "
                         f"final acc={t.get('final_acc', float('nan')):.3f}")
        lines.append("")

    # Outcome classification
    lines.append(f"## §4.2 Outcome classification")
    o = report["outcome"]
    lines.append(f"- **Classification: {o['classification']}**")
    if "seed_classes" in o:
        lines.append(f"- per-seed: {o['seed_classes']}")
    if "vanilla_steady_median_mean" in o:
        lines.append(f"- vanilla steady-λ_max median (mean across seeds): "
                     f"{o['vanilla_steady_median_mean']:.2f}")
    if "plant_steady_medians" in o:
        lines.append(f"- plant steady-λ_max medians per seed: "
                     f"{[round(x,2) for x in o['plant_steady_medians']]}")
    lines.append("")

    text = "\n".join(lines)
    (OUT_DIR / "ANALYSIS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
