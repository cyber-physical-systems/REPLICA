from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT = Path("/workspace/sc26_rebuttal")
PROFILE_FILE = PROJECT / "experiment2/generated/execution_profiles.json"
OUTDIR = PROJECT / "experiment2/generated/scenarios"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42

TARGET_RHOS = [1.0, 0.8, 0.6, 0.4, 0.2]


# ============================================================
# Load nominal empirical feasibility
# ============================================================

profiles = json.loads(PROFILE_FILE.read_text())

assignments = sorted({
    (p["model"], p["stage"], p["resource_id"])
    for p in profiles
})

models = sorted({x[0] for x in assignments})
stages = ["update", "evaluate", "package",
          "deploy", "validate", "reactivate"]
resources = sorted({x[2] for x in assignments})

nominal_n = len(assignments)

print("Models:     ", models)
print("Stages:     ", stages)
print("Resources:  ", resources)
print("Nominal N:  ", nominal_n)


# ============================================================
# Solvability
#
# Every model-stage must retain >= 1 resource.
# ============================================================

def is_solvable(active):
    active = set(active)

    for model in models:
        for stage in stages:
            if not any(
                (model, stage, r) in active
                for r in resources
            ):
                return False

    return True


def option_counts(active):
    active = set(active)

    out = {}

    for model in models:
        for stage in stages:
            n = sum(
                (model, stage, r) in active
                for r in resources
            )
            out[f"{model}:{stage}"] = n

    return out


# ============================================================
# Generate constrained matrix
#
# Random removal is ONLY the mechanism for producing a
# controlled density ladder here.
#
# We preserve at least one resource option for every task.
# Multiple seeds will later provide repetitions.
# ============================================================

def generate_for_rho(target_rho, seed):
    rng = random.Random(seed)

    active = set(assignments)

    target_n = round(nominal_n * target_rho)

    candidates = list(active)
    rng.shuffle(candidates)

    for assignment in candidates:

        if len(active) <= target_n:
            break

        trial = active - {assignment}

        if is_solvable(trial):
            active = trial

    return sorted(active)


# ============================================================
# Build scenarios
# ============================================================

summary = []

for i, rho_target in enumerate(TARGET_RHOS):

    active = generate_for_rho(
        target_rho=rho_target,
        seed=SEED + i,
    )

    n = len(active)
    actual_rho = n / nominal_n

    counts = option_counts(active)

    scenario = {
        "scenario_id": f"rho_{rho_target:.1f}",
        "seed": SEED + i,

        "target_rho": rho_target,
        "actual_rho": actual_rho,

        "nominal_assignments": nominal_n,
        "feasible_assignments": n,
        "removed_assignments": nominal_n - n,

        "globally_solvable": is_solvable(active),

        "min_options_per_task": min(counts.values()),
        "max_options_per_task": max(counts.values()),

        "task_option_counts": counts,

        "feasible": [
            {
                "model": model,
                "stage": stage,
                "resource_id": resource,
            }
            for model, stage, resource in active
        ],
    }

    outfile = OUTDIR / f"{scenario['scenario_id']}.json"
    outfile.write_text(json.dumps(scenario, indent=2))

    summary.append({
        "scenario_id": scenario["scenario_id"],
        "target_rho": rho_target,
        "actual_rho": actual_rho,
        "feasible": n,
        "removed": nominal_n - n,
        "solvable": scenario["globally_solvable"],
        "min_options": scenario["min_options_per_task"],
        "max_options": scenario["max_options_per_task"],
    })


# ============================================================
# Report
# ============================================================

print()
print("=" * 90)
print("EXPERIMENT 2 DEGRADATION LADDER")
print("=" * 90)

print(
    f"{'SCENARIO':12s}"
    f"{'TARGET':>10s}"
    f"{'ACTUAL':>10s}"
    f"{'FEASIBLE':>12s}"
    f"{'REMOVED':>10s}"
    f"{'MIN OPT':>10s}"
    f"{'MAX OPT':>10s}"
    f"{'SOLVABLE':>12s}"
)

print("-" * 90)

for x in summary:
    print(
        f"{x['scenario_id']:12s}"
        f"{x['target_rho']:10.3f}"
        f"{x['actual_rho']:10.3f}"
        f"{x['feasible']:12d}"
        f"{x['removed']:10d}"
        f"{x['min_options']:10d}"
        f"{x['max_options']:10d}"
        f"{str(x['solvable']):>12s}"
    )

print("=" * 90)

(PROJECT / "experiment2/generated/scenario_summary.json").write_text(
    json.dumps(summary, indent=2)
)

print("\nSaved scenarios to:")
print(OUTDIR)
