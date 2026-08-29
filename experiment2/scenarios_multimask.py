from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from collections import Counter


PROJECT = Path("/workspace/sc26_rebuttal")

PROFILE_FILE = (
    PROJECT
    / "experiment2/generated/execution_profiles.json"
)

OUTDIR = (
    PROJECT
    / "experiment2/generated/scenarios_multimask"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

BASE_SEED = 42000
MASKS_PER_LEVEL = 20

# Use ACTUAL feasible densities in the final experiment.
#
# 30 model-stage tasks × minimum 1 resource/task
# ------------------------------------------------------------
# 90 -> 1.000
# 72 -> 0.800
# 54 -> 0.600
# 36 -> 0.400
# 30 -> 0.333...
#
TARGET_COUNTS = {
    "rho_1.000": 90,
    "rho_0.800": 72,
    "rho_0.600": 54,
    "rho_0.400": 36,
    "rho_0.333": 30,
}


# ============================================================
# LOAD NOMINAL EXECUTION PROFILES
# ============================================================

profiles = json.loads(PROFILE_FILE.read_text())

assignments = sorted({
    (
        p["model"],
        p["stage"],
        p["resource_id"],
    )
    for p in profiles
})

models = sorted({
    model
    for model, _, _ in assignments
})

stages = [
    "update",
    "evaluate",
    "package",
    "deploy",
    "validate",
    "reactivate",
]

resources = sorted({
    resource
    for _, _, resource in assignments
})

nominal_n = len(assignments)


print("=" * 80)
print("MULTI-MASK EXPERIMENT 2 SCENARIO GENERATOR")
print("=" * 80)

print("Models:             ", len(models))
print("Stages/model:       ", len(stages))
print("Resources:          ", resources)
print("Nominal assignments:", nominal_n)


# ============================================================
# VALIDITY
#
# Every model-stage task must retain >= 1 feasible resource.
# ============================================================

def option_counts(active):
    active = set(active)

    counts = {}

    for model in models:
        for stage in stages:

            counts[(model, stage)] = sum(
                (model, stage, resource) in active
                for resource in resources
            )

    return counts


def is_solvable(active):
    counts = option_counts(active)

    return all(
        count >= 1
        for count in counts.values()
    )


# ============================================================
# GENERATE ONE RANDOM MASK
#
# Randomly remove feasible assignment edges while preserving
# at least one resource option for every task.
# ============================================================

def generate_mask(
    target_n: int,
    seed: int,
    max_restarts: int = 100,
):

    if target_n < len(models) * len(stages):
        raise ValueError(
            f"Target {target_n} is below global feasibility "
            f"boundary {len(models) * len(stages)}."
        )

    if target_n > nominal_n:
        raise ValueError(
            f"Target {target_n} exceeds nominal {nominal_n}."
        )

    # Nominal case is deterministic.
    if target_n == nominal_n:
        return sorted(assignments)

    # Multiple restarts prevent a bad random deletion order
    # from trapping us above the requested assignment count.
    for restart in range(max_restarts):

        rng = random.Random(
            seed + restart * 1_000_003
        )

        active = set(assignments)

        while len(active) > target_n:

            removable = []

            counts = option_counts(active)

            for assignment in active:

                model, stage, resource = assignment

                # Never delete the final option for a task.
                if counts[(model, stage)] <= 1:
                    continue

                removable.append(assignment)

            if not removable:
                break

            chosen = rng.choice(removable)
            active.remove(chosen)

        if (
            len(active) == target_n
            and is_solvable(active)
        ):
            return sorted(active)

    raise RuntimeError(
        f"Could not generate solvable mask with "
        f"{target_n} assignments after "
        f"{max_restarts} restarts."
    )


# ============================================================
# DESCRIPTIVE STATISTICS FOR EACH MASK
# ============================================================

def mask_stats(active):

    active = set(active)

    resource_counts = Counter(
        resource
        for _, _, resource in active
    )

    counts = option_counts(active)
    values = list(counts.values())

    n_tasks_with_1 = sum(v == 1 for v in values)
    n_tasks_with_2 = sum(v == 2 for v in values)
    n_tasks_with_3 = sum(v == 3 for v in values)

    return {
        "min_options_per_task": min(values),
        "max_options_per_task": max(values),

        "tasks_with_1_option": n_tasks_with_1,
        "tasks_with_2_options": n_tasks_with_2,
        "tasks_with_3_options": n_tasks_with_3,

        "a100_assignments": resource_counts.get(
            "A100", 0
        ),

        "rtx4090_assignments": resource_counts.get(
            "RTX4090", 0
        ),

        "rtx5090_assignments": resource_counts.get(
            "RTX5090", 0
        ),
    }


# ============================================================
# SAVE ONE SCENARIO
# ============================================================

def save_scenario(
    level_name,
    mask_index,
    seed,
    active,
):

    n = len(active)
    rho = n / nominal_n

    stats = mask_stats(active)

    if level_name == "rho_1.000":
        scenario_id = "rho_1.000_nominal"
    else:
        scenario_id = (
            f"{level_name}_mask_{mask_index:02d}"
        )

    counts = option_counts(active)

    scenario = {
        "scenario_id": scenario_id,
        "level": level_name,

        "mask_index": mask_index,
        "seed": seed,

        "nominal_assignments": nominal_n,
        "feasible_assignments": n,
        "removed_assignments": nominal_n - n,

        "actual_rho": rho,

        "globally_solvable": is_solvable(active),

        **stats,

        "task_option_counts": {
            f"{model}:{stage}": count
            for (model, stage), count
            in sorted(counts.items())
        },

        "feasible": [
            {
                "model": model,
                "stage": stage,
                "resource_id": resource,
            }
            for model, stage, resource
            in active
        ],
    }

    outfile = OUTDIR / f"{scenario_id}.json"

    outfile.write_text(
        json.dumps(
            scenario,
            indent=2,
        )
    )

    return scenario


# ============================================================
# GENERATE FINAL EXPERIMENT SET
# ============================================================

manifest = []


# ------------------------------------------------------------
# Nominal
#
# Only one unique mask exists at rho=1.0.
# ------------------------------------------------------------

nominal_active = generate_mask(
    target_n=TARGET_COUNTS["rho_1.000"],
    seed=BASE_SEED,
)

manifest.append(
    save_scenario(
        level_name="rho_1.000",
        mask_index=0,
        seed=BASE_SEED,
        active=nominal_active,
    )
)


# ------------------------------------------------------------
# Constrained levels
# ------------------------------------------------------------

levels = [
    "rho_0.800",
    "rho_0.600",
    "rho_0.400",
    "rho_0.333",
]

for level_idx, level_name in enumerate(
    levels,
    start=1,
):

    target_n = TARGET_COUNTS[level_name]

    seen_masks = set()

    for mask_index in range(
        1,
        MASKS_PER_LEVEL + 1,
    ):

        # Make seed reproducible and unique.
        seed = (
            BASE_SEED
            + level_idx * 1000
            + mask_index
        )

        # Ensure masks within a rho level are actually unique.
        attempt = 0

        while True:

            active = generate_mask(
                target_n=target_n,
                seed=seed + attempt * 100_000,
            )

            signature = tuple(active)

            if signature not in seen_masks:
                seen_masks.add(signature)
                break

            attempt += 1

            if attempt > 100:
                raise RuntimeError(
                    f"Unable to generate unique "
                    f"mask for {level_name}"
                )

        scenario = save_scenario(
            level_name=level_name,
            mask_index=mask_index,
            seed=seed,
            active=active,
        )

        manifest.append(scenario)


# ============================================================
# SAVE MANIFEST JSON
# ============================================================

manifest_json = (
    PROJECT
    / "experiment2/generated/"
      "scenario_multimask_manifest.json"
)

manifest_json.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
)


# ============================================================
# SAVE MANIFEST CSV
# ============================================================

manifest_csv = (
    PROJECT
    / "experiment2/generated/"
      "scenario_multimask_manifest.csv"
)

fields = [
    "scenario_id",
    "level",
    "mask_index",
    "seed",
    "actual_rho",
    "nominal_assignments",
    "feasible_assignments",
    "removed_assignments",
    "globally_solvable",
    "min_options_per_task",
    "max_options_per_task",
    "tasks_with_1_option",
    "tasks_with_2_options",
    "tasks_with_3_options",
    "a100_assignments",
    "rtx4090_assignments",
    "rtx5090_assignments",
]

with manifest_csv.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for s in manifest:

        writer.writerow({
            key: s[key]
            for key in fields
        })


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 105)
print("FINAL EXPERIMENT 2 MULTI-MASK SCENARIOS")
print("=" * 105)

print(
    f"{'LEVEL':12s}"
    f"{'MASKS':>8s}"
    f"{'RHO':>10s}"
    f"{'N':>8s}"
    f"{'1-OPT AVG':>12s}"
    f"{'A100 AVG':>12s}"
    f"{'4090 AVG':>12s}"
    f"{'5090 AVG':>12s}"
)

print("-" * 105)

for level in TARGET_COUNTS:

    rows = [
        s
        for s in manifest
        if s["level"] == level
    ]

    def avg(key):
        return (
            sum(r[key] for r in rows)
            / len(rows)
        )

    print(
        f"{level:12s}"
        f"{len(rows):8d}"
        f"{rows[0]['actual_rho']:10.3f}"
        f"{rows[0]['feasible_assignments']:8d}"
        f"{avg('tasks_with_1_option'):12.2f}"
        f"{avg('a100_assignments'):12.2f}"
        f"{avg('rtx4090_assignments'):12.2f}"
        f"{avg('rtx5090_assignments'):12.2f}"
    )

print("=" * 105)

print()
print("Total scenarios:", len(manifest))
print()
print("Scenario directory:")
print(OUTDIR)
print()
print("Manifest JSON:")
print(manifest_json)
print()
print("Manifest CSV:")
print(manifest_csv)
