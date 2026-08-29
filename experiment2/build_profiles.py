from pathlib import Path
import csv
import json

from experiment2.common import ExecutionProfile


PROJECT = Path("/workspace/sc26_rebuttal")

# Change these only if your CSV names/locations differ.
PROFILE_FILES = {
    "A100": PROJECT / "outputs/experiment1_e2e/figures/experiment1_a100_professor_metrics.csv",
    "RTX4090": PROJECT / "outputs_rtx4090/experiment1_e2e/figures/experiment1_rtx4090_professor_metrics.csv",
    "RTX5090": PROJECT / "outputs_rtx5090/experiment1_e2e/figures/experiment1_rtx5090_professor_metrics.csv",
}

OUT = PROJECT / "experiment2/generated"
OUT.mkdir(parents=True, exist_ok=True)


def clean(s):
    return s.strip() if isinstance(s, str) else s


def load_table(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------
# First: inspect what we actually have.
# ------------------------------------------------------------

tables = {}

for resource_id, path in PROFILE_FILES.items():

    rows = load_table(path)
    tables[resource_id] = rows

    print("\n" + "=" * 70)
    print(resource_id)
    print(path)
    print("=" * 70)

    if rows:
        print("Columns:")
        for c in rows[0]:
            print("  ", repr(c))

        print(f"\nModels: {len(rows)}")
        for r in rows:
            print("  ", r.get("Model"))


# ------------------------------------------------------------
# We need STAGE-LEVEL profiles for scheduling.
#
# The professor tables are model summaries. The original workflow
# JSON/metric files contain update/evaluate/package/deploy/
# validate/reactivate stage runtimes.
#
# So locate the workflow JSONs and extract those runtimes.
# ------------------------------------------------------------

SEARCH_ROOTS = [
    PROJECT / "outputs",
    PROJECT / "outputs_rtx4090",
    PROJECT / "outputs_rtx5090",
    PROJECT / "outputs_test",
]

workflow_files = []

for root in SEARCH_ROOTS:
    if root.exists():
        workflow_files.extend(
            root.rglob("*_service_*_r01.json")
        )

# Only workflow summaries
workflow_files = [
    p for p in workflow_files
    if p.parent.name == "workflows"
]

print("\n" + "=" * 70)
print("WORKFLOW FILES")
print("=" * 70)

for p in sorted(workflow_files):
    print(p)


# ------------------------------------------------------------
# Build model-resource empirical resource envelopes from the
# Experiment-1 professor summary tables.
#
# These are model-level / resource-level measurements, not
# stage-specific measurements. We attach them to each stage
# profile as the empirical workload envelope for feasibility
# and contention reasoning.
# ------------------------------------------------------------

def norm_model_name(name):
    x = str(name).strip().lower()

    aliases = {
        "yolo11n": "yolo11n",
        "yolov5s": "yolov5s",
        "lstm": "lstm",
        "random forest": "random_forest",
        "random_forest": "random_forest",
        "rf": "random_forest",
        "xgboost": "xgboost",
        "xgb": "xgboost",
    }

    return aliases.get(x, x)


resource_envelopes = {}

for resource_id, rows in tables.items():

    for row in rows:

        model = norm_model_name(
            row.get("Model", "")
        )

        if not model:
            continue

        resource_envelopes[
            (
                model,
                resource_id,
            )
        ] = {
            # Experiment-1 reports CPU utilization as a percentage,
            # while ExecutionProfile expects a core count. Do not
            # mix these units; leave cpu_peak_cores unset for now.
            "cpu_peak_cores":
                0.0,

            "ram_peak_mb":
                float(
                    row.get(
                        "Peak RAM\n(MB)",
                        0.0,
                    )
                    or 0.0
                ),

            "gpu_mean_pct":
                float(
                    row.get(
                        "GPU Mean\nUpdate (%)",
                        0.0,
                    )
                    or 0.0
                ),

            "gpu_peak_pct":
                float(
                    row.get(
                        "GPU Peak\nWorkflow (%)",
                        0.0,
                    )
                    or 0.0
                ),

            "vram_peak_mb":
                float(
                    row.get(
                        "Peak VRAM\n(MB)",
                        0.0,
                    )
                    or 0.0
                ),

            "disk_write_mb":
                float(
                    row.get(
                        "Disk Write\n(MB)",
                        0.0,
                    )
                    or 0.0
                ),

            "artifact_size_mb":
                float(
                    row.get(
                        "Package Size\n(MB)",
                        0.0,
                    )
                    or 0.0
                ),
        }


profiles = []

for path in workflow_files:

    with path.open() as f:
        w = json.load(f)

    if not w.get("success", False):
        continue

    model = norm_model_name(
        w["model"]
    )

    resource_id = w["host"]

    envelope = resource_envelopes.get(
        (
            model,
            resource_id,
        ),
        {},
    )

    for stage, runtime in w["stage_runtime_sec"].items():

        profiles.append(
            ExecutionProfile(
                model=model,
                stage=stage,
                resource_id=resource_id,
                runtime_sec=float(runtime),

                cpu_peak_cores=float(
                    envelope.get(
                        "cpu_peak_cores",
                        0.0,
                    )
                ),

                ram_peak_mb=float(
                    envelope.get(
                        "ram_peak_mb",
                        0.0,
                    )
                ),

                gpu_mean_pct=float(
                    envelope.get(
                        "gpu_mean_pct",
                        0.0,
                    )
                ),

                gpu_peak_pct=float(
                    envelope.get(
                        "gpu_peak_pct",
                        0.0,
                    )
                ),

                vram_peak_mb=float(
                    envelope.get(
                        "vram_peak_mb",
                        0.0,
                    )
                ),

                disk_write_mb=float(
                    envelope.get(
                        "disk_write_mb",
                        0.0,
                    )
                ),

                artifact_size_mb=float(
                    envelope.get(
                        "artifact_size_mb",
                        0.0,
                    )
                ),
            )
        )


print("\n" + "=" * 70)
print("STAGE PROFILES")
print("=" * 70)

for p in sorted(
    profiles,
    key=lambda x: (x.model, x.stage, x.resource_id)
):
    print(
        f"{p.model:15s} "
        f"{p.stage:12s} "
        f"{p.resource_id:10s} "
        f"{p.runtime_sec:10.3f} s"
    )


# ------------------------------------------------------------
# Save portable JSON database
# ------------------------------------------------------------

records = [
    {
        "model": p.model,
        "stage": p.stage,
        "resource_id": p.resource_id,
        "runtime_sec": p.runtime_sec,

        "cpu_peak_cores":
            p.cpu_peak_cores,

        "ram_peak_mb":
            p.ram_peak_mb,

        "gpu_mean_pct":
            p.gpu_mean_pct,

        "gpu_peak_pct":
            p.gpu_peak_pct,

        "vram_peak_mb":
            p.vram_peak_mb,

        "disk_read_mb":
            p.disk_read_mb,

        "disk_write_mb":
            p.disk_write_mb,

        "artifact_size_mb":
            p.artifact_size_mb,
    }
    for p in profiles
]

outfile = OUT / "execution_profiles.json"

with outfile.open("w") as f:
    json.dump(records, f, indent=2)

print("\nSaved:")
print(outfile)


# ------------------------------------------------------------
# TASK × RESOURCE FEASIBILITY MATRIX
#
# At this point "feasible" means we possess a valid empirical
# execution profile for that model-stage-resource combination.
#
# Later we intersect this with dynamic availability/capacity.
# ------------------------------------------------------------

models = sorted(set(p.model for p in profiles))

stages = [
    "update",
    "evaluate",
    "package",
    "deploy",
    "validate",
    "reactivate",
]

resources = [
    "A100",
    "RTX4090",
    "RTX5090",
]

lookup = {
    (p.model, p.stage, p.resource_id)
    for p in profiles
}

print("\n")
print("=" * 100)
print("TASK × RESOURCE FEASIBILITY")
print("=" * 100)

print(
    f"{'MODEL':15s} "
    f"{'STAGE':12s} "
    + " ".join(f"{r:>10s}" for r in resources)
)

print("-" * 100)

total = 0
feasible = 0

for model in models:

    for stage in stages:

        vals = []

        for resource in resources:

            ok = (model, stage, resource) in lookup

            vals.append("YES" if ok else "---")

            total += 1

            if ok:
                feasible += 1

        print(
            f"{model:15s} "
            f"{stage:12s} "
            + " ".join(f"{x:>10s}" for x in vals)
        )


rho = feasible / total if total else 0.0

print("\n" + "=" * 100)
print(f"Feasible assignments : {feasible}")
print(f"Possible assignments : {total}")
print(f"Nominal rho           : {rho:.3f}")
print("=" * 100)
