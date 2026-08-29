#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import shutil
import tarfile
import time


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--package", required=True)
ap.add_argument("--active-dir", required=True)
ap.add_argument("--metric-file", required=True)
args = ap.parse_args()


package = Path(args.package)
active_dir = Path(args.active_dir)
metric_file = Path(args.metric_file)

active_dir.mkdir(parents=True, exist_ok=True)
metric_file.parent.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()

# ------------------------------------------------------------
# Create/replace active model directory
# ------------------------------------------------------------

version_dir = active_dir / f"{args.model}_active"

if version_dir.exists():
    shutil.rmtree(version_dir)

version_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Extract UNCOMPRESSED TAR package
# ------------------------------------------------------------

with tarfile.open(package, "r") as tf:
    tf.extractall(version_dir)


# ------------------------------------------------------------
# Locate deployed model artifact
# ------------------------------------------------------------

artifacts = [
    p
    for p in version_dir.iterdir()
    if p.is_file() and p.name != "manifest.json"
]

if not artifacts:
    raise RuntimeError(
        f"No model artifact found after extracting {package}"
    )

artifact = artifacts[0]


# ------------------------------------------------------------
# Activate model
# ------------------------------------------------------------

pointer = active_dir / f"{args.model}_CURRENT.txt"
pointer.write_text(str(artifact) + "\n")


# ------------------------------------------------------------
# Readiness check
# ------------------------------------------------------------

ready = pointer.exists() and artifact.exists()

elapsed = time.perf_counter() - t0


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

result = {
    "model_family": args.model,
    "service_operation": "reactivate",
    "reactivation_elapsed_sec": elapsed,
    "readiness_passed": ready,
    "active_artifact": str(artifact),
}

metric_file.write_text(
    json.dumps(result, indent=2)
)

print(json.dumps(result, indent=2))


if not ready:
    raise RuntimeError("Reactivation readiness check failed.")
