#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, tarfile, time

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True)
ap.add_argument("--artifact",required=True)
ap.add_argument("--package",required=True)
ap.add_argument("--metric-file",required=True)
a=ap.parse_args()

artifact=Path(a.artifact); package=Path(a.package)
package.parent.mkdir(parents=True,exist_ok=True)
t0=time.perf_counter()
manifest={"model":a.model,"artifact_name":artifact.name,
          "artifact_size_bytes":artifact.stat().st_size,
          "artifact_sha256":sha256(artifact)}
mp=package.with_suffix(".manifest.json")
mp.write_text(json.dumps(manifest,indent=2))
with tarfile.open(package, "w") as tf:
    tf.add(artifact,arcname=artifact.name)
    tf.add(mp,arcname="manifest.json")
elapsed=time.perf_counter()-t0
Path(a.metric_file).parent.mkdir(parents=True,exist_ok=True)
Path(a.metric_file).write_text(json.dumps({
    "service_operation":"package",
    "package_elapsed_sec":elapsed,
    "input_artifact_size_mb":artifact.stat().st_size/1024**2,
    "package_size_mb":package.stat().st_size/1024**2,
    "artifact_sha256":manifest["artifact_sha256"],
    "package_sha256":sha256(package)},indent=2))
