#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, time

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True)
ap.add_argument("--package",required=True)
ap.add_argument("--target-dir",required=True)
ap.add_argument("--metric-file",required=True)
ap.add_argument("--remote-host",default="")
a=ap.parse_args()

src=Path(a.package); target=Path(a.target_dir); target.mkdir(parents=True,exist_ok=True)
t0=time.perf_counter()
if a.remote_host:
    subprocess.run(["rsync","-a","--checksum",str(src),f"{a.remote_host}:{target}/"],check=True)
    mode="rsync_remote"; verified=None; deployed=str(target/src.name)
else:
    dst=target/src.name; shutil.copy2(src,dst)
    mode="local_staging"; verified=sha256(src)==sha256(dst); deployed=str(dst)
elapsed=time.perf_counter()-t0
size=src.stat().st_size
Path(a.metric_file).parent.mkdir(parents=True,exist_ok=True)
Path(a.metric_file).write_text(json.dumps({
    "service_operation":"deploy","deployment_mode":mode,
    "deployment_elapsed_sec":elapsed,"bytes_transferred":size,
    "transfer_mb":size/1024**2,
    "throughput_mb_s":(size/1024**2)/elapsed if elapsed else None,
    "deployed_path":deployed,"checksum_verified":verified},indent=2))
