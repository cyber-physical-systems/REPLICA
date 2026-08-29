#!/usr/bin/env python3
from pathlib import Path
import argparse, json, subprocess, sys, tarfile, tempfile, time

PROJECT=Path("/workspace/sc26_rebuttal")
WORK=PROJECT/"experiment1_workloads"

ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True,choices=["random_forest","xgboost","lstm","yolo11n","yolov5s"])
ap.add_argument("--package",required=True)
ap.add_argument("--metric-file",required=True)
ap.add_argument("--gpu-index",default="0")
a=ap.parse_args()

t0=time.perf_counter()
with tempfile.TemporaryDirectory(prefix="replica_validate_") as td:
    td=Path(td)
    with tarfile.open(a.package,"r") as tf: tf.extractall(td)
    arts=[p for p in td.iterdir() if p.is_file() and p.name!="manifest.json"]
    if not arts: raise RuntimeError("No model artifact in package")
    art=arts[0]; m=td/"m.json"
    if a.model=="random_forest":
        cmd=[sys.executable,WORK/"hvac_evaluate.py","--model-type","rf","--model-path",art,"--metric-file",m]
    elif a.model=="xgboost":
        cmd=[sys.executable,WORK/"hvac_evaluate.py","--model-type","xgb","--model-path",art,"--metric-file",m]
    elif a.model=="lstm":
        cmd=[sys.executable,WORK/"hvac_evaluate.py","--model-type","lstm","--model-path",art,"--metric-file",m]
    elif a.model=="yolo11n":
        cmd=[sys.executable,WORK/"yolo_evaluate.py","--weights",art,"--device",a.gpu_index,"--metric-file",m]
    else:
        cmd=[sys.executable,WORK/"yolov5_evaluate.py","--weights",art,"--device",a.gpu_index,"--metric-file",m]
    subprocess.run([str(x) for x in cmd],check=True)
    quality=json.loads(m.read_text())

out={"service_operation":"validate","validation_elapsed_sec":time.perf_counter()-t0,
     "validation_passed":True}
for k,v in quality.items():
    if isinstance(v,(str,int,float,bool)) or v is None: out[f"validation_{k}"]=v
Path(a.metric_file).parent.mkdir(parents=True,exist_ok=True)
Path(a.metric_file).write_text(json.dumps(out,indent=2))
