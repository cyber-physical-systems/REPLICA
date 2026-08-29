#!/usr/bin/env python3
from pathlib import Path
import argparse, json, shutil, subprocess, sys

PROJECT = Path("/workspace/sc26_rebuttal")
YOLOV5_REPO = Path("/workspace/YOLOv5")
DEFAULT_DATA = PROJECT / "outputs/experiment1/workload_inputs/yolo/data_frozen.yaml"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--metric-file", required=True)
    args = ap.parse_args()

    run_name = "yolov5s_update_tmp"
    project_dir = PROJECT / "outputs/experiment1/yolov5_runs"

    cmd = [
        sys.executable, str(YOLOV5_REPO / "train.py"),
        "--weights", args.weights,
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch),
        "--imgsz", str(args.imgsz),
        "--device", str(args.device),
        "--workers", "8",
        "--project", str(project_dir),
        "--name", run_name,
        "--exist-ok",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(YOLOV5_REPO), check=True)

    run_dir = project_dir / run_name
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    src = best if best.exists() else last
    if not src.exists():
        raise FileNotFoundError("No YOLOv5 checkpoint produced.")

    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, artifact)

    Path(args.metric_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metric_file).write_text(json.dumps({
        "model_family": "yolov5s",
        "service_operation": "fine_tune_frozen_coco_native_yolov5",
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "data": args.data,
        "source_checkpoint": args.weights,
        "yolov5_repo": str(YOLOV5_REPO),
        "cache_policy": "none"
    }, indent=2))

if __name__ == "__main__":
    main()
