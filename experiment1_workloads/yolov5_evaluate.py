#!/usr/bin/env python3
from pathlib import Path
import argparse, json, subprocess, sys, time, re

PROJECT = Path("/workspace/sc26_rebuttal")
YOLOV5_REPO = Path("/workspace/YOLOv5")
DEFAULT_DATA = PROJECT / "outputs/experiment1/workload_inputs/yolo/data_frozen.yaml"
DEFAULT_VAL_LIST = PROJECT / "outputs/experiment1/workload_inputs/yolo/val_frozen.txt"
ANSI_RE = re.compile(r"\\x1b\\[[0-9;]*m")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--metric-file", required=True)
    args = ap.parse_args()

    cmd = [
        sys.executable, str(YOLOV5_REPO / "val.py"),
        "--weights", args.weights,
        "--data", args.data,
        "--imgsz", str(args.imgsz),
        "--batch-size", str(args.batch),
        "--device", str(args.device),
        "--workers", "8",
        "--project", str(PROJECT / "outputs/experiment1/yolov5_eval_runs"),
        "--name", "yolov5s_eval_tmp",
        "--exist-ok",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(YOLOV5_REPO), check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    clean = [ANSI_RE.sub("", x).strip() for x in proc.stdout.splitlines()]
    precision = recall = map50 = map5095 = None
    summary_line = None

    for line in reversed(clean):
        parts = line.split()
        if parts and parts[0] == "all" and len(parts) >= 7:
            try:
                precision = float(parts[-4])
                recall = float(parts[-3])
                map50 = float(parts[-2])
                map5095 = float(parts[-1])
                summary_line = line
                break
            except ValueError:
                pass

    if None in (precision, recall, map50, map5095):
        raise RuntimeError("Could not parse YOLOv5 validation metrics.\n" + "\n".join(clean[-50:]))

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "f1": f1,
        "model_family": "yolov5s",
        "evaluation_elapsed_sec": elapsed,
        "precision_mean": precision,
        "recall_mean": recall,
        "map50": map50,
        "map50_95": map5095,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "n_eval_images": sum(1 for x in DEFAULT_VAL_LIST.read_text().splitlines() if x.strip()) if DEFAULT_VAL_LIST.exists() else None,
        "evaluation_backend": "native_yolov5_labels",
        "summary_line": summary_line
    }
    Path(args.metric_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metric_file).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
