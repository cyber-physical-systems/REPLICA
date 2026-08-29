#!/usr/bin/env python3
from pathlib import Path
import argparse, json, shutil

PROJECT=Path("/workspace/sc26_rebuttal")
DEFAULT_DATA=PROJECT/"outputs/experiment1/workload_inputs/yolo/data_frozen.yaml"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,choices=["yolo11n","yolov5s"])
    ap.add_argument("--weights",required=True)
    ap.add_argument("--data",default=str(DEFAULT_DATA))
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    ap.add_argument("--artifact",required=True)
    ap.add_argument("--metric-file",required=True)
    args=ap.parse_args()

    from ultralytics import YOLO
    model=YOLO(args.weights)
    result=model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, workers=8, patience=10, amp=True,
        deterministic=True, seed=42, plots=False, verbose=True,
        project=str(PROJECT/"outputs/experiment1/yolo_runs"),
        name=f"{args.model}_tmp", exist_ok=True
    )
    save_dir=Path(result.save_dir)
    src=save_dir/"weights/best.pt"
    if not src.exists(): src=save_dir/"weights/last.pt"
    if not src.exists(): raise FileNotFoundError("No YOLO checkpoint produced.")
    artifact=Path(args.artifact)
    artifact.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,artifact)
    Path(args.metric_file).parent.mkdir(parents=True,exist_ok=True)
    Path(args.metric_file).write_text(json.dumps({
        "model_family":args.model,
        "service_operation":"fine_tune_frozen_coco",
        "epochs_requested":args.epochs,
        "batch":args.batch,
        "imgsz":args.imgsz,
        "data":args.data,
        "source_checkpoint":args.weights
    },indent=2))

if __name__=="__main__":
    main()
