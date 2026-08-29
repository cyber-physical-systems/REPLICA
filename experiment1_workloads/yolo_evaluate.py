#!/usr/bin/env python3
from pathlib import Path
import argparse, json, time

PROJECT=Path("/workspace/sc26_rebuttal")
DEFAULT_DATA=PROJECT/"outputs/experiment1/workload_inputs/yolo/data_frozen.yaml"
DEFAULT_VAL=PROJECT/"outputs/experiment1/workload_inputs/yolo/val_frozen.txt"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights",required=True)
    ap.add_argument("--data",default=str(DEFAULT_DATA))
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--device",default="0")
    ap.add_argument("--metric-file",required=True)
    args=ap.parse_args()

    from ultralytics import YOLO
    model=YOLO(args.weights)
    t0=time.perf_counter()
    r=model.val(data=args.data,imgsz=args.imgsz,batch=args.batch,
                device=args.device,plots=False,verbose=False)
    elapsed=time.perf_counter()-t0

    precision = float(r.box.mp)
    recall = float(r.box.mr)

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics={
        "model_family":"YOLO",
        "evaluation_elapsed_sec":elapsed,
        "map50":float(r.box.map50),
        "map50_95":float(r.box.map),
        "precision_mean":precision,
        "recall_mean":recall,
        "f1":f1,
        "imgsz":args.imgsz,
        "batch":args.batch,
        "n_eval_images":sum(1 for x in DEFAULT_VAL.read_text().splitlines() if x.strip()) if DEFAULT_VAL.exists() else None
    }
    Path(args.metric_file).parent.mkdir(parents=True,exist_ok=True)
    Path(args.metric_file).write_text(json.dumps(metrics,indent=2))

if __name__=="__main__":
    main()
