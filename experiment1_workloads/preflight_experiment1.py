#!/usr/bin/env python3
from pathlib import Path
import sys

PROJECT = Path("/workspace/sc26_rebuttal")
YOLOV5 = Path("/workspace/YOLOv5")

checks = [
    PROJECT / "outputs/experiment1/workload_inputs/hvac_tabular_train_600000.npz",
    PROJECT / "outputs/experiment1/workload_inputs/hvac_lstm_train_600000.npz",
    PROJECT / "outputs/models/yolo11n.pt",
    PROJECT / "outputs/models/yolov5s.pt",
    PROJECT / "data/coco/images/train2017",
    PROJECT / "data/coco/images/val2017",
    YOLOV5 / "hubconf.py",
    YOLOV5 / "train.py",
    YOLOV5 / "val.py",
]
bad = False
for p in checks:
    ok = p.exists()
    print(("OK     " if ok else "MISSING"), p)
    bad |= not ok

try:
    import ultralytics, torch
    print("ultralytics", ultralytics.__version__)
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as e:
    print("software error:", repr(e))
    bad = True

if not bad:
    from ultralytics import YOLO
    try:
        YOLO(str(PROJECT / "outputs/models/yolo11n.pt"))
        print("YOLO11n LOAD OK")
    except Exception as e:
        print("YOLO11n LOAD FAILED", repr(e))
        bad = True

    try:
        import torch
        m = torch.hub.load(
            str(YOLOV5),
            "custom",
            path=str(PROJECT / "outputs/models/yolov5s.pt"),
            source="local",
            force_reload=False,
        )
        print("YOLOv5s LOAD OK", type(m))
    except Exception as e:
        print("YOLOv5s LOAD FAILED", repr(e))
        bad = True

sys.exit(1 if bad else 0)
