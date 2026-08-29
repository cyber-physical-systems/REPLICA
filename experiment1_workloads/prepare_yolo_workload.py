#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, random

PROJECT = Path("/workspace/sc26_rebuttal")
COCO = PROJECT / "data" / "coco"
OUT = PROJECT / "outputs" / "experiment1" / "workload_inputs" / "yolo"
SEED = 42

COCO_NAMES = [
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
"elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
"skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
"tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
"sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
"potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
"microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush"
]

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def normalize(s):
    s=s.strip()
    if s.startswith("./"): s=s[2:]
    p=Path(s)
    return p if p.is_absolute() else COCO/p

def choose(items,n,seed):
    items=list(items)
    if n>=len(items): return sorted(items)
    return sorted(random.Random(seed).sample(items,n))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-images",type=int,default=5000)
    ap.add_argument("--val-images",type=int,default=1000)
    args=ap.parse_args()

    train=[normalize(x) for x in (COCO/"train2017.txt").read_text().splitlines() if x.strip()]
    val=[normalize(x) for x in (COCO/"val2017.txt").read_text().splitlines() if x.strip()]
    train=[p for p in train if p.exists()]
    val=[p for p in val if p.exists()]
    if not train: raise RuntimeError("No COCO train images found.")
    if not val: raise RuntimeError("No COCO val images found.")

    train=choose(train,args.train_images,SEED)
    val=choose(val,args.val_images,SEED+1)

    OUT.mkdir(parents=True,exist_ok=True)
    train_list=OUT/"train_frozen.txt"
    val_list=OUT/"val_frozen.txt"
    train_list.write_text("\n".join(map(str,train))+"\n")
    val_list.write_text("\n".join(map(str,val))+"\n")

    yaml_lines=[
        f"path: {COCO}",
        f"train: {train_list}",
        f"val: {val_list}",
        "names:"
    ] + [f"  {i}: {name}" for i,name in enumerate(COCO_NAMES)]
    data_yaml=OUT/"data_frozen.yaml"
    data_yaml.write_text("\n".join(yaml_lines)+"\n")

    manifest={
        "seed":SEED,
        "train_images":len(train),
        "val_images":len(val),
        "train_list_sha256":sha256(train_list),
        "val_list_sha256":sha256(val_list),
        "data_yaml":str(data_yaml),
    }
    (OUT/"yolo_workload_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()
