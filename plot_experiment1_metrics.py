#!/usr/bin/env python3
from pathlib import Path
import json, math
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path("/workspace/sc26_rebuttal")
ROOT = PROJECT / "outputs" / "experiment1_e2e"
HOST = "A100"
REP = "r01"
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODELS = ["random_forest","xgboost","lstm","yolo11n","yolov5s"]
LABELS = {"random_forest":"Random Forest","xgboost":"XGBoost","lstm":"LSTM","yolo11n":"YOLO11n","yolov5s":"YOLOv5s"}

def load_json(p):
    return json.loads(p.read_text()) if p.exists() else {}

def pick(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

rows=[]
hs=HOST.lower().replace(" ","")

for model in MODELS:
    wid=f"{model}_service_{hs}_{REP}"
    wp=ROOT/"workflows"/f"{wid}.json"
    if not wp.exists():
        continue

    w=load_json(wp)
    stage_results={}
    stage_metrics={}
    for s in ["update","evaluate","package","deploy","validate","reactivate"]:
        stage_results[s]=load_json(ROOT/"profiles"/"runs"/f"{wid}_{s}"/"result.json")
        stage_metrics[s]=load_json(ROOT/"metrics"/f"{wid}_{s}.json")

    eval_m=stage_metrics["evaluate"]
    update_r=stage_results["update"]
    package_m=stage_metrics["package"]

    train_sec=float(w["stage_runtime_sec"]["update"])
    eval_sec=float(w["stage_runtime_sec"]["evaluate"])
    e2e=float(w["end_to_end_response_sec"])

    n_eval=pick(eval_m,"n","n_eval_images")
    latency=(eval_sec/float(n_eval))*1000 if n_eval else math.nan
    throughput=float(n_eval)/eval_sec if n_eval else math.nan

    cpu_peak=max(float(r.get("process_cpu_peak_pct") or 0) for r in stage_results.values())
    sys_cpu_peak=max(float(r.get("system_cpu_peak_pct") or 0) for r in stage_results.values())

    mean_r2=pick(eval_m,"mean_r2")
    precision=pick(eval_m,"precision_mean")
    recall=pick(eval_m,"recall_mean")
    f1=pick(eval_m,"f1")
    map50=pick(eval_m,"map50")
    map5095=pick(eval_m,"map50_95")

    if mean_r2 is not None:
        quality=f"R²={float(mean_r2):.3f}"
    elif precision is not None:
        quality="\n".join([
            f"P={float(precision):.3f}",
            f"R={float(recall):.3f}" if recall is not None else "",
            f"F1={float(f1):.3f}" if f1 is not None else "",
            f"mAP50={float(map50):.3f}" if map50 is not None else "",
            f"mAP50-95={float(map5095):.3f}" if map5095 is not None else "",
        ]).strip()
    else:
        quality="—"

    rows.append({
        "Model":LABELS[model],
        "Training\nTime (s)":train_sec,
        "Inference /\nEval Time (s)":eval_sec,
        "Latency\n(ms/sample)":latency,
        "Throughput\n(samples/s)":throughput,
        "E2E Response\nTime (s)":e2e,
        "CPU Burst\nProc. Peak (%)":cpu_peak,
        "System CPU\nPeak (%)":sys_cpu_peak,
        "GPU Mean\nUpdate (%)":float(update_r.get("gpu_util_mean_pct") or 0),
        "GPU Peak\nWorkflow (%)":float(w.get("peak_gpu_util_pct") or 0),
        "Peak RAM\n(MB)":float(w.get("peak_process_rss_mb") or 0),
        "Peak VRAM\n(MB)":float(w.get("peak_vram_mb") or 0),
        "Model Size\n(MB)":pick(update_r,"artifact_size_mb"),
        "Package Size\n(MB)":pick(package_m,"package_size_mb"),
        "Disk Write\n(MB)":float(w.get("total_disk_write_mb") or 0),
        "Model Quality":quality,
    })

df=pd.DataFrame(rows)
csv_path=OUTDIR/f"experiment1_{hs}_professor_metrics.csv"
df.to_csv(csv_path,index=False)

import textwrap

disp = df.copy()

# Format the 'Model Quality' column with clean line wrapping
quality_col = [c for c in disp.columns if "Model Quality" in str(c)]
if quality_col:
    q_col = quality_col[0]

    def format_quality(x):
        if pd.isna(x) or x is None:
            return "—"
        flat_str = ", ".join(
            [line.strip() for line in str(x).split("\n") if line.strip()]
        )
        return textwrap.fill(flat_str, width=32)

    disp[q_col] = disp[q_col].map(format_quality)

for c in [
    "Training\nTime (s)",
    "Inference /\nEval Time (s)",
    "Latency\n(ms/sample)",
    "Throughput\n(samples/s)",
    "E2E Response\nTime (s)",
    "CPU Burst\nProc. Peak (%)",
    "System CPU\nPeak (%)",
    "GPU Mean\nUpdate (%)",
    "GPU Peak\nWorkflow (%)",
]:
    if c in disp.columns:
        disp[c] = disp[c].map(
            lambda x: "—" if pd.isna(x) else f"{float(x):.2f}"
        )

for c in [
    "Peak RAM\n(MB)",
    "Peak VRAM\n(MB)",
    "Model Size\n(MB)",
    "Package Size\n(MB)",
    "Disk Write\n(MB)",
]:
    if c in disp.columns:
        disp[c] = disp[c].map(
            lambda x: "—"
            if x is None or pd.isna(x)
            else f"{float(x):,.1f}"
        )

plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots(figsize=(28, 8.5))
ax.axis("off")

tbl = ax.table(
    cellText=disp.values,
    colLabels=disp.columns,
    loc="center",
    cellLoc="center",
    colLoc="center",
)

tbl.auto_set_font_size(False)
tbl.scale(1.0, 4.2)

nrows, ncols = disp.shape

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#E6E6E6")
        cell.set_text_props(weight="bold", fontsize=14)
        cell.set_linewidth(1.0)
    else:
        cell.set_facecolor("white")
        cell.set_fontsize(16)

for r in range(1, nrows + 1):
    tbl[(r, 0)].get_text().set_weight("bold")
    tbl[(r, 0)].get_text().set_ha("left")
    tbl[(r, 5)].get_text().set_weight("bold")
    # Set model quality text to 13.5pt for clean multi-line wrapping
    tbl[(r, ncols - 1)].get_text().set_fontsize(13.5)

# Allocate appropriate widths so names and multi-metric texts fit
for r in range(nrows + 1):
    tbl[(r, 0)].set_width(0.10)  # Fits "Random Forest", "YOLO11n", etc.
    tbl[(r, ncols - 1)].set_width(0.18)  # Fits wrapped Model Quality metrics

ax.set_title(
    f"Experiment 1 — System-Level End-to-End Metrics on NVIDIA {HOST}",
    fontsize=20,
    fontweight="bold",
    pad=10,
    y=0.96,
)

fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])

png=OUTDIR/f"experiment1_{hs}_professor_metrics_table.png"
pdf=OUTDIR/f"experiment1_{hs}_professor_metrics_table.pdf"
fig.savefig(png,dpi=300,bbox_inches="tight")
fig.savefig(pdf,bbox_inches="tight")
plt.close(fig)

print(csv_path)
print(png)
print(pdf)
