from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT = Path("outputs/experiment1/experiment1_analysis/a100/official_runs.csv")
OUTDIR = Path("outputs/experiment1/analysis/money_shot")
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "random_forest",
    "xgboost",
    "lstm",
    "yolo11n",
    "yolov5s",
]

MODEL_LABELS = {
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "yolo11n": "YOLO11n",
    "yolov5s": "YOLOv5s",
}

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

# Keep successful official runs only
if "success" in df.columns:
    if df["success"].dtype == object:
        df = df[df["success"].astype(str).str.lower() == "true"]
    else:
        df = df[df["success"] == True]

# Aggregate the 5 repetitions
summary = (
    df.groupby(["model", "task"], as_index=False)
    .agg(
        runtime_mean=("runtime_sec", "mean"),
        runtime_std=("runtime_sec", "std"),
        gpu_mean=("gpu_util_mean_pct", "mean"),
        gpu_mean_std=("gpu_util_mean_pct", "std"),
        gpu_peak=("gpu_util_peak_pct", "mean"),
        vram_peak=("gpu_mem_used_peak_mb", "mean"),
        ram_peak=("process_rss_peak_mb", "mean"),
        n=("runtime_sec", "size"),
    )
)

summary.to_csv(OUTDIR / "resource_signature_summary.csv", index=False)

print(summary.to_string(index=False))


# ============================================================
# FIGURE 1
# RUNTIME vs GPU UTILIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

markers = {
    "update": "o",
    "evaluate": "s",
}

for task in ["update", "evaluate"]:

    subset = summary[summary["task"] == task]

    ax.scatter(
        subset["runtime_mean"],
        subset["gpu_mean"],
        marker=markers[task],
        s=70,
        label=task.capitalize(),
        edgecolors="black",
        linewidths=0.8,
    )

    for _, row in subset.iterrows():

        label = MODEL_LABELS.get(row["model"], row["model"])

        ax.annotate(
            label,
            (row["runtime_mean"], row["gpu_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

ax.set_xlabel("Mean task runtime (s)")
ax.set_ylabel("Mean GPU utilization (%)")

ax.set_title(
    "Experiment 1: Runtime and GPU Demand by Workload"
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.5,
)

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    OUTDIR / "experiment1_runtime_gpu_signature.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTDIR / "experiment1_runtime_gpu_signature.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# RAM AND VRAM
# ============================================================

update = (
    summary[summary["task"] == "update"]
    .set_index("model")
    .reindex(MODEL_ORDER)
    .reset_index()
)

x = range(len(update))
width = 0.36

fig, ax = plt.subplots(figsize=(8, 5.5))

x1 = [i - width / 2 for i in x]
x2 = [i + width / 2 for i in x]

ax.bar(
    x1,
    update["ram_peak"],
    width=width,
    label="Peak process RAM",
)

ax.bar(
    x2,
    update["vram_peak"],
    width=width,
    label="Peak GPU memory",
)

ax.set_xticks(list(x))

ax.set_xticklabels(
    [MODEL_LABELS[m] for m in update["model"]]
)

ax.set_ylabel("Memory (MB)")

ax.set_title(
    "Experiment 1: Memory Demand During Model Update"
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.5,
)

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    OUTDIR / "experiment1_update_memory.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTDIR / "experiment1_update_memory.pdf",
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Saved figures to:")
print(OUTDIR)