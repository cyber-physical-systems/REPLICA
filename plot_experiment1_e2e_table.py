from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/workspace/sc26_rebuttal")
WORKFLOW_DIR = ROOT / "outputs/experiment1_e2e/workflows"
OUT_DIR = ROOT / "outputs/experiment1_e2e/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Read completed A100 workflows
# ------------------------------------------------------------
rows = []

for path in sorted(WORKFLOW_DIR.glob("*_service_a100_r01.json")):
    with open(path) as f:
        d = json.load(f)

    stages = d["stage_runtime_sec"]

    rows.append({
        "Model": d["model"],
        "E2E (s)": d["end_to_end_response_sec"],
        "Update": stages.get("update", 0),
        "Evaluate": stages.get("evaluate", 0),
        "Package": stages.get("package", 0),
        "Deploy": stages.get("deploy", 0),
        "Validate": stages.get("validate", 0),
        "Reactivate": stages.get("reactivate", 0),
        "Peak RAM\n(MB)": d["peak_process_rss_mb"],
        "Peak VRAM\n(MB)": d["peak_vram_mb"],
        "Disk Write\n(MB)": d["total_disk_write_mb"],
    })

df = pd.DataFrame(rows)

# Nice names
name_map = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "yolo11n": "YOLO11n",
    "yolov5s": "YOLOv5s",
}

df["Model"] = df["Model"].replace(name_map)

# ------------------------------------------------------------
# Format values
# ------------------------------------------------------------
display = df.copy()

for col in [
    "E2E (s)",
    "Update",
    "Evaluate",
    "Package",
    "Deploy",
    "Validate",
    "Reactivate",
]:
    display[col] = display[col].map(lambda x: f"{x:.2f}")

for col in ["Peak RAM\n(MB)", "Peak VRAM\n(MB)", "Disk Write\n(MB)"]:
    display[col] = display[col].map(lambda x: f"{x:,.0f}")

# ------------------------------------------------------------
# Classic publication / gnuplot-ish appearance
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(16, 4.6))
ax.axis("off")

table = ax.table(
    cellText=display.values,
    colLabels=display.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
nrows, ncols = display.shape

for (r, c), cell in table.get_celld().items():

    cell.set_edgecolor("black")
    cell.set_linewidth(0.55)

    if r == 0:
        # Header
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#E8E8E8")
        cell.set_linewidth(0.9)

    else:
        cell.set_facecolor("white")

# Left align model names
for r in range(1, nrows + 1):
    table[(r, 0)].get_text().set_ha("left")
    table[(r, 0)].get_text().set_weight("bold")

# Slightly emphasize E2E
for r in range(1, nrows + 1):
    table[(r, 1)].get_text().set_weight("bold")

# Wider model column
for r in range(0, nrows + 1):
    table[(r, 0)].set_width(0.13)

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
ax.set_title(
    "Experiment 1 — End-to-End Model Service Performance on NVIDIA A100",
    fontsize=15,
    fontweight="bold",
    pad=18,
)

fig.text(
    0.5,
    0.025,
    "Stage runtimes are in seconds. Values shown for repetition r01.",
    ha="center",
    fontsize=9,
    style="italic",
)

plt.tight_layout(rect=[0.01, 0.06, 0.99, 0.94])

png = OUT_DIR / "experiment1_a100_e2e_table.png"
pdf = OUT_DIR / "experiment1_a100_e2e_table.pdf"

plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")

print("\nSaved:")
print(png)
print(pdf)

print("\nTable data:")
print(display.to_string(index=False))