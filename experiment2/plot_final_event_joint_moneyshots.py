from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT = Path("/workspace/sc26_rebuttal")

ROOT = (
    PROJECT
    / "experiment2"
    / "generated"
    / "final_benchmark_event_joint"
)

CSV = ROOT / "final_benchmark_master.csv"

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FINAL EVENT-JOINT BENCHMARK
# ============================================================

df = pd.read_csv(CSV)

required = {
    "scenario_id",
    "target_rho",
    "scheduler",
    "makespan_sec",
    "success",
    "validation_pass",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )

# Only successful and validated runs.
df = df[
    (df["success"] == True)
    & (df["validation_pass"] == True)
].copy()

expected_schedulers = {
    "replica",
    "cpsat",
    "heft",
    "easy",
}

actual_schedulers = set(
    df["scheduler"].astype(str).unique()
)

if actual_schedulers != expected_schedulers:
    raise RuntimeError(
        "Unexpected scheduler set.\n"
        f"Expected: {sorted(expected_schedulers)}\n"
        f"Found:    {sorted(actual_schedulers)}"
    )


# ============================================================
# VALIDATE FINAL DATASET
# ============================================================

counts = (
    df.groupby(
        ["target_rho", "scheduler"]
    )
    .size()
    .unstack()
)

if not (counts == 61).all().all():
    raise RuntimeError(
        "Expected exactly 61 scenarios per scheduler/rho.\n"
        f"{counts}"
    )

print("=" * 90)
print("FINAL EVENT-JOINT BENCHMARK")
print("=" * 90)
print("rows:", len(df))
print()
print(counts)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = (
    df.groupby(
        ["target_rho", "scheduler"]
    )["makespan_sec"]
    .agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        mean="mean",
    )
    .reset_index()
)

summary.to_csv(
    FIGDIR / "experiment2_makespan_vs_rho_data.csv",
    index=False,
)


# ============================================================
# PAPER ORDER / DISPLAY
# ============================================================

scheduler_order = [
    "replica",
    "cpsat",
    "heft",
    "easy",
]

display_names = {
    "replica": "REPLICA",
    "cpsat": "CP-SAT",
    "heft": "HEFT",
    "easy": "EASY",
}

markers = {
    "replica": "o",
    "cpsat": "^",
    "heft": "s",
    "easy": "D",
}

linestyles = {
    "replica": "-",
    "cpsat": ":",
    "heft": "--",
    "easy": "-.",
}


# ============================================================
# FIGURE 1: MEDIAN MAKESPAN + IQR
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.5, 5.4)
)

for scheduler in scheduler_order:

    x = (
        summary[
            summary["scheduler"] == scheduler
        ]
        .sort_values("target_rho")
    )

    rho = x["target_rho"].to_numpy()
    med = x["median"].to_numpy()
    q1 = x["q1"].to_numpy()
    q3 = x["q3"].to_numpy()

    line, = ax.plot(
        rho,
        med,
        marker=markers[scheduler],
        linestyle=linestyles[scheduler],
        linewidth=(
            3.0
            if scheduler == "replica"
            else 2.2
        ),
        markersize=(
            8
            if scheduler == "replica"
            else 7
        ),
        label=display_names[scheduler],
    )

    # Match the IQR shading to the corresponding line
    # without hard-coding colors.
    ax.fill_between(
        rho,
        q1,
        q3,
        alpha=0.10,
        color=line.get_color(),
    )


# ============================================================
# RHO = 0.50 HEADLINE RESULT
# ============================================================

rho50 = (
    summary[
        np.isclose(
            summary["target_rho"],
            0.50,
        )
    ]
    .set_index("scheduler")
)

replica = float(
    rho50.loc["replica", "median"]
)

heft = float(
    rho50.loc["heft", "median"]
)

easy = float(
    rho50.loc["easy", "median"]
)

cpsat = float(
    rho50.loc["cpsat", "median"]
)

vs_heft = (
    100.0
    * (heft - replica)
    / heft
)

vs_easy = (
    100.0
    * (easy - replica)
    / easy
)

cpsat_gap = (
    100.0
    * (replica - cpsat)
    / cpsat
)

print("\n" + "=" * 90)
print("RHO = 0.50 MEDIAN CHECK")
print("=" * 90)

print(f"REPLICA : {replica:.3f} s")
print(f"CP-SAT  : {cpsat:.3f} s")
print(f"HEFT    : {heft:.3f} s")
print(f"EASY    : {easy:.3f} s")

print()
print(
    f"REPLICA vs HEFT: "
    f"{vs_heft:.3f}% lower"
)

print(
    f"REPLICA vs EASY: "
    f"{vs_easy:.3f}% lower"
)

print(
    f"REPLICA vs CP-SAT: "
    f"{cpsat_gap:.3f}% higher"
)


# ============================================================
# ANNOTATION
# ============================================================

annotation = (
    r"At $\rho=0.50$:"
    "\n"
    f"{vs_heft:.1f}% lower than HEFT"
    "\n"
    f"{vs_easy:.1f}% lower than EASY"
    "\n"
    f"within {cpsat_gap:.1f}% of CP-SAT"
)

ax.annotate(
    annotation,
    xy=(0.50, replica),
    xytext=(0.66, 445),
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.2,
    ),
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="0.6",
    ),
)


# ============================================================
# AESTHETICS
# ============================================================

ax.set_title(
    "Scheduling Performance as Feasible Resource Assignments Contract",
    fontsize=16,
    fontweight="bold",
)

ax.set_xlabel(
    r"Available assignment fraction, $\rho$",
    fontsize=13,
)

ax.set_ylabel(
    "Workflow makespan (s)",
    fontsize=13,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    ncol=4,
    frameon=False,
    fontsize=11,
)

# Show high availability on left and increasing
# constraint toward the right.
ax.invert_xaxis()

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

png = (
    FIGDIR
    / "experiment2_makespan_vs_rho.png"
)

pdf = (
    FIGDIR
    / "experiment2_makespan_vs_rho.pdf"
)

fig.savefig(
    png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FINAL AUDIT
# ============================================================

print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)
print(png)
print(pdf)
print(
    FIGDIR
    / "experiment2_makespan_vs_rho_data.csv"
)

print("\nHeadline paper values:")
print(f"vs HEFT  = {vs_heft:.1f}%")
print(f"vs EASY  = {vs_easy:.1f}%")
print(f"CP-SAT gap = {cpsat_gap:.1f}%")

