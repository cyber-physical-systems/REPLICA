#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

PROJECT = Path("/workspace/sc26_rebuttal")
ROOT = PROJECT / "experiment2" / "generated" / "final_benchmark_event_joint"
CSV = ROOT / "final_benchmark_master.csv"

OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV)

required = [
    "scheduler",
    "actual_rho",
    "makespan_sec",
    "scheduling_overhead_sec",
    "mean_resource_utilization",
    "success",
    "validation_pass",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

print(f"Loaded {len(df)} rows")
print(f"Scenarios: {df['scenario_id'].nunique()}")
print(f"Schedulers: {sorted(df['scheduler'].unique())}")

# ============================================================
# LARGE PUBLICATION STYLE (LABELS: 20pt, TICKS: 18pt, TITLES: 22pt)
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 17,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.titlesize": 24,
    }
)

LABELS = {
    "replica": "REPLICA",
    "heft": "HEFT",
    "cpsat": "CP-SAT",
    "easy": "EASY",
}

COLORS = {
    "replica": "#1f4e79",
    "heft": "#d95f02",
    "cpsat": "#2ca02c",
    "easy": "#7570b3",
}

MARKERS = {
    "replica": "o",
    "heft": "s",
    "cpsat": "^",
    "easy": "D",
}

LINESTYLES = {
    "replica": "-",
    "heft": "--",
    "cpsat": ":",
    "easy": "-.",
}

ORDER = ["replica", "cpsat", "heft", "easy"]

# ============================================================
# AGGREGATE
# ============================================================

agg = (
    df.groupby(["actual_rho", "scheduler"])
    .agg(
        median_makespan=("makespan_sec", "median"),
        q25_makespan=("makespan_sec", lambda x: x.quantile(0.25)),
        q75_makespan=("makespan_sec", lambda x: x.quantile(0.75)),
        median_overhead=("scheduling_overhead_sec", "median"),
        mean_utilization=("mean_resource_utilization", "mean"),
        success_rate=("success", "mean"),
        validation_rate=("validation_pass", "mean"),
    )
    .reset_index()
)

rhos = sorted(df["actual_rho"].unique(), reverse=True)

# ============================================================
# MONEY SHOT 1: MAKESPAN VS RHO (ZOOMED Y-AXIS, LARGE LABELS)
# ============================================================

fig, ax = plt.subplots(figsize=(12.5, 7.8))

for scheduler in ORDER:
    x = (
        agg[agg["scheduler"] == scheduler]
        .sort_values("actual_rho", ascending=False)
    )

    ax.plot(
        x["actual_rho"],
        x["median_makespan"],
        marker=MARKERS[scheduler],
        linestyle=LINESTYLES[scheduler],
        linewidth=3.5 if scheduler == "replica" else 2.5,
        markersize=10 if scheduler == "replica" else 8.5,
        label=LABELS[scheduler],
        color=COLORS[scheduler],
        zorder=6 if scheduler == "replica" else 4,
    )

    ax.fill_between(
        x["actual_rho"],
        x["q25_makespan"],
        x["q75_makespan"],
        color=COLORS[scheduler],
        alpha=0.15 if scheduler == "replica" else 0.06,
        linewidth=0,
        zorder=2,
    )

ax.set_xlabel(r"Available assignment fraction, $\rho$", fontsize=20, labelpad=12)
ax.set_ylabel("Workflow makespan (s)", fontsize=20, labelpad=12)
ax.set_title(
    "Scheduling Performance as Feasible Resource Assignments Contract",
    fontweight="bold",
    fontsize=22,
    pad=18,
)

# Dynamic Zoom on Y
min_y = agg["q25_makespan"].min()
max_y = agg["q75_makespan"].max()
ax.set_ylim(min_y * 0.90, max_y * 1.06)

ax.set_xlim(1.04, 0.28)
ax.set_xticks(rhos)
ax.set_xticklabels([f"{r:.2f}" if r != 0.333 else "0.33" for r in rhos], fontsize=18)
ax.tick_params(axis="y", labelsize=18)

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.grid(axis="x", linestyle=":", alpha=0.20)
ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=4, loc="upper left", fontsize=17)

# ============================================================
# SINGLE PAPER-LEVEL CALLOUT AT rho = 0.50
#
# Message:
# REPLICA remains close to CP-SAT while outperforming
# HEFT and EASY in the constrained regime.
# ============================================================

median_pivot = agg.pivot(
    index="actual_rho",
    columns="scheduler",
    values="median_makespan",
)

rho_callout = 0.50

if (
    rho_callout in median_pivot.index
    and all(
        name in median_pivot.columns
        for name in [
            "replica",
            "cpsat",
            "heft",
            "easy",
        ]
    )
):

    r_val = float(
        median_pivot.loc[
            rho_callout,
            "replica",
        ]
    )

    c_val = float(
        median_pivot.loc[
            rho_callout,
            "cpsat",
        ]
    )

    h_val = float(
        median_pivot.loc[
            rho_callout,
            "heft",
        ]
    )

    e_val = float(
        median_pivot.loc[
            rho_callout,
            "easy",
        ]
    )

    vs_heft = (
        100.0
        * (h_val - r_val)
        / h_val
    )

    vs_easy = (
        100.0
        * (e_val - r_val)
        / e_val
    )

    vs_cpsat = (
        100.0
        * (r_val - c_val)
        / c_val
    )

    callout = (
        r"At $\rho=0.50$:"
        "\n"
        f"REPLICA is {vs_heft:.1f}% lower than HEFT"
        "\n"
        f"and {vs_easy:.1f}% lower than EASY,"
        "\n"
        f"while within {abs(vs_cpsat):.1f}% of CP-SAT"
    )

    ax.annotate(
        callout,
        xy=(
            rho_callout,
            r_val,
        ),
        xytext=(
            0.68,
            r_val + 78,
        ),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.6,
            color="black",
        ),
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="0.65",
            alpha=0.95,
        ),
        zorder=10,
    )

    print()
    print(
        f"[rho=.50] REPLICA={r_val:.3f}s"
    )
    print(
        f"[rho=.50] CP-SAT={c_val:.3f}s"
    )
    print(
        f"[rho=.50] HEFT={h_val:.3f}s"
    )
    print(
        f"[rho=.50] EASY={e_val:.3f}s"
    )
    print(
        f"[rho=.50] REPLICA vs HEFT={vs_heft:+.2f}%"
    )
    print(
        f"[rho=.50] REPLICA vs EASY={vs_easy:+.2f}%"
    )
    print(
        f"[rho=.50] REPLICA gap to CP-SAT={vs_cpsat:+.2f}%"
    )


fig.tight_layout(rect=[0.02, 0.05, 0.99, 0.96])
p1 = OUTDIR / "experiment2_moneyshot1_makespan_vs_rho"
fig.savefig(p1.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(p1.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# MONEY SHOT 2: REPLICA RELATIVE TO HEFT BAR CHART
# ============================================================

pivot = agg.pivot(
    index="actual_rho",
    columns="scheduler",
    values="median_makespan",
)

comparison = pd.DataFrame(index=pivot.index)
comparison["replica_vs_heft_pct"] = (
    (pivot["heft"] - pivot["replica"]) / pivot["heft"] * 100.0
)
comparison = comparison.sort_index(ascending=False)

fig, ax = plt.subplots(figsize=(12.5, 7.5))
x = np.arange(len(comparison))
vals = comparison["replica_vs_heft_pct"].values

colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]

bars = ax.bar(
    x,
    vals,
    width=0.55,
    color=colors,
    edgecolor="black",
    linewidth=1.0,
)

for bar, value in zip(bars, vals):
    if value > 0:
        bar.set_hatch("//")
    elif value < 0:
        bar.set_hatch("..")

ax.axhline(0, color="black", linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels([f"{r:.2f}" if r != 0.333 else "0.33" for r in comparison.index], fontsize=18)
ax.tick_params(axis="y", labelsize=18)

ax.set_xlabel(r"Available assignment fraction, $\rho$", fontsize=20, labelpad=12)
ax.set_ylabel("REPLICA makespan improvement\nover HEFT (%)", fontsize=20, labelpad=12)
ax.set_title(
    "Where Symbolic Replanning Changes Scheduling Performance",
    fontweight="bold",
    fontsize=22,
    pad=18,
)

# Zoom in Y-axis with clean buffer for large bold labels
min_val, max_val = min(vals), max(vals)
ax.set_ylim(min_val - 2.0, max_val + 2.5)
ax.grid(axis="y", linestyle="--", alpha=0.35)

for bar, value in zip(bars, vals):
    if abs(value) >= 0.15:
        offset = 0.50 if value >= 0 else -0.60
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.1f}%",
            ha="center",
            va=va,
            fontsize=17,
            fontweight="bold",
        )


fig.tight_layout(rect=[0.02, 0.05, 0.99, 0.96])
p2 = OUTDIR / "experiment2_moneyshot2_replica_vs_heft"
fig.savefig(p2.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(p2.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# COMPACT PROFESSOR / PAPER TABLE
# ============================================================

makespan = (
    df.pivot_table(
        index="actual_rho",
        columns="scheduler",
        values="makespan_sec",
        aggfunc="median",
    ).sort_index(ascending=False)
)

rep = df[df["scheduler"] == "replica"].copy()
rep_summary = (
    rep.groupby("actual_rho")
    .agg(
        replica_util=("mean_resource_utilization", "mean"),
        replica_overhead=("scheduling_overhead_sec", "median"),
    )
    .sort_index(ascending=False)
)

rep_summary["symbolic_overhead"] = (
    rep.groupby("actual_rho")["symbolic_planning_overhead_sec"].median()
    if "symbolic_planning_overhead_sec" in rep.columns
    else np.nan
)
rep_summary["planner_calls"] = (
    rep.groupby("actual_rho")["planner_calls"].median()
    if "planner_calls" in rep.columns
    else np.nan
)

table = pd.DataFrame(index=makespan.index)
table[r"$\rho$"] = makespan.index
table["CP-SAT\nMakespan (s)"] = makespan["cpsat"].values
table["HEFT\nMakespan (s)"] = makespan["heft"].values
table["REPLICA\nMakespan (s)"] = makespan["replica"].values
table["EASY\nMakespan (s)"] = makespan["easy"].values
table["REPLICA vs.\nHEFT (%)"] = ((makespan["heft"] - makespan["replica"]) / makespan["heft"] * 100).values
table["REPLICA Mean\nUtilization"] = rep_summary["replica_util"].values
table["Planner\nCalls"] = rep_summary["planner_calls"].values
table["Symbolic\nOverhead (s)"] = rep_summary["symbolic_overhead"].values
table["Total\nOverhead (s)"] = rep_summary["replica_overhead"].values

csv_out = OUTDIR / "experiment2_final_benchmark_professor_metrics.csv"
table.to_csv(csv_out, index=False)

disp = table.copy()
disp[r"$\rho$"] = disp[r"$\rho$"].map(lambda x: f"{float(x):.2f}")
for c in ["CP-SAT\nMakespan (s)", "HEFT\nMakespan (s)", "REPLICA\nMakespan (s)", "EASY\nMakespan (s)"]:
    disp[c] = disp[c].map(lambda x: f"{float(x):.2f}")
disp["REPLICA vs.\nHEFT (%)"] = disp["REPLICA vs.\nHEFT (%)"].map(lambda x: f"{float(x):+.2f}%")
disp["REPLICA Mean\nUtilization"] = disp["REPLICA Mean\nUtilization"].map(lambda x: f"{float(x):.3f}")
disp["Planner\nCalls"] = disp["Planner\nCalls"].map(lambda x: "—" if pd.isna(x) else f"{int(round(float(x)))}")
for c in ["Symbolic\nOverhead (s)", "Total\nOverhead (s)"]:
    disp[c] = disp[c].map(lambda x: "—" if pd.isna(x) else f"{float(x):.3f}")

# Upgraded Table Visuals
fig, ax = plt.subplots(figsize=(20, 8.0))
ax.axis("off")

tbl = ax.table(
    cellText=disp.values,
    colLabels=disp.columns,
    loc="center",
    cellLoc="center",
    colLoc="center",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(15)
tbl.scale(1.0, 3.2)

nrows, ncols = disp.shape
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#E6E6E6")
        cell.set_text_props(weight="bold", fontsize=15.5)
        cell.set_linewidth(1.0)
    else:
        cell.set_facecolor("white")
        cell.set_fontsize(15)

for r in range(1, nrows + 1):
    tbl[(r, 0)].get_text().set_weight("bold")
    tbl[(r, 3)].get_text().set_weight("bold")

improve_col = list(disp.columns).index("REPLICA vs.\nHEFT (%)")
for r, value in enumerate(table["REPLICA vs.\nHEFT (%)"].values, start=1):
    if value > 0:
        tbl[(r, improve_col)].get_text().set_weight("bold")

ax.set_title(
    "Experiment 2 — Scheduling Performance Under Progressive Assignment Loss",
    fontsize=20,
    fontweight="bold",
    pad=18,
    y=0.96,
)


fig.tight_layout(rect=[0.01, 0.04, 0.99, 0.96])
p3 = OUTDIR / "experiment2_final_benchmark_professor_table"
fig.savefig(p3.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(p3.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)

print("=" * 80)
print("EXPERIMENT 2 VISUALIZATIONS COMPLETE (ENLARGED LABELS & TICKS)")
print("=" * 80)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 90)
print("EXPERIMENT 2 VISUALIZATION OUTPUTS")
print("=" * 90)

for f in [
    OUTDIR / "experiment2_moneyshot1_makespan_vs_rho.png",
    OUTDIR / "experiment2_moneyshot1_makespan_vs_rho.pdf",
    OUTDIR / "experiment2_moneyshot2_replica_vs_heft.png",
    OUTDIR / "experiment2_moneyshot2_replica_vs_heft.pdf",
    OUTDIR / "experiment2_final_benchmark_professor_table.png",
    OUTDIR / "experiment2_final_benchmark_professor_table.pdf",
    csv_out,
]:
    print(f)

print("=" * 90)
