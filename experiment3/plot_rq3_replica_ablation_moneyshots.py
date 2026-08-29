#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT / OUTPUT
# ============================================================

ROOT = Path("/workspace/sc26_rebuttal")

CSV = (
    ROOT
    / "experiment3/generated/rq3_replica_ablation"
    / "rq3_replica_ablation_results.csv"
)

OUTDIR = (
    ROOT
    / "experiment3/generated/rq3_replica_ablation"
    / "figures"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_CSV = (
    OUTDIR
    / "replica_framework_ablation_table_data.csv"
)

TABLE_TEX = (
    OUTDIR
    / "replica_framework_ablation_table.tex"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(CSV)

required = {
    "case_id",
    "variant",
    "recovery_success",
    "trusted_service_restore_sec",
    "planning_overhead_sec",
    "replanned",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# VARIANTS USED IN PAPER
# ============================================================

paper_variants = [
    "full_replica",
    "no_state_reconstruction",
    "no_strategy_selection",
    "no_replanning",
]

pretty = {
    "full_replica":
        "Full REPLICA",

    "no_state_reconstruction":
        "No State\nReconstruction",

    "no_strategy_selection":
        "No Strategy\nSelection",

    "no_replanning":
        "No Replanning",
}


# ============================================================
# COMPUTE SUMMARY DIRECTLY FROM RAW RESULTS
# ============================================================

rows = []

for variant in paper_variants:

    x = df[
        df["variant"] == variant
    ].copy()

    if x.empty:
        raise RuntimeError(
            f"No rows found for variant: {variant}"
        )

    success_rate = (
        100.0
        * x["recovery_success"]
        .astype(bool)
        .mean()
    )

    successful = x[
        x["recovery_success"]
        .astype(bool)
    ].copy()

    trusted_mean = (
        successful[
            "trusted_service_restore_sec"
        ].mean()
        if not successful.empty
        else np.nan
    )

    overhead_mean = (
        x[
            "planning_overhead_sec"
        ].mean()
    )

    replans = (
        x[
            "replanned"
        ].astype(bool)
        .sum()
    )

    rows.append(
        {
            "variant":
                variant,

            "configuration":
                pretty[variant]
                .replace("\n", " "),

            "n":
                len(x),

            "success_rate_pct":
                success_rate,

            "trusted_restore_mean_sec":
                trusted_mean,

            "planning_overhead_mean_sec":
                overhead_mean,

            "replans":
                int(replans),
        }
    )


summary = pd.DataFrame(rows)

summary.to_csv(
    TABLE_CSV,
    index=False,
)


print("=" * 100)
print("REPLICA FRAMEWORK ABLATION")
print("=" * 100)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# VALUES FOR MONEY SHOTS
# ============================================================

full = summary[
    summary["variant"]
    == "full_replica"
].iloc[0]

no_state = summary[
    summary["variant"]
    == "no_state_reconstruction"
].iloc[0]

no_strategy = summary[
    summary["variant"]
    == "no_strategy_selection"
].iloc[0]

import matplotlib.pyplot as plt

# ============================================================
# GLOBAL GNUPLOT / ACADEMIC PRESETS
# ============================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.top": True,
        "ytick.right": True,
    }
)

# Gnuplot-style categorical colors
c_full = "#1f4e78"  # Solid navy
c_ablate = "#c00000"  # Accent red/burgundy


# ============================================================
# FIGURE 1: STATE RECONSTRUCTION -> RECOVERY FEASIBILITY
# ============================================================
labels = ["Full\nREPLICA", "Without State\nReconstruction"]
values = [full["success_rate_pct"], no_state["success_rate_pct"]]

fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)

bars = ax.bar(
    labels,
    values,
    width=0.52,
    color=[c_full, c_ablate],
    edgecolor="black",
    linewidth=1.1,
    zorder=3,
)

# Cross-hatch the ablated configuration
bars[1].set_hatch("//")

# Generous upper limit prevents collision with callout badges
ax.set_ylim(0, 125)
ax.set_ylabel("Recovery Success (%)", fontsize=11, fontweight="bold")
ax.set_title(
    "State Reconstruction Determines\nRecovery Feasibility",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#888888", zorder=0)
ax.tick_params(axis="both", labelsize=10.5)

# Direct data labels
for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2.5,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )

drop_pp = full["success_rate_pct"] - no_state["success_rate_pct"]

# Squared gnuplot-style summary badge
ax.text(
    0.50,
    0.88,
    f"{drop_pp:.1f} percentage-point drop",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    bbox=dict(
        boxstyle="square,pad=0.35",
        facecolor="#f9f9f9",
        edgecolor="#333333",
        linewidth=1.0,
    ),
    zorder=5,
)

# plt.savefig(
#     "ablation_state_reconstruction_success.png", dpi=300, bbox_inches="tight"
# )
# plt.show()

png1 = (
    OUTDIR
    / "ablation_state_reconstruction_success.png"
)

pdf1 = (
    OUTDIR
    / "ablation_state_reconstruction_success.pdf"
)

fig.savefig(
    png1,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf1,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2: STRATEGY SELECTION -> RESTORATION TIME
# ============================================================
labels = ["Full\nREPLICA", "Without Strategy\nSelection"]
values = [
    full["trusted_restore_mean_sec"],
    no_strategy["trusted_restore_mean_sec"],
]

fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)

bars = ax.bar(
    labels,
    values,
    width=0.52,
    color=[c_full, c_ablate],
    edgecolor="black",
    linewidth=1.1,
    zorder=3,
)

bars[1].set_hatch("//")

# Compute headroom dynamically based on the max value
max_val = max(values)
ax.set_ylim(0, max_val * 1.32)

ax.set_ylabel(
    "Mean Trusted-Service\nRestoration Time (s)", fontsize=11, fontweight="bold"
)
ax.set_title(
    "Recovery-Strategy Selection\nDetermines Restoration Time",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#888888", zorder=0)
ax.tick_params(axis="both", labelsize=10.5)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + (max_val * 0.02),
        f"{value:.1f} s",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )

slowdown = (
    no_strategy["trusted_restore_mean_sec"] / full["trusted_restore_mean_sec"]
)
reduction = (
    100.0
    * (
        no_strategy["trusted_restore_mean_sec"]
        - full["trusted_restore_mean_sec"]
    )
    / no_strategy["trusted_restore_mean_sec"]
)

ax.text(
    0.50,
    0.88,
    (
        f"{slowdown:.1f}$\\times$ longer without strategy selection\n"
        f"({reduction:.1f}\\% reduction with full REPLICA)"
    ),
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=9.5,
    fontweight="bold",
    bbox=dict(
        boxstyle="square,pad=0.35",
        facecolor="#f9f9f9",
        edgecolor="#333333",
        linewidth=1.0,
    ),
    zorder=5,
)

# plt.savefig(
#     "ablation_strategy_selection_restore_time.png", dpi=300, bbox_inches="tight"
# )
# plt.show()
png2 = (
    OUTDIR
    / "ablation_strategy_selection_restore_time.png"
)

pdf2 = (
    OUTDIR
    / "ablation_strategy_selection_restore_time.pdf"
)

fig.savefig(
    png2,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf2,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# LATEX TABLE
# ============================================================

def get_row(variant):

    return summary[
        summary["variant"]
        == variant
    ].iloc[0]


full_r = get_row(
    "full_replica"
)

state_r = get_row(
    "no_state_reconstruction"
)

strategy_r = get_row(
    "no_strategy_selection"
)

replan_r = get_row(
    "no_replanning"
)


# Important:
# Do not report restore time for the no-state-reconstruction
# variant because more than half of the cases fail. The mean
# among survivors is not comparable with the 100%-success runs.

latex = rf"""
\begin{{table}}[!t]
\centering
\caption{{REPLICA Framework Ablation Under Runtime Disruption}}
\label{{tab:replica-ablation}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.15}}

\begin{{tabular}}{{lccc}}
\toprule

\textbf{{Configuration}} &
\textbf{{\makecell{{Recovery \\ Success}}}} &
\textbf{{\makecell{{Trusted Restore \\ Time (s)}}}} &
\textbf{{Replans}} \\

\midrule

Full REPLICA
& \textbf{{{full_r["success_rate_pct"]:.1f}\%}}
& \textbf{{{full_r["trusted_restore_mean_sec"]:.1f}}}
& {int(full_r["replans"])} \\

No State Reconstruction
& \textbf{{{state_r["success_rate_pct"]:.1f}\%}}
& --$^{{*}}$
& {int(state_r["replans"])} \\

No Strategy Selection
& {strategy_r["success_rate_pct"]:.1f}\%
& {strategy_r["trusted_restore_mean_sec"]:.1f}
& {int(strategy_r["replans"])} \\

No Replanning
& {replan_r["success_rate_pct"]:.1f}\%
& {replan_r["trusted_restore_mean_sec"]:.1f}
& {int(replan_r["replans"])} \\

\bottomrule
\end{{tabular}}

\vspace{{2pt}}
\raggedright
\footnotesize
$^{{*}}$Trusted-service restoration time is not reported for the
no-state-reconstruction variant because failed recovery cases do not
have a restoration time; reporting the mean over successful cases
would therefore not be directly comparable.

\end{{table}}
"""

TABLE_TEX.write_text(
    latex
)


# ============================================================
# FINAL AUDIT
# ============================================================

print()
print("=" * 100)
print("HEADLINE RESULTS")
print("=" * 100)

print(
    "State reconstruction:"
)

print(
    f"  success "
    f"{full['success_rate_pct']:.1f}% "
    f"-> "
    f"{no_state['success_rate_pct']:.1f}%"
)

print(
    f"  drop = {drop_pp:.1f} percentage points"
)

print()

print(
    "Strategy selection:"
)

print(
    f"  trusted restore "
    f"{full['trusted_restore_mean_sec']:.3f}s "
    f"-> "
    f"{no_strategy['trusted_restore_mean_sec']:.3f}s"
)

print(
    f"  slowdown = {slowdown:.2f}x"
)

print(
    f"  full REPLICA reduction = {reduction:.2f}%"
)

print()
print("=" * 100)
print("SAVED")
print("=" * 100)

for p in [
    png1,
    pdf1,
    png2,
    pdf2,
    TABLE_CSV,
    TABLE_TEX,
]:
    print(p)

print()
print("=" * 100)
print("LATEX")
print("=" * 100)
print(latex)

