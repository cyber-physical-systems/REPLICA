import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# INPUT / OUTPUT
# ============================================================

INFILE = Path(
    "experiment3/generated/rq3_final_replanned_full/"
    "rq3_final_paired_audit.csv"
)

OUTDIR = Path(
    "experiment3/generated/rq3_final_replanned_full/figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INFILE)

# ============================================================
# SUBSETS
# ============================================================

recovery = df[
    df["preferred_recovery_action"]
    != "no_recovery_required"
].copy()

failover = df[
    df["recovery_strategy"]
    .astype(str)
    .str.startswith("failover_")
].copy()

methods = [
    ("replica", "REPLICA"),
    ("heft", "HEFT"),
    ("cpsat", "CP-SAT"),
    ("easy", "EASY"),
]

# ============================================================
# VALUES
# ============================================================

recovery_means = [
    recovery[
        f"trusted_service_restore_sec_{m}"
    ].mean()
    for m, _ in methods
]

failover_means = [
    failover[
        f"trusted_service_restore_sec_{m}"
    ].mean()
    for m, _ in methods
]

labels = [label for _, label in methods]

print("=" * 90)
print("MONEY PLOT DATA")
print("=" * 90)

print("\nRecovery-required cases:", len(recovery))
for label, value in zip(labels, recovery_means):
    print(f"{label:8s}: {value:.6f} s")

print("\nFailover cases:", len(failover))
for label, value in zip(labels, failover_means):
    print(f"{label:8s}: {value:.6f} s")

overall_reduction = (
    100
    * (recovery_means[1] - recovery_means[0])
    / recovery_means[1]
)

failover_reduction = (
    100
    * (failover_means[1] - failover_means[0])
    / failover_means[1]
)

print(
    f"\nRecovery reduction: "
    f"{overall_reduction:.3f}%"
)
print(
    f"Failover reduction: "
    f"{failover_reduction:.3f}%"
)


# Apply gnuplot / academic paper style presets
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

# Colors and hatch patterns inspired by gnuplot terminal outputs
colors = ["#1f4e78", "#c00000", "#385723", "#d68910"]
hatches = ["//", "\\\\", "xx", ".."]

# ============================================================
# FIGURE SETUP
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
fig.suptitle(
    "Trusted-Service Restoration Under Runtime Disruption",
    fontsize=13,
    fontweight="bold",
    y=1.04,
)

# ------------------------------------------------------------
# LEFT: ALL RECOVERY-REQUIRED CASES
# ------------------------------------------------------------
ax = axes[0]
bars = ax.bar(
    labels,
    recovery_means,
    width=0.62,
    color=colors,
    edgecolor="black",
    linewidth=1.0,
    hatch=hatches,
    zorder=3,
)

ax.set_ylabel("Mean Trusted-Service\nRestoration Time (s)", fontsize=10)
ax.set_title(
    f"All Recovery-Required Cases\n(n={len(recovery)})",
    fontsize=11,
    fontweight="bold",
    pad=10,
)
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#888888", zorder=0)

# Expand y-limit to ensure annotations and comparison boxes do not collide
max_left = max(recovery_means)
ax.set_ylim(0, max_left * 1.30)

for bar, value in zip(bars, recovery_means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + (max_left * 0.02),
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax.text(
    0.5,
    0.92,
    f"REPLICA: {overall_reduction:.1f}% lower",
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
)

# ------------------------------------------------------------
# RIGHT: FAILOVER CASES
# ------------------------------------------------------------
ax = axes[1]
bars = ax.bar(
    labels,
    failover_means,
    width=0.62,
    color=colors,
    edgecolor="black",
    linewidth=1.0,
    hatch=hatches,
    zorder=3,
)

ax.set_title(
    f"Trusted-Failover Cases\n(n={len(failover)})",
    fontsize=11,
    fontweight="bold",
    pad=10,
)
ax.set_ylabel("Mean Trusted-Service\nRestoration Time (s)", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#888888", zorder=0)

max_right = max(failover_means)
ax.set_ylim(0, max_right * 1.30)

for i, (bar, value) in enumerate(zip(bars, failover_means)):
    if i == 0:
        # Custom pointer for near-zero value
        ax.annotate(
            f"{value:.3f} s",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 32),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.2,
                color="black",
                shrinkA=0,
                shrinkB=3,
            ),
        )
    else:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (max_right * 0.02),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

ax.text(
    0.5,
    0.92,
    f"REPLICA: {failover_reduction:.1f}% lower",
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
)
# ============================================================
# SAVE
# ============================================================

fig.suptitle(
    "Trusted-Service Restoration Under Runtime Disruption",
    fontsize=13,
    fontweight="bold",
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93]
)

png = (
    OUTDIR
    / "experiment3_trusted_service_moneyshot.png"
)

pdf = (
    OUTDIR
    / "experiment3_trusted_service_moneyshot.pdf"
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

print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)
print(png)
print(pdf)
