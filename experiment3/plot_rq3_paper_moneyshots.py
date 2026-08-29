#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/workspace/sc26_rebuttal")
CSV = (
    ROOT
    / "experiment3/generated/rq3_final_replanned_full"
    / "rq3_final_replanned_results.csv"
)
OUTDIR = (
    ROOT
    / "experiment3/generated/rq3_final_replanned_full"
    / "paper_figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV)
if len(df) != 6480:
    raise RuntimeError(f"Expected 6480 rows; found {len(df)}")

print("=" * 80)
print(f"EXPERIMENT 3 — PAPER PLOTS (Rows: {len(df)}, Cases: {df['case_id'].nunique()})")
print("=" * 80)

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
RHO_ORDER = [1.000, 0.800, 0.600, 0.400, 0.333]

# ============================================================
# HELPERS
# ============================================================

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def style_axis(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.20)

def finish(fig, stem):
    fig.tight_layout()
    p = OUTDIR / stem
    fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p.with_suffix('.png')}")

def method_lines(ax, data, xcol, median_col, q1_col=None, q3_col=None):
    for method in ORDER:
        g = data[data["method"] == method].copy()
        if g.empty:
            continue
        g = g.sort_values(xcol, ascending=True)

        ax.plot(
            g[xcol],
            g[median_col],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=3.5 if method == "replica" else 2.5,
            markersize=10 if method == "replica" else 8.5,
            label=LABELS[method],
            color=COLORS[method],
            zorder=8 if method == "replica" else 4,
        )

        if q1_col is not None and q3_col is not None:
            ax.fill_between(
                g[xcol],
                g[q1_col],
                g[q3_col],
                color=COLORS[method],
                alpha=0.15 if method == "replica" else 0.06,
                linewidth=0,
                zorder=2,
            )

rec = df[df["recovery_required"].astype(bool)].copy()

# ============================================================
# 1. ATTACK-SPECIFIC RECOVERY PLOTS
# ============================================================

def attack_recovery_plot(service, attack_type, xcol, xlabel, title, stem):
    x = rec[(rec["service"] == service) & (rec["attack_type"] == attack_type)].copy()
    if x.empty:
        return

    agg = (
        x.groupby([xcol, "method"], as_index=False)
        .agg(
            median_sec=("trusted_service_restore_sec", "median"),
            q1_sec=("trusted_service_restore_sec", q25),
            q3_sec=("trusted_service_restore_sec", q75),
        )
    )

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    method_lines(ax, agg, xcol, "median_sec", "q1_sec", "q3_sec")

    ax.set_xlabel(xlabel, fontsize=20, labelpad=12)
    ax.set_ylabel("Trusted-service restoration time (s)", fontsize=20, labelpad=12)
    ax.set_title(title, fontweight="bold", fontsize=22, pad=18)
    ax.tick_params(axis="both", labelsize=18)
    style_axis(ax)

    # Dynamic Zoom on Y-Axis
    ymin = max(0, agg["q1_sec"].min() - 5)
    ymax = agg["q3_sec"].max() * 1.15
    ax.set_ylim(ymin, ymax)

    ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=4, loc="upper left", fontsize=17)

    # Max severity comparison annotation
    xmax = agg[xcol].max()
    at_max = agg[agg[xcol] == xmax].set_index("method")
    if "replica" in at_max.index and "heft" in at_max.index:
        rep = float(at_max.loc["replica", "median_sec"])
        heft = float(at_max.loc["heft", "median_sec"])
        saved = heft - rep
        pct = 100.0 * saved / heft if heft > 0 else 0

        ax.annotate(
            f"REPLICA: {saved:.1f} s earlier\n({pct:.2f}% less interruption)",
            xy=(xmax, rep),
            xytext=(xmax * 0.70, heft * 0.60),
            arrowprops=dict(arrowstyle="->", linewidth=1.6, color="black"),
            fontsize=16,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.95),
        )

    finish(fig, stem)
    agg.to_csv(OUTDIR / f"{stem}_data.csv", index=False)

attack_recovery_plot(
    service="detection",
    attack_type="pgd",
    xcol="attack_severity",
    xlabel="PGD attack severity",
    title="Detection Recovery Time Under Increasing PGD Severity",
    stem="attack_detection_pgd",
)

attack_recovery_plot(
    service="detection",
    attack_type="patch",
    xcol="attack_severity",
    xlabel="Patch attack severity",
    title="Detection Recovery Time Under Increasing Patch Severity",
    stem="attack_detection_patch",
)

attack_recovery_plot(
    service="hvac",
    attack_type="additive_bias",
    xcol="attack_magnitude_c",
    xlabel="Outdoor-temperature additive bias magnitude (°C)",
    title="HVAC Recovery Time Under Increasing Sensor Bias",
    stem="attack_hvac_additive_bias",
)

attack_recovery_plot(
    service="hvac",
    attack_type="drift",
    xcol="attack_magnitude_c",
    xlabel="Outdoor-temperature drift magnitude (°C)",
    title="HVAC Recovery Time Under Increasing Sensor Drift",
    stem="attack_hvac_drift",
)

# ============================================================
# 2. COMBINED ATTACK RECOVERY (2-PANEL FIGURE)
# ============================================================

attack_combined = (
    rec.groupby(["service", "attack_severity", "method"], as_index=False)
    .agg(
        median_sec=("trusted_service_restore_sec", "median"),
        q1_sec=("trusted_service_restore_sec", q25),
        q3_sec=("trusted_service_restore_sec", q75),
    )
)

fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.8), sharey=False)

panel_specs = [
    (axes[0], "detection", "(a) Detection Attacks (PGD + Patch)"),
    (axes[1], "hvac", "(b) HVAC Attacks (Additive Bias + Drift)"),
]

for ax, service, title in panel_specs:
    panel = attack_combined[attack_combined["service"] == service].copy()
    method_lines(ax, panel, "attack_severity", "median_sec", "q1_sec", "q3_sec")

    ax.set_xlabel("Normalized attack severity", fontsize=20, labelpad=12)
    ax.set_ylabel("Trusted restoration time (s)", fontsize=20, labelpad=12)
    ax.set_title(title, fontweight="bold", fontsize=21, pad=16)
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_xticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    style_axis(ax)

    # Zoomed Y limits
    ymax = panel["q3_sec"].max() * 1.18
    ax.set_ylim(0, ymax)

    max_sev = panel["attack_severity"].max()
    endpoint = panel[panel["attack_severity"] == max_sev].set_index("method")
    if "replica" in endpoint.index and "heft" in endpoint.index:
        rep = float(endpoint.loc["replica", "median_sec"])
        heft = float(endpoint.loc["heft", "median_sec"])
        saved = heft - rep
        pct = 100.0 * saved / heft if heft > 0 else 0

        ax.annotate(
            f"{saved:.1f} s earlier\n({pct:.2f}% less)",
            xy=(max_sev, rep),
            xytext=(0.68, heft * 0.50),
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="black"),
            fontsize=16,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.95),
        )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=True,
    facecolor="white",
    edgecolor="none",
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    fontsize=18,
)

fig.suptitle(
    "Trusted-Service Recovery Under Increasing AI Degradation",
    fontsize=23,
    fontweight="bold",
    y=0.998,
)


finish(fig, "attack_recovery_combined")
attack_combined.to_csv(OUTDIR / "attack_recovery_combined_data.csv", index=False)

# ============================================================
# 3. DETECTION RESTORATION VS RHO
# ============================================================

det = rec[rec["service"] == "detection"].copy()
det_rho = (
    det.groupby(["rho", "method"], as_index=False)
    .agg(
        median_sec=("trusted_service_restore_sec", "median"),
        q1_sec=("trusted_service_restore_sec", q25),
        q3_sec=("trusted_service_restore_sec", q75),
    )
)

fig, ax = plt.subplots(figsize=(12.5, 7.5))
method_lines(ax, det_rho, "rho", "median_sec", "q1_sec", "q3_sec")

ax.set_xlim(1.04, 0.28)
ax.set_xticks(RHO_ORDER)
ax.set_xticklabels(["1.00", "0.80", "0.60", "0.40", "0.33"], fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_xlabel(r"Available assignment fraction, $\rho$", fontsize=20, labelpad=12)
ax.set_ylabel("Trusted-service restoration time (s)", fontsize=20, labelpad=12)
ax.set_title("Detection Service Restoration Under Resource Contraction", fontweight="bold", fontsize=22, pad=18)

# Zoom in Y-axis
ax.set_ylim(-5, det_rho["q3_sec"].max() * 1.15)
style_axis(ax)
ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=4, loc="upper left", fontsize=17)

temp = det_rho[det_rho["rho"] == 0.40].set_index("method")
if "heft" in temp.index and "replica" in temp.index:
    h = float(temp.loc["heft", "median_sec"])
    r = float(temp.loc["replica", "median_sec"])
    saved = h - r
    pct = 100 * saved / h

    ax.annotate(
        f"REPLICA: {saved:.1f} s earlier\n({pct:.2f}% less interruption)",
        xy=(0.40, r),
        xytext=(0.58, h * 0.52),
        arrowprops=dict(arrowstyle="->", linewidth=1.6, color="black"),
        fontsize=16,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.95),
    )

finish(fig, "detection_restore_vs_rho")
det_rho.to_csv(OUTDIR / "detection_restore_vs_rho_data.csv", index=False)

# ============================================================
# 4. HVAC FALLBACK BENEFIT RESTORATION VS RHO
# ============================================================

hvac = rec[rec["service"] == "hvac"].copy()
hvac_benefit = hvac[hvac["fallback_quality"] < hvac["attacked_quality"]].copy()

hvac_rho = (
    hvac_benefit.groupby(["rho", "method"], as_index=False)
    .agg(
        median_sec=("trusted_service_restore_sec", "median"),
        q1_sec=("trusted_service_restore_sec", q25),
        q3_sec=("trusted_service_restore_sec", q75),
    )
)

fig, ax = plt.subplots(figsize=(12.5, 7.5))
method_lines(ax, hvac_rho, "rho", "median_sec", "q1_sec", "q3_sec")

ax.set_xlim(1.04, 0.28)
ax.set_xticks(RHO_ORDER)
ax.set_xticklabels(["1.00", "0.80", "0.60", "0.40", "0.33"], fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_xlabel(r"Available assignment fraction, $\rho$", fontsize=20, labelpad=12)
ax.set_ylabel("Trusted-service restoration time (s)", fontsize=20, labelpad=12)
ax.set_title("HVAC Service Restoration Under Fallback Benefit", fontweight="bold", fontsize=22, pad=18)

ax.set_ylim(-5, hvac_rho["q3_sec"].max() * 1.15)
style_axis(ax)
ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=4, loc="upper left", fontsize=17)

temp_h = hvac_rho[hvac_rho["rho"] == 0.40].set_index("method")
if "heft" in temp_h.index and "replica" in temp_h.index:
    h = float(temp_h.loc["heft", "median_sec"])
    r = float(temp_h.loc["replica", "median_sec"])
    saved = h - r
    pct = 100 * saved / h

    ax.annotate(
        f"REPLICA restores {saved:.1f} s earlier\n({pct:.2f}% less interruption)",
        xy=(0.40, r),
        xytext=(0.58, h * 0.52),
        arrowprops=dict(arrowstyle="->", linewidth=1.6, color="black"),
        fontsize=16,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.95),
    )

finish(fig, "hvac_fallback_beneficial_restore_vs_rho")
hvac_rho.to_csv(OUTDIR / "hvac_fallback_beneficial_restore_vs_rho_data.csv", index=False)

# ============================================================
# 5. REMAINING INTERRUPTION PERCENTAGE BAR CHART
# ============================================================

pair = (
    det.pivot_table(
        index=["service", "attack_type", "attack_severity", "rho", "scenario_name"],
        columns="method",
        values="trusted_service_restore_sec",
        aggfunc="first",
    ).reset_index()
)
pair["replica_remaining_pct"] = 100.0 * pair["replica"] / pair["heft"]

remaining = (
    pair.groupby("rho", as_index=False)
    .agg(
        median_pct=("replica_remaining_pct", "median"),
        q1_pct=("replica_remaining_pct", q25),
        q3_pct=("replica_remaining_pct", q75),
    )
    .set_index("rho")
    .reindex(RHO_ORDER)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12.5, 7.5))
x = np.arange(len(remaining))

bars = ax.bar(
    x,
    remaining["median_pct"],
    width=0.55,
    color=COLORS["replica"],
    edgecolor="black",
    linewidth=1.0,
    hatch="//",
)

ax.set_xticks(x)
ax.set_xticklabels(["1.00", "0.80", "0.60", "0.40", "0.33"], fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_xlabel(r"Available assignment fraction, $\rho$", labelpad=12, fontsize=20)
ax.set_ylabel("REPLICA interruption remaining\nrelative to HEFT (%)", labelpad=12, fontsize=20)
ax.set_title("Detection-Service Interruption Remaining After Recovery Planning", fontweight="bold", fontsize=22, pad=18)

# Headroom for values
max_val = remaining["median_pct"].max()
ax.set_ylim(0, max_val * 1.35)
ax.grid(axis="y", linestyle="--", alpha=0.35)

for bar, value in zip(bars, remaining["median_pct"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + (max_val * 0.04),
        f"{value:.4f}%",
        ha="center",
        va="bottom",
        fontsize=17,
        fontweight="bold",
    )

finish(fig, "detection_interruption_remaining_vs_heft")
remaining.to_csv(OUTDIR / "detection_interruption_remaining_vs_heft_data.csv", index=False)

# ============================================================
# 6. PLANNING OVERHEAD VS RHO
# ============================================================

overhead = (
    rec.groupby(["rho", "method"], as_index=False)
    .agg(
        median_sec=("planning_overhead_sec", "median"),
        q1_sec=("planning_overhead_sec", q25),
        q3_sec=("planning_overhead_sec", q75),
    )
)

fig, ax = plt.subplots(figsize=(12.5, 7.5))
method_lines(ax, overhead, "rho", "median_sec", "q1_sec", "q3_sec")

ax.set_xlim(1.04, 0.28)
ax.set_xticks(RHO_ORDER)
ax.set_xticklabels(["1.00", "0.80", "0.60", "0.40", "0.33"], fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_xlabel(r"Available assignment fraction, $\rho$", fontsize=20, labelpad=12)
ax.set_ylabel("Planning / scheduling overhead (s)", fontsize=20, labelpad=12)
ax.set_title("Planning Cost as Feasible Resource Assignments Contract", fontweight="bold", fontsize=22, pad=18)

ax.set_ylim(-0.05, overhead["q3_sec"].max() * 1.25)
style_axis(ax)
ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=4, loc="upper right", fontsize=17)

o = overhead[overhead["rho"] == 0.40].set_index("method")
if "replica" in o.index:
    rv = float(o.loc["replica", "median_sec"])
    ax.annotate(
        f"REPLICA planning cost = {rv:.3f} s",
        xy=(0.40, rv),
        xytext=(0.58, rv - 0.22),
        arrowprops=dict(arrowstyle="->", linewidth=1.6, color="black"),
        fontsize=16,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.95),
    )

finish(fig, "planning_overhead_vs_rho")
overhead.to_csv(OUTDIR / "planning_overhead_vs_rho_data.csv", index=False)

# ============================================================
# 7. EXPORT SUMMARY METRICS TABLE
# ============================================================

perf = (
    rec.groupby(["rho", "method"])
    .agg(
        trusted_restore_sec=("trusted_service_restore_sec", "median"),
        preferred_restore_sec=("preferred_restore_sec", "median"),
        success_pct=("recovery_success", lambda x: 100 * x.astype(bool).mean()),
        overhead_sec=("planning_overhead_sec", "median"),
    )
    .reset_index()
)

rows = []
for rho in RHO_ORDER:
    g = perf[perf["rho"] == rho].set_index("method")
    row = {"rho": rho}
    for method in ORDER:
        if method not in g.index:
            continue
        row[f"{LABELS[method]} trusted restore (s)"] = float(g.loc[method, "trusted_restore_sec"])
        row[f"{LABELS[method]} success (%)"] = float(g.loc[method, "success_pct"])

    if "replica" in g.index and "heft" in g.index:
        heft_val = g.loc["heft", "trusted_restore_sec"]
        rep_val = g.loc["replica", "trusted_restore_sec"]
        row["REPLICA interruption reduction vs HEFT (%)"] = 100 * (heft_val - rep_val) / heft_val
        row["REPLICA overhead (s)"] = float(g.loc["replica", "overhead_sec"])

    rows.append(row)

paper_table = pd.DataFrame(rows)
paper_table.to_csv(OUTDIR / "experiment3_professor_metrics.csv", index=False)

print("\n" + "=" * 80)
print("EXPERIMENT 3 VISUALIZATIONS COMPLETE (CLEAN MARGINS, NO SUBTITLES)")
print("=" * 80)
for p in sorted(OUTDIR.glob("*")):
    print(" ", p)
print("=" * 80)

# ============================================================

# OUTPUT SUMMARY

# ============================================================
print()
print("=" * 100)
print("PAPER FIGURES COMPLETE")
print("=" * 100)
for p in sorted(
    OUTDIR.glob("*")

):
    print(p)

print("=" * 100) 