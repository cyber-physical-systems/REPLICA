#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("/workspace/sc26_rebuttal")

INPUT = (
    ROOT
    / "experiment3/generated/rq3_final_replanned_full/"
      "rq3_final_replanned_results.csv"
)

OUT = (
    ROOT
    / "experiment3/generated/rq3_final_replanned_full/"
      "analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# LOAD + AUDIT
# =====================================================================

df = pd.read_csv(INPUT)

print("=" * 110)
print("EXPERIMENT 3 — FINAL REPLANNED ANALYSIS")
print("=" * 110)

print("Rows:", len(df))
print("Unique case IDs:", df["case_id"].nunique())
print()

if len(df) != 6480:
    raise RuntimeError(
        f"Expected 6480 rows, found {len(df)}"
    )

if df["case_id"].nunique() != 6480:
    raise RuntimeError(
        "Duplicate/missing case IDs detected."
    )


# Stable rho order: resource-rich -> resource-constrained
RHO_ORDER = [
    1.000,
    0.800,
    0.600,
    0.400,
    0.333,
]

METHOD_ORDER = [
    "cpsat",
    "heft",
    "replica",
    "easy",
]


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def savefig(name):
    png_path = OUT / name
    pdf_path = OUT / Path(name).with_suffix(".pdf")

    plt.tight_layout()

    # High-resolution raster copy.
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # Vector copy for publication.
    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", png_path)
    print("Saved:", pdf_path)


# =====================================================================
# TABLE IV
# EMPIRICAL RECOVERY WORKLOAD / MODEL QUALITY
# =====================================================================

rows = []

for service, g in df.groupby("service"):

    unique_states = (
        g[
            [
                "attack_type",
                "attack_severity",
                "attacked_quality",
            ]
        ]
        .drop_duplicates()
    )

    rows.append(
        {
            "service":
                service,

            "primary_model":
                ", ".join(
                    sorted(
                        g["primary_model"]
                        .astype(str)
                        .unique()
                    )
                ),

            "fallback_model":
                ", ".join(
                    sorted(
                        g["fallback_model"]
                        .astype(str)
                        .unique()
                    )
                ),

            "quality_metric":
                ", ".join(
                    sorted(
                        g["quality_metric"]
                        .astype(str)
                        .unique()
                    )
                ),

            "clean_quality":
                g["clean_quality"].iloc[0],

            "attacked_quality_min":
                g["attacked_quality"].min(),

            "attacked_quality_max":
                g["attacked_quality"].max(),

            "fallback_quality":
                g["fallback_quality"].iloc[0],

            "attack_types":
                " | ".join(
                    sorted(
                        g["attack_type"]
                        .astype(str)
                        .unique()
                    )
                ),

            "attack_severity_min":
                g["attack_severity"].min(),

            "attack_severity_max":
                g["attack_severity"].max(),

            "empirical_attack_states":
                len(unique_states),

            "resource_worlds":
                g["scenario_name"].nunique(),

            "experiment_cases":
                len(g),
        }
    )

table4 = pd.DataFrame(rows)

table4.to_csv(
    OUT / "table4_empirical_recovery_workload.csv",
    index=False,
)

print(
    "\nTABLE IV — EMPIRICAL RECOVERY WORKLOAD"
)
print(
    table4.to_string(index=False)
)


# =====================================================================
# TABLE V
# RECOVERY PERFORMANCE UNDER ASSIGNMENT LOSS
#
# Only recovery-required cases. Healthy cases would trivially
# contribute zero restoration time.
# =====================================================================

rec = df[
    df["recovery_required"] == True
].copy()

table5 = (
    rec
    .groupby(
        [
            "service",
            "rho",
            "method",
        ],
        as_index=False,
    )
    .agg(
        cases=(
            "case_id",
            "size",
        ),

        recovery_success_pct=(
            "recovery_success",
            lambda x:
                100.0
                * x.astype(bool).mean(),
        ),

        trusted_restore_median_sec=(
            "trusted_service_restore_sec",
            "median",
        ),

        trusted_restore_q1_sec=(
            "trusted_service_restore_sec",
            q25,
        ),

        trusted_restore_q3_sec=(
            "trusted_service_restore_sec",
            q75,
        ),

        preferred_restore_median_sec=(
            "preferred_restore_sec",
            "median",
        ),

        preferred_restore_q1_sec=(
            "preferred_restore_sec",
            q25,
        ),

        preferred_restore_q3_sec=(
            "preferred_restore_sec",
            q75,
        ),

        planning_overhead_median_sec=(
            "planning_overhead_sec",
            "median",
        ),

        planning_overhead_q1_sec=(
            "planning_overhead_sec",
            q25,
        ),

        planning_overhead_q3_sec=(
            "planning_overhead_sec",
            q75,
        ),
    )
)

table5.to_csv(
    OUT / "table5_recovery_performance_by_rho.csv",
    index=False,
)

print(
    "\nTABLE V — RECOVERY PERFORMANCE"
)
print(
    table5.to_string(index=False)
)


# =====================================================================
# TABLE VI
# NUMERIC REPLANNING / STRATEGY TRANSITIONS
# =====================================================================

replica = df[
    df["method"] == "replica"
].copy()

table6 = (
    replica
    .groupby(
        [
            "service",
            "rho",
        ],
        as_index=False,
    )
    .agg(
        cases=(
            "case_id",
            "size",
        ),

        recovery_required_cases=(
            "recovery_required",
            lambda x:
                int(
                    x.astype(bool).sum()
                ),
        ),

        replanned_cases=(
            "replanned",
            lambda x:
                int(
                    x.astype(bool).sum()
                ),
        ),
    )
)

table6[
    "replanning_pct_of_all_cases"
] = (
    100.0
    * table6["replanned_cases"]
    / table6["cases"]
)

table6[
    "replanning_pct_of_recovery_cases"
] = np.where(
    table6["recovery_required_cases"] > 0,

    100.0
    * table6["replanned_cases"]
    / table6[
        "recovery_required_cases"
    ],

    0.0,
)

table6.to_csv(
    OUT / "table6_replica_replanning_by_rho.csv",
    index=False,
)

print(
    "\nTABLE VI — REPLANNING BY RESOURCE STATE"
)
print(
    table6.to_string(index=False)
)


# =====================================================================
# FIGURE 1A
# DETECTION MODEL QUALITY VS ATTACK SEVERITY
# =====================================================================

det = (
    df[
        df["service"] == "detection"
    ][
        [
            "attack_type",
            "attack_severity",
            "clean_quality",
            "attacked_quality",
            "fallback_quality",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "attack_type",
            "attack_severity",
        ]
    )
)

plt.figure(
    figsize=(8.5, 5.2)
)

for attack, g in det.groupby(
    "attack_type"
):

    plt.plot(
        g["attack_severity"],
        g["attacked_quality"],
        marker="o",
        linewidth=2,
        label=attack.upper(),
    )

clean = det[
    "clean_quality"
].iloc[0]

fallback = det[
    "fallback_quality"
].iloc[0]

plt.axhline(
    clean,
    linestyle="--",
    linewidth=1.5,
    label=f"Clean YOLO11n = {clean:.3f}",
)

plt.axhline(
    fallback,
    linestyle=":",
    linewidth=1.8,
    label=f"YOLOv5s fallback = {fallback:.3f}",
)

plt.xlabel(
    "Attack severity"
)

plt.ylabel(
    "Detection quality, mAP@50–95"
)

plt.xticks(
    [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
    ]
)

plt.title(
    "Detection Quality Under Increasing Attack Severity"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend(
    frameon=False
)

savefig(
    "fig1a_detection_quality_vs_attack_severity.png"
)


# =====================================================================
# FIGURE 1B
# HVAC MODEL QUALITY VS ATTACK MAGNITUDE
#
# Use °C on x and RMSE °C on y: both axes are physical quantities.
# =====================================================================

hvac = (
    df[
        df["service"] == "hvac"
    ][
        [
            "attack_type",
            "attack_magnitude_c",
            "clean_quality",
            "attacked_quality",
            "fallback_quality",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "attack_type",
            "attack_magnitude_c",
        ]
    )
)

plt.figure(
    figsize=(8.5, 5.2)
)

for attack, g in hvac.groupby(
    "attack_type"
):

    plt.plot(
        g["attack_magnitude_c"],
        g["attacked_quality"],
        marker="o",
        linewidth=2,
        label=attack.replace(
            "_",
            " ",
        ).title(),
    )

clean = hvac[
    "clean_quality"
].iloc[0]

fallback = hvac[
    "fallback_quality"
].iloc[0]

plt.axhline(
    clean,
    linestyle="--",
    linewidth=1.5,
    label=f"Clean LSTM = {clean:.3f} °C",
)

plt.axhline(
    fallback,
    linestyle=":",
    linewidth=1.8,
    label=f"RF fallback = {fallback:.3f} °C",
)

plt.xlabel(
    "Outdoor-temperature perturbation magnitude (°C)"
)

plt.ylabel(
    "Zone-temperature RMSE (°C)"
)

plt.title(
    "HVAC Prediction Error Under Increasing Sensor Perturbation"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend(
    frameon=False
)

savefig(
    "fig1b_hvac_rmse_vs_attack_magnitude.png"
)


# =====================================================================
# FIGURE 2
# REPLANNING FREQUENCY VS AVAILABLE ASSIGNMENT FRACTION
#
# Denominator = recovery-required REPLICA cases.
# =====================================================================

rr = replica[
    replica["recovery_required"] == True
].copy()

replan = (
    rr
    .groupby(
        [
            "service",
            "rho",
        ],
        as_index=False,
    )
    .agg(
        cases=(
            "case_id",
            "size",
        ),

        replanned=(
            "replanned",
            lambda x:
                x.astype(bool).sum(),
        ),
    )
)

replan[
    "replanning_pct"
] = (
    100.0
    * replan["replanned"]
    / replan["cases"]
)

plt.figure(
    figsize=(8.5, 5.2)
)

for service, g in replan.groupby(
    "service"
):

    g = (
        g
        .set_index("rho")
        .reindex(RHO_ORDER)
        .reset_index()
    )

    plt.plot(
        g["rho"],
        g["replanning_pct"],
        marker="o",
        linewidth=2,
        label=service.title(),
    )

plt.gca().invert_xaxis()

plt.xlabel(
    r"Available assignment fraction, $\rho$"
)

plt.ylabel(
    "Recovery cases requiring replanning (%)"
)

plt.ylim(
    -3,
    103,
)

plt.title(
    "Recovery Replanning as Feasible Assignments Contract"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend(
    frameon=False
)

savefig(
    "fig2_replanning_pct_vs_rho.png"
)


# =====================================================================
# FIGURE 3A — MONEY SHOT
# DETECTION TRUSTED-SERVICE RESTORATION TIME
# =====================================================================

det_rec = rec[
    rec["service"] == "detection"
].copy()

det_time = (
    det_rec
    .groupby(
        [
            "rho",
            "method",
        ],
        as_index=False,
    )
    .agg(
        median_sec=(
            "trusted_service_restore_sec",
            "median",
        ),

        q1_sec=(
            "trusted_service_restore_sec",
            q25,
        ),

        q3_sec=(
            "trusted_service_restore_sec",
            q75,
        ),
    )
)

plt.figure(
    figsize=(9.3, 5.6)
)

for method in METHOD_ORDER:

    g = det_time[
        det_time["method"] == method
    ]

    g = (
        g
        .set_index("rho")
        .reindex(RHO_ORDER)
        .reset_index()
    )

    plt.plot(
        g["rho"],
        g["median_sec"],
        marker="o",
        linewidth=2,
        label=method.upper(),
    )

    plt.fill_between(
        g["rho"],
        g["q1_sec"],
        g["q3_sec"],
        alpha=0.08,
    )

plt.gca().invert_xaxis()

plt.xlabel(
    r"Available assignment fraction, $\rho$"
)

plt.ylabel(
    "Trusted-service restoration time (s)"
)

plt.title(
    "Detection Service Restoration as Feasible Assignments Contract"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend(
    frameon=False,
    ncol=4,
)

savefig(
    "fig3a_detection_trusted_restore_vs_rho.png"
)


# =====================================================================
# FIGURE 3B
# HVAC TRUSTED-SERVICE RESTORATION TIME
# =====================================================================

hvac_rec = rec[
    rec["service"] == "hvac"
].copy()

hvac_time = (
    hvac_rec
    .groupby(
        [
            "rho",
            "method",
        ],
        as_index=False,
    )
    .agg(
        median_sec=(
            "trusted_service_restore_sec",
            "median",
        ),

        q1_sec=(
            "trusted_service_restore_sec",
            q25,
        ),

        q3_sec=(
            "trusted_service_restore_sec",
            q75,
        ),
    )
)

plt.figure(
    figsize=(9.3, 5.6)
)

for method in METHOD_ORDER:

    g = hvac_time[
        hvac_time["method"] == method
    ]

    g = (
        g
        .set_index("rho")
        .reindex(RHO_ORDER)
        .reset_index()
    )

    plt.plot(
        g["rho"],
        g["median_sec"],
        marker="o",
        linewidth=2,
        label=method.upper(),
    )

    plt.fill_between(
        g["rho"],
        g["q1_sec"],
        g["q3_sec"],
        alpha=0.08,
    )

plt.gca().invert_xaxis()

plt.xlabel(
    r"Available assignment fraction, $\rho$"
)

plt.ylabel(
    "Trusted-service restoration time (s)"
)

plt.title(
    "HVAC Service Restoration as Feasible Assignments Contract"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend(
    frameon=False,
    ncol=4,
)

savefig(
    "fig3b_hvac_trusted_restore_vs_rho.png"
)


# =====================================================================
# FIGURE 4 — MONEY SHOT
# DETECTION INTERRUPTION REDUCTION VS HEFT
#
# Paired by exact attack state AND exact frozen resource world.
# =====================================================================

KEY = [
    "service",
    "attack_type",
    "attack_severity",
    "rho",
    "scenario_name",
]

paired = (
    det_rec
    .pivot_table(
        index=KEY,
        columns="method",
        values="trusted_service_restore_sec",
        aggfunc="first",
    )
    .reset_index()
)

paired[
    "replica_reduction_vs_heft_pct"
] = (
    100.0
    * (
        1.0
        -
        paired["replica"]
        / paired["heft"]
    )
)

reduction = (
    paired
    .groupby(
        "rho",
        as_index=False,
    )
    .agg(
        median_pct=(
            "replica_reduction_vs_heft_pct",
            "median",
        ),

        q1_pct=(
            "replica_reduction_vs_heft_pct",
            q25,
        ),

        q3_pct=(
            "replica_reduction_vs_heft_pct",
            q75,
        ),
    )
)

reduction = (
    reduction
    .set_index("rho")
    .reindex(RHO_ORDER)
    .reset_index()
)

plt.figure(
    figsize=(8.7, 5.3)
)

plt.bar(
    reduction["rho"].astype(str),
    reduction["median_pct"],
)

for i, row in reduction.iterrows():

    plt.text(
        i,
        row["median_pct"] + 0.002,
        f"{row['median_pct']:.3f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.xlabel(
    r"Available assignment fraction, $\rho$"
)

plt.ylabel(
    "Reduction in trusted-service interruption vs. HEFT (%)"
)

plt.title(
    "Reduction in Detection-Service Interruption"
)

plt.ylim(
    reduction["median_pct"].min() - 0.02,
    100.005,
)

plt.grid(
    axis="y",
    alpha=0.25,
)

savefig(
    "fig4_detection_interruption_reduction_vs_heft.png"
)


# =====================================================================
# FIGURE 5 — MONEY SHOT
# REPLICA TRUSTED SERVICE VS PREFERRED-MODEL RESTORATION
# DETECTION
# =====================================================================

det_rep = det_rec[
    det_rec["method"] == "replica"
].copy()

restore_gap = (
    det_rep
    .groupby(
        "rho",
        as_index=False,
    )
    .agg(
        trusted_median_sec=(
            "trusted_service_restore_sec",
            "median",
        ),

        preferred_median_sec=(
            "preferred_restore_sec",
            "median",
        ),
    )
)

restore_gap = (
    restore_gap
    .set_index("rho")
    .reindex(RHO_ORDER)
    .reset_index()
)

plt.figure(
    figsize=(8.8, 5.4)
)

plt.plot(
    restore_gap["rho"],
    restore_gap[
        "trusted_median_sec"
    ],
    marker="o",
    linewidth=2.4,
    label="Trusted service restored",
)

plt.plot(
    restore_gap["rho"],
    restore_gap[
        "preferred_median_sec"
    ],
    marker="s",
    linewidth=2.4,
    label="Preferred model restored",
)

plt.gca().invert_xaxis()

plt.xlabel(
    r"Available assignment fraction, $\rho$"
)

plt.ylabel(
    "Restoration time (s)"
)

plt.title(
    "Trusted-Service Restoration Precedes Preferred-Model Recovery"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend(
    frameon=False
)

savefig(
    "fig5_detection_trusted_vs_preferred_restore.png"
)


# =====================================================================
# FIGURE 6
# PLANNING OVERHEAD VS RHO
# =====================================================================

over = (
    rec
    .groupby(
        [
            "rho",
            "method",
        ],
        as_index=False,
    )
    .agg(
        median_sec=(
            "planning_overhead_sec",
            "median",
        ),

        q1_sec=(
            "planning_overhead_sec",
            q25,
        ),

        q3_sec=(
            "planning_overhead_sec",
            q75,
        ),
    )
)

plt.figure(
    figsize=(9.0, 5.4)
)

for method in METHOD_ORDER:

    g = over[
        over["method"] == method
    ]

    g = (
        g
        .set_index("rho")
        .reindex(RHO_ORDER)
        .reset_index()
    )

    plt.plot(
        g["rho"],
        g["median_sec"],
        marker="o",
        linewidth=2,
        label=method.upper(),
    )

plt.gca().invert_xaxis()

plt.xlabel(
    r"Available assignment fraction, $\rho$"
)

plt.ylabel(
    "Planning/scheduling overhead (s)"
)

plt.title(
    "Planning Overhead as Feasible Assignments Contract"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend(
    frameon=False,
    ncol=4,
)

savefig(
    "fig6_planning_overhead_vs_rho.png"
)


# =====================================================================
# SAVE FROZEN FIGURE DATA TOO
# =====================================================================

det.to_csv(
    OUT / "figure1a_detection_quality_data.csv",
    index=False,
)

hvac.to_csv(
    OUT / "figure1b_hvac_quality_data.csv",
    index=False,
)

replan.to_csv(
    OUT / "figure2_replanning_data.csv",
    index=False,
)

det_time.to_csv(
    OUT / "figure3a_detection_restore_data.csv",
    index=False,
)

hvac_time.to_csv(
    OUT / "figure3b_hvac_restore_data.csv",
    index=False,
)

reduction.to_csv(
    OUT / "figure4_interruption_reduction_data.csv",
    index=False,
)

restore_gap.to_csv(
    OUT / "figure5_restore_gap_data.csv",
    index=False,
)

over.to_csv(
    OUT / "figure6_overhead_data.csv",
    index=False,
)


print()
print("=" * 110)
print("ANALYSIS COMPLETE")
print("=" * 110)

print("Output directory:")
print(OUT)

print()
print("REPLANNED CASES:")
print(
    df["replanned"]
    .astype(bool)
    .value_counts()
)

print()
print("RECOVERY SUCCESS:")
print(
    df["recovery_success"]
    .astype(bool)
    .value_counts()
)
