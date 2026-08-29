from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/workspace/sc26_rebuttal")

CSV = (
    ROOT
    / "experiment2/generated/final_benchmark_event_joint"
    / "final_benchmark_master.csv"
)

OUTDIR = (
    ROOT
    / "experiment2/generated/final_benchmark_event_joint"
)

CSV_OUT = OUTDIR / "experiment2_median_table_data.csv"
TEX_OUT = OUTDIR / "experiment2_median_table.tex"


# ============================================================
# LOAD FINAL EXPERIMENT 2 BENCHMARK
# ============================================================

df = pd.read_csv(CSV)

required = {
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


# ============================================================
# KEEP ONLY SUCCESSFUL / VALID FINAL RUNS
# ============================================================

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

found_schedulers = set(
    df["scheduler"].astype(str).unique()
)

if found_schedulers != expected_schedulers:
    raise RuntimeError(
        "Unexpected schedulers.\n"
        f"Expected: {sorted(expected_schedulers)}\n"
        f"Found:    {sorted(found_schedulers)}"
    )


# ============================================================
# VALIDATE PAIRED DESIGN
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
        "Expected exactly 61 scenarios per scheduler "
        "at every rho.\n"
        f"{counts}"
    )


# ============================================================
# MEDIAN MAKESPAN
#
# IMPORTANT:
# This is the same statistic used by the final Experiment 2
# makespan figure. The table and figure therefore report the
# same aggregation.
# ============================================================

median = (
    df.groupby(
        ["target_rho", "scheduler"]
    )["makespan_sec"]
    .median()
    .unstack()
    .sort_index(ascending=False)
)

median = median[
    [
        "replica",
        "cpsat",
        "heft",
        "easy",
    ]
].copy()


# ============================================================
# REPLICA COMPARISONS
#
# Positive value:
#   REPLICA has LOWER makespan than baseline.
#
# Negative value:
#   REPLICA has HIGHER makespan than baseline.
# ============================================================

median["vs_cpsat_pct"] = (
    100.0
    * (
        median["cpsat"]
        - median["replica"]
    )
    / median["cpsat"]
)

median["vs_heft_pct"] = (
    100.0
    * (
        median["heft"]
        - median["replica"]
    )
    / median["heft"]
)

median["vs_easy_pct"] = (
    100.0
    * (
        median["easy"]
        - median["replica"]
    )
    / median["easy"]
)


# ============================================================
# SAVE NUMERIC SOURCE TABLE
# ============================================================

median.reset_index().to_csv(
    CSV_OUT,
    index=False,
)


# ============================================================
# PRINT AUDIT
# ============================================================

print("=" * 110)
print("EXPERIMENT 2 — FINAL MEDIAN TABLE")
print("=" * 110)

print(
    median.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\n" + "=" * 110)
print("RHO = 0.50 HEADLINE CHECK")
print("=" * 110)

r = median.loc[0.5]

print(
    f"REPLICA median : {r['replica']:.3f} s"
)
print(
    f"CP-SAT median  : {r['cpsat']:.3f} s"
)
print(
    f"HEFT median    : {r['heft']:.3f} s"
)
print(
    f"EASY median    : {r['easy']:.3f} s"
)

print()
print(
    f"REPLICA vs HEFT : "
    f"{r['vs_heft_pct']:+.3f}%"
)
print(
    f"REPLICA vs EASY : "
    f"{r['vs_easy_pct']:+.3f}%"
)
print(
    f"REPLICA vs CP-SAT: "
    f"{r['vs_cpsat_pct']:+.3f}%"
)


# ============================================================
# LATEX HELPERS
# ============================================================

def fmt_rho(x):
    if abs(float(x) - (1.0 / 3.0)) < 0.01:
        return "0.33"
    return f"{float(x):.2f}"


def fmt_time(x):
    return f"{float(x):.2f}"


def fmt_pct(x):
    """
    Table convention:
      + means REPLICA lower/better
      - means REPLICA higher/worse
    """
    x = float(x)

    if abs(x) < 0.05:
        return "0.0"

    return f"{x:+.1f}"


# ============================================================
# BUILD LATEX ROWS
# ============================================================

rows = []

for rho, row in median.iterrows():

    rows.append(
        "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
            fmt_rho(rho),
            fmt_time(row["replica"]),
            fmt_time(row["cpsat"]),
            fmt_time(row["heft"]),
            fmt_time(row["easy"]),
            fmt_pct(row["vs_cpsat_pct"]),
            fmt_pct(row["vs_heft_pct"]),
            fmt_pct(row["vs_easy_pct"]),
        )
    )

body = "\n".join(rows)


# ============================================================
# LATEX TABLE
# ============================================================

latex = r"""
\begin{table*}[!t]
\centering
\caption{Experiment 2 --- Median Workflow Makespan Under Increasing Resource Constraints}
\label{tab:exp2-scheduling}
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}

\begin{tabular}{c rrrr rrr}
\toprule

& \multicolumn{4}{c}{\textbf{Median Makespan (s)}}
& \multicolumn{3}{c}{\textbf{REPLICA Improvement (\%)}} \\

\cmidrule(lr){2-5}
\cmidrule(lr){6-8}

\textbf{$\rho$} &
\textbf{REPLICA} &
\textbf{CP-SAT} &
\textbf{HEFT} &
\textbf{EASY} &
\textbf{vs. CP-SAT} &
\textbf{vs. HEFT} &
\textbf{vs. EASY} \\

\midrule
""" + body + r"""
\bottomrule

\end{tabular}

\vspace{2pt}
\footnotesize
Positive improvement indicates lower median makespan for REPLICA;
negative values indicate higher median makespan.

\end{table*}
"""

TEX_OUT.write_text(latex)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 110)
print("LATEX TABLE")
print("=" * 110)
print(latex)

print("\n" + "=" * 110)
print("SAVED")
print("=" * 110)
print(CSV_OUT)
print(TEX_OUT)

