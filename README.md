Task scheduling across edge-to-cloud environments is increasingly demanding due to the wide adoption and continued emergence of edge devices. Traditional scheduling approaches perform well under normal resource conditions, but abrupt disruptions or adversarial attacks can create runtime states in which existing task assignments or recovery decisions fail or are no longer valid. To address this issue, we present REPLICA, a state-aware orchestration framework that uses the Planning Domain Definition Language to represent the current system state, resource capabilities, task requirements, and workflow goals. REPLICA operates as a closed loop that observes system conditions, reconstructs the planning problem, executes validated actions, and replans unfinished workloads when resource availability or cloud model performance degrades. We evaluate REPLICA against Dynamic HEFT, CP-SAT, and EASY Backfill, and our results show that as resource availability decreases, REPLICA remains competitive with CP-SAT while reducing workflow makespan by up to 21.6\% over HEFT and 40.5\% over EASY Backfill. In particular, under runtime resource loss and adversarial attacks, REPLICA outperforms all other baselines, achieving the best recovery success and reducing trusted-service interruption by 99.3~99.7\%. We release REPLICA to the public research community

## Setup

REPLICA requires Python 3.12. The following commands create an isolated Python environment, clone the repository, and install the required dependencies.

### 1. Create a Python 3.12 virtual environment

```bash
python3.12 -m venv replica_env
```

### 2. Activate the virtual environment

On Linux or macOS:

```bash
source replica_env/bin/activate
```

On Windows PowerShell:

```powershell
replica_env\Scripts\Activate.ps1
```

After activation, verify that Python 3.12 is being used:

```bash
python --version
```

The output should report Python 3.12.

### 3. Clone the REPLICA repository

```bash
git clone https://github.com/cyber-physical-systems/REPLICA.git
cd REPLICA
```

All remaining commands should be run from the repository root (`REPLICA/`) with the virtual environment activated.

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install the required Python packages

```bash
python -m pip install -r requirements.txt
```

### 6. Verify the installation

```bash
python - <<'PY'
import numpy
import pandas
import matplotlib
import ortools
import unified_planning

print("REPLICA Python environment is ready.")
PY
```

If the command completes without an import error, the Python environment is ready.

---

## Running the Experiments

From this point forward, make sure the virtual environment is active:

```bash
source replica_env/bin/activate
```

and move to the repository root before running the experiments:

```bash
cd REPLICA
```

The experiments can then be reproduced using the commands provided in the sections below.

---

## Experiment 1: Execution Profiles

Experiment 1 provides the empirical execution costs used by the scheduling and recovery experiments.

To rebuild the execution profiles:

```bash
python -m experiment2.build_profiles
```

Expected output:

```text
experiment2/generated/execution_profiles.json
```

Quick check:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("experiment2/generated/execution_profiles.json")
rows = json.loads(p.read_text())

print("profiles:", len(rows))
print(rows[:3])
PY
```

If `execution_profiles.json` is already included in the release, it does not need to be regenerated.

---

## Experiment 2: Scheduling Under Increasing Resource Constraints

Experiment 2 compares REPLICA with CP-SAT, HEFT, and EASY Backfill as feasible task-to-resource assignments are progressively removed.

### 1. Build the empirical resource-state inputs

If these artifacts are already included in the release, this step can be skipped.

```bash
python -m experiment2.build_dataset_replay
python -m experiment2.build_dynamic_feasibility
python -m experiment2.build_empirical_rho_ladder
```

### 2. Build the frozen scheduling scenarios

```bash
python -m experiment2.scenarios_multimask
```

The frozen scenarios used by the final experiments are stored in:

```text
experiment2/generated/scenarios_multimask_FROZEN/
```

The final evaluation contains multiple resource masks at:

```text
rho = 0.333
rho = 0.400
rho = 0.600
rho = 0.800
```

and a nominal case at:

```text
rho = 1.000
```

### 3. Run the final Experiment 2 benchmark

```bash
python -m experiment2.run_final_benchmark
```

The final paper benchmark contains:

```text
61 scenarios
8 resource-constraint levels
4 schedulers
1952 total scheduler runs
```

The final source-of-truth results are stored in:

```text
experiment2/generated/final_benchmark_event_joint/final_benchmark_master.csv
```

### 4. Generate the Experiment 2 figures

```bash
python -m experiment2.plot_final_benchmark_event_joint
python -m experiment2.plot_final_event_joint_moneyshots
python -m experiment2.plot_replica_vs_cpsat_competitiveness
```

Important paper figures include:

```text
experiment2_makespan_vs_rho.pdf
experiment2_replica_vs_heft.pdf
experiment2_replica_vs_cpsat_competitiveness.pdf
```

### 5. Generate the Experiment 2 table

The paper reports median workflow makespan across the 61 scenarios at each resource-constraint level.

```bash
python -m experiment2.generate_final_median_table
```

Expected outputs:

```text
experiment2/generated/final_benchmark_event_joint/experiment2_median_table_data.csv
experiment2/generated/final_benchmark_event_joint/experiment2_median_table.tex
```

At `rho = 0.50`, the expected median values are approximately:

```text
REPLICA : 366.06 s
CP-SAT  : 364.93 s
HEFT    : 466.56 s
EASY    : 387.06 s
```

This corresponds to approximately:

```text
REPLICA vs. HEFT   : 21.5% lower
REPLICA vs. EASY   : 5.4% lower
REPLICA vs. CP-SAT : within 0.3%
```

---

## Experiment 3: Recovery Under Runtime Disruption

Experiment 3 evaluates recovery when resource availability or AI service quality changes during execution.

The public release does **not** include code for generating adversarial attacks. Instead, the release contains the frozen experimental conditions and measured model-quality states used by the orchestration experiment. This allows the scheduling and recovery results reported in the paper to be reproduced without regenerating the attacks.

### 1. Final Experiment 3 cases

The frozen Experiment 3 cases are stored in:

```text
experiment3/generated/rq3_final_multimask/rq3_final_cases.csv
```

### 2. Run the final recovery experiment

```bash
python -m experiment3.run_rq3_final_replanned_full
```

Expected output:

```text
experiment3/generated/rq3_final_replanned_full/rq3_final_replanned_results.csv
```

The final experiment contains:

```text
1620 experimental conditions
4 schedulers
6480 total runs
```

### 3. Analyze the final recovery results

```bash
python -m experiment3.analyze_rq3_final_replanned
```

The final paired audit is stored in:

```text
experiment3/generated/rq3_final_replanned_full/rq3_final_paired_audit.csv
```

Expected high-level results include:

```text
Recovery-required cases : 1215
Trusted-failover cases  : 891
Replanned cases          : 315

REPLICA recovery success : 100%
```

Across recovery-required cases, mean trusted-service restoration time is approximately:

```text
REPLICA : 23.6 s
HEFT    : 161.0 s
CP-SAT  : 161.0 s
EASY    : 161.0 s
```

For trusted-failover cases:

```text
REPLICA : 0.240 s
Baselines: approximately 187.6 s
```

### 4. Generate the Experiment 3 paper figures

```bash
python -m experiment3.plot_rq3_paper_moneyshots
python -m experiment3.plot_final_trusted_service_moneyshot
```

Paper figures are written under:

```text
experiment3/generated/rq3_final_replanned_full/paper_figures/
experiment3/generated/rq3_final_replanned_full/figures/
```

---

## REPLICA Component Ablation

The ablation evaluates which REPLICA components contribute most to runtime recovery.

The evaluated configurations include:

```text
Full REPLICA
No State Reconstruction
No Strategy Selection
No Replanning
No Closed-Loop Adaptation
```

### 1. Run the ablation

```bash
python -m experiment3.run_rq3_replica_ablation
```

Expected output:

```text
experiment3/generated/rq3_replica_ablation/rq3_replica_ablation_results.csv
experiment3/generated/rq3_replica_ablation/rq3_replica_ablation_summary.csv
```

Expected headline results:

```text
Full REPLICA
  Recovery success: 100.0%
  Mean trusted-service restoration: 17.7 s

No State Reconstruction
  Recovery success: 49.4%

No Strategy Selection
  Recovery success: 100.0%
  Mean trusted-service restoration: 120.8 s
```

These results show that state reconstruction primarily affects whether a valid recovery path can be found, while recovery-strategy selection primarily affects how quickly trusted service can be restored.

### 2. Generate the ablation figures and table

```bash
python -m experiment3.plot_rq3_replica_ablation_moneyshots
```

Expected figures:

```text
experiment3/generated/rq3_replica_ablation/figures/ablation_state_reconstruction_success.pdf

experiment3/generated/rq3_replica_ablation/figures/ablation_strategy_selection_restore_time.pdf
```

Expected LaTeX table:

```text
experiment3/generated/rq3_replica_ablation/figures/replica_framework_ablation_table.tex
```

---

## Quick Reproduction Using Released Results

Reviewers who only want to regenerate the paper figures and tables do not need to rerun the full scheduling and recovery sweeps.

Experiment 2:

```bash
python -m experiment2.plot_final_benchmark_event_joint
python -m experiment2.plot_final_event_joint_moneyshots
python -m experiment2.plot_replica_vs_cpsat_competitiveness
python -m experiment2.generate_final_median_table
```

Experiment 3:

```bash
python -m experiment3.analyze_rq3_final_replanned
python -m experiment3.plot_rq3_paper_moneyshots
python -m experiment3.plot_final_trusted_service_moneyshot
python -m experiment3.plot_rq3_replica_ablation_moneyshots
```

These commands regenerate the paper tables and figures from the released final result files.
