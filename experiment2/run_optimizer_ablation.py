from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from experiment2.common import (
    Task,
    Resource,
    ResourceState,
    SystemState,
    Workflow,
    ExecutionProfile,
)

from experiment2.replica_scheduler_optimizer_ablation import ReplicaOptimizerAblationScheduler
from experiment2.heft_scheduler import HEFTScheduler
from experiment2.easy_scheduler import EASYBackfillScheduler
from experiment2.cpsat_scheduler import CPSATScheduler


# ============================================================
# PATHS
# ============================================================

PROJECT = Path("/workspace/sc26_rebuttal")

GENERATED = (
    PROJECT
    / "experiment2"
    / "generated"
)

PROFILE_PATH = (
    GENERATED
    / "execution_profiles.json"
)

SCENARIO_DIR = (
    GENERATED
    / "empirical_rho_ladder"
)

MANIFEST_PATH = (
    GENERATED
    / "empirical_rho_ladder_manifest.csv"
)

LADDER_METADATA_PATH = (
    GENERATED
    / "empirical_rho_ladder_metadata.json"
)

DEFAULT_OUTPUT = (
    GENERATED
    / "optimizer_ablation"
)


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

MODELS = [
    "random_forest",
    "xgboost",
    "lstm",
    "yolo11n",
    "yolov5s",
]

STAGES = [
    "update",
    "evaluate",
    "package",
    "deploy",
    "validate",
    "reactivate",
]

RESOURCE_IDS = [
    "A100",
    "RTX4090",
    "RTX5090",
]

SCHEDULERS = [
    "replica_ablation_base",
    "replica_joint",
    "replica_event_joint",
    "replica_sufferage",
    "replica_scarcity",
    "replica_dominance",
    "replica_memo",
    "replica_adaptive_beam",
    "replica_adaptive_depth",
    "replica_reservation",
    "cpsat",
]


# ============================================================
# REPRODUCIBILITY HELPERS
# ============================================================

def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def git_commit():

    try:

        return (
            subprocess
            .check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=PROJECT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )

    except Exception:
        return None


# ============================================================
# LOAD EXPERIMENT-1 EXECUTION PROFILES
# ============================================================

def load_profiles():

    rows = json.loads(
        PROFILE_PATH.read_text()
    )

    profiles = {}

    for row in rows:

        p = ExecutionProfile(
            model=row["model"],
            stage=row["stage"],
            resource_id=row["resource_id"],

            runtime_sec=float(
                row["runtime_sec"]
            ),

            cpu_peak_cores=float(
                row.get(
                    "cpu_peak_cores",
                    0.0,
                )
            ),

            ram_peak_mb=float(
                row.get(
                    "ram_peak_mb",
                    0.0,
                )
            ),

            gpu_mean_pct=float(
                row.get(
                    "gpu_mean_pct",
                    0.0,
                )
            ),

            gpu_peak_pct=float(
                row.get(
                    "gpu_peak_pct",
                    0.0,
                )
            ),

            vram_peak_mb=float(
                row.get(
                    "vram_peak_mb",
                    0.0,
                )
            ),

            disk_read_mb=float(
                row.get(
                    "disk_read_mb",
                    0.0,
                )
            ),

            disk_write_mb=float(
                row.get(
                    "disk_write_mb",
                    0.0,
                )
            ),

            artifact_size_mb=float(
                row.get(
                    "artifact_size_mb",
                    0.0,
                )
            ),
        )

        profiles[
            (
                p.model,
                p.stage,
                p.resource_id,
            )
        ] = p

    return profiles


# ============================================================
# PHYSICAL RESOURCES
# ============================================================

def build_resources():

    return {
        "A100": Resource(
            resource_id="A100",
            resource_type="gpu",
            cpu_cores=64,
            ram_mb=128_000,
            has_gpu=True,
            gpu_name="A100",
            vram_mb=80_000,
        ),

        "RTX4090": Resource(
            resource_id="RTX4090",
            resource_type="gpu",
            cpu_cores=64,
            ram_mb=128_000,
            has_gpu=True,
            gpu_name="RTX4090",
            vram_mb=24_000,
        ),

        "RTX5090": Resource(
            resource_id="RTX5090",
            resource_type="gpu",
            cpu_cores=64,
            ram_mb=128_000,
            has_gpu=True,
            gpu_name="RTX5090",
            vram_mb=32_000,
        ),
    }


def build_state(resources):

    return SystemState(
        time_sec=0.0,

        resources={
            rid: ResourceState(
                resource_id=rid,
                available=True,
            )
            for rid in resources
        },
    )


# ============================================================
# SAME 30-TASK WORKFLOW FOR EVERY RUN
# ============================================================

def build_workflow(
    profiles,
    scenario_id,
):

    tasks = {}

    for model in MODELS:

        previous = None

        for stage in STAGES:

            task_id = (
                f"{model}:{stage}"
            )

            predecessors = (
                (previous,)
                if previous is not None
                else ()
            )

            candidate_profiles = [
                p
                for (m, s, _), p
                in profiles.items()
                if m == model
                and s == stage
            ]

            min_cpu = (
                min(
                    p.cpu_peak_cores
                    for p
                    in candidate_profiles
                )
                if candidate_profiles
                else 0.0
            )

            min_ram = (
                min(
                    p.ram_peak_mb
                    for p
                    in candidate_profiles
                )
                if candidate_profiles
                else 0.0
            )

            min_vram = (
                min(
                    p.vram_peak_mb
                    for p
                    in candidate_profiles
                )
                if candidate_profiles
                else 0.0
            )

            tasks[task_id] = Task(
                task_id=task_id,
                model=model,
                stage=stage,

                predecessors=
                    predecessors,

                min_cpu_cores=
                    min_cpu,

                min_ram_mb=
                    min_ram,

                min_vram_mb=
                    min_vram,
            )

            previous = task_id

    return Workflow(
        workflow_id=(
            f"final_{scenario_id}"
        ),
        tasks=tasks,
    )


# ============================================================
# LOAD EMPIRICAL SCENARIO
# ============================================================

def load_scenario(path):

    scenario = json.loads(
        path.read_text()
    )

    feasible = {
        (
            x["model"],
            x["stage"],
            x["resource_id"],
        )
        for x in scenario["feasible"]
    }

    return scenario, feasible


# ============================================================
# SCHEDULER FACTORY
# ============================================================

def make_scheduler(
    scheduler_name,
    profiles,
    feasible,
    replica_engine,
    cpsat_time_limit,
):

    if scheduler_name == "replica_ablation_base":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
        )

    if scheduler_name == "replica_joint":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            joint_frontier=True,
        )

    if scheduler_name == "replica_event_joint":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            joint_frontier=True,
            event_driven=True,
        )

    if scheduler_name == "replica_sufferage":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            conditional_sufferage=True,
        )

    if scheduler_name == "replica_scarcity":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            scarcity=True,
        )

    if scheduler_name == "replica_dominance":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            dominance_pruning=True,
        )

    if scheduler_name == "replica_memo":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            memoization=True,
        )

    if scheduler_name == "replica_adaptive_beam":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            adaptive_beam=True,
        )

    if scheduler_name == "replica_adaptive_depth":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            adaptive_depth=True,
        )

    if scheduler_name == "replica_reservation":
        return ReplicaOptimizerAblationScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
            engine_name=replica_engine,
            resource_reservation=True,
        )


    if scheduler_name == "heft":

        return HEFTScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
        )


    if scheduler_name == "easy":

        return EASYBackfillScheduler(
            profiles=profiles,
            feasible_assignments=feasible,
        )


    if scheduler_name == "cpsat":

        return CPSATScheduler(
            time_limit_sec=
                cpsat_time_limit,
        )


    raise ValueError(
        scheduler_name
    )


# ============================================================
# NORMALIZE SCHEDULER OUTPUT
# ============================================================

def execute_scheduler(
    scheduler_name,
    scheduler,
    workflow,
    resources,
    state,
    profiles,
    feasible,
):

    if scheduler_name == "cpsat":

        raw = scheduler.schedule(
            tasks=list(
                workflow.tasks.values()
            ),

            profiles=list(
                profiles.values()
            ),

            feasible=feasible,
        )

        assignments = [
            {
                "task_id":
                    a["task_id"],

                "resource_id":
                    a["resource_id"],

                "start_sec":
                    float(
                        a["start_sec"]
                    ),

                "end_sec":
                    float(
                        a["end_sec"]
                    ),

                "runtime_sec":
                    float(
                        a["runtime_sec"]
                    ),
            }

            for a in
            raw["assignments"]
        ]

        return {
            "success":
                bool(
                    raw["success"]
                ),

            "makespan_sec":
                float(
                    raw["makespan_sec"]
                )
                if raw[
                    "makespan_sec"
                ] is not None
                else None,

            "scheduling_overhead_sec":
                float(
                    raw[
                        "scheduling_overhead_sec"
                    ]
                ),

            "assignments":
                assignments,

            "solver_status":
                raw.get(
                    "solver_status"
                ),
        }


    result = scheduler.schedule(
        workflow=workflow,
        resources=resources,
        state=state,
    )

    assignments = [
        {
            "task_id":
                a.task_id,

            "resource_id":
                a.resource_id,

            "start_sec":
                float(
                    a.start_sec
                ),

            "end_sec":
                float(
                    a.end_sec
                ),

            "runtime_sec":
                float(
                    a.estimated_runtime_sec
                ),
        }

        for a in
        result.assignments
    ]


    return {
        "success":
            bool(
                result.success
            ),

        "makespan_sec":
            (
                float(
                    result.makespan_sec
                )
                if result.makespan_sec
                is not None
                else None
            ),

        "scheduling_overhead_sec":
            float(
                result.scheduling_overhead_sec
            ),

        "assignments":
            assignments,

        "solver_status":
            None,

        # REPLICA-specific diagnostics.
        #
        # For non-symbolic schedulers these remain None.
        "symbolic_planning_overhead_sec":
            getattr(
                scheduler,
                "last_symbolic_planning_overhead_sec",
                None,
            ),

        "planner_calls":
            getattr(
                scheduler,
                "last_planner_calls",
                None,
            ),

        "decision_epochs":
            getattr(
                scheduler,
                "last_decision_epochs",
                None,
            ),

        "decision_trace":
            getattr(
                scheduler,
                "last_decision_trace",
                None,
            ),

        "candidate_trace":
            getattr(
                scheduler,
                "last_candidate_trace",
                None,
            ),

        "joint_depth_trace":
            getattr(
                scheduler,
                "last_joint_depth_trace",
                None,
            ),
    }


# ============================================================
# VALIDATION
# ============================================================

def validate(
    result,
    workflow,
    feasible,
):

    errors = []

    assignments = result[
        "assignments"
    ]

    by_task = {
        a["task_id"]: a
        for a in assignments
    }


    # --------------------------------------------------------
    # Complete assignment set
    # --------------------------------------------------------

    if len(assignments) != len(
        workflow.tasks
    ):

        errors.append(
            "assignment_count"
        )


    # --------------------------------------------------------
    # Resource overlap
    # --------------------------------------------------------

    by_resource = defaultdict(
        list
    )

    for a in assignments:

        by_resource[
            a["resource_id"]
        ].append(a)


    for rid, jobs in (
        by_resource.items()
    ):

        jobs = sorted(
            jobs,
            key=lambda x:
                x["start_sec"],
        )

        for previous, current in zip(
            jobs,
            jobs[1:],
        ):

            if (
                current[
                    "start_sec"
                ]
                <
                previous[
                    "end_sec"
                ]
                - 1e-9
            ):

                errors.append(
                    (
                        "resource_overlap:"
                        f"{rid}"
                    )
                )


    # --------------------------------------------------------
    # Workflow precedence
    # --------------------------------------------------------

    for task_id, task in (
        workflow.tasks.items()
    ):

        if task_id not in by_task:
            continue

        current = by_task[
            task_id
        ]

        for predecessor in (
            task.predecessors
        ):

            if predecessor not in by_task:

                errors.append(
                    (
                        "missing_predecessor:"
                        f"{predecessor}"
                    )
                )

                continue


            previous = by_task[
                predecessor
            ]

            if (
                current[
                    "start_sec"
                ]
                <
                previous[
                    "end_sec"
                ]
                - 1e-9
            ):

                errors.append(
                    (
                        "precedence:"
                        f"{predecessor}"
                        f"->{task_id}"
                    )
                )


    # --------------------------------------------------------
    # Empirical feasibility
    # --------------------------------------------------------

    for a in assignments:

        task = workflow.tasks[
            a["task_id"]
        ]

        key = (
            task.model,
            task.stage,
            a["resource_id"],
        )

        if key not in feasible:

            errors.append(
                (
                    "infeasible:"
                    f"{a['task_id']}"
                    f"->{a['resource_id']}"
                )
            )


    return {
        "pass":
            len(errors) == 0,

        "errors":
            errors,
    }


# ============================================================
# RESULT METRICS
# ============================================================

def schedule_metrics(
    result,
    workflow,
):

    assignments = result[
        "assignments"
    ]

    counts = Counter()

    busy = defaultdict(
        float
    )

    switches = 0

    previous_resource = {}


    for a in assignments:

        rid = a[
            "resource_id"
        ]

        counts[rid] += 1

        busy[rid] += (
            a["end_sec"]
            - a["start_sec"]
        )

        task = workflow.tasks[
            a["task_id"]
        ]

        if task.model in (
            previous_resource
        ):

            if (
                previous_resource[
                    task.model
                ]
                != rid
            ):

                switches += 1

        previous_resource[
            task.model
        ] = rid


    makespan = result[
        "makespan_sec"
    ]

    utilization = {}

    for rid in RESOURCE_IDS:

        utilization[rid] = (
            busy[rid] / makespan
            if makespan
            and makespan > 0
            else 0.0
        )


    return {
        "a100_tasks":
            counts["A100"],

        "rtx4090_tasks":
            counts["RTX4090"],

        "rtx5090_tasks":
            counts["RTX5090"],

        "a100_busy_sec":
            busy["A100"],

        "rtx4090_busy_sec":
            busy["RTX4090"],

        "rtx5090_busy_sec":
            busy["RTX5090"],

        "a100_utilization":
            utilization["A100"],

        "rtx4090_utilization":
            utilization[
                "RTX4090"
            ],

        "rtx5090_utilization":
            utilization[
                "RTX5090"
            ],

        "mean_resource_utilization":
            sum(
                utilization.values()
            )
            / len(
                RESOURCE_IDS
            ),

        "max_resource_utilization":
            max(
                utilization.values()
            ),

        "resource_switches":
            switches,
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    rows,
    path,
):

    if not rows:
        return

    fields = sorted({
        key
        for row in rows
        for key in row
    })

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = (
            csv.DictWriter(
                f,
                fieldnames=fields,
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=SCHEDULERS,
        choices=SCHEDULERS,
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )


    parser.add_argument(
        "--replica-engine",
        default="enhsp-opt",
    )


    parser.add_argument(
        "--cpsat-time-limit",
        type=float,
        default=60.0,
    )


    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help=(
            "Optional target rho filter. "
            "Example: --rho 0.5"
        ),
    )


    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help=(
            "Optional exact scenario_id filter. "
            "Example: moderate_13__rho_0.500"
        ),
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
    )


    args = parser.parse_args()


    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_dir = (
        args.output_dir
        / "runs"
    )

    runs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # INPUTS
    # ========================================================

    profiles = load_profiles()

    resources = build_resources()

    manifest = pd.read_csv(
        MANIFEST_PATH
    )


    # Optional rho slice for development ablations.
    #
    # Use target_rho so every variant receives the exact same
    # predefined scenario set for that contraction level.
    if args.rho is not None:

        manifest = manifest[
            (
                manifest["target_rho"].astype(float)
                - float(args.rho)
            ).abs()
            < 1e-9
        ].reset_index(
            drop=True
        )

        if manifest.empty:
            raise ValueError(
                f"No scenarios found for rho={args.rho}"
            )


    # Optional exact scenario filter.
    if args.scenario_id is not None:

        manifest = manifest[
            manifest[
                "scenario_id"
            ].astype(str)
            ==
            str(
                args.scenario_id
            )
        ].reset_index(
            drop=True
        )

        if manifest.empty:
            raise ValueError(
                "No scenario found with "
                f"scenario_id={args.scenario_id}"
            )


    if args.limit is not None:

        manifest = manifest.iloc[
            :args.limit
        ]


    total_runs = (
        len(manifest)
        * len(
            args.schedulers
        )
    )


    # ========================================================
    # REPRODUCIBILITY METADATA
    # ========================================================

    metadata = {
        "created_unix_sec":
            time.time(),

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "git_commit":
            git_commit(),

        "profile_path":
            str(
                PROFILE_PATH
            ),

        "profile_sha256":
            sha256(
                PROFILE_PATH
            ),

        "manifest_path":
            str(
                MANIFEST_PATH
            ),

        "manifest_sha256":
            sha256(
                MANIFEST_PATH
            ),

        "ladder_metadata_path":
            str(
                LADDER_METADATA_PATH
            ),

        "ladder_metadata_sha256":
            sha256(
                LADDER_METADATA_PATH
            ),

        "scheduler_names":
            args.schedulers,

        "replica_engine":
            args.replica_engine,

        "cpsat_time_limit_sec":
            args.cpsat_time_limit,

        "scenario_count":
            len(manifest),

        "total_runs":
            total_runs,
    }


    (
        args.output_dir
        / "benchmark_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


    print()
    print("=" * 100)
    print("FINAL EMPIRICAL HPC SCHEDULING BENCHMARK")
    print("=" * 100)

    print(
        "Scenarios:",
        len(manifest),
    )

    print(
        "Schedulers:",
        args.schedulers,
    )

    print(
        "Total runs:",
        total_runs,
    )

    print("=" * 100)


    master = []

    run_index = 0


    # ========================================================
    # BENCHMARK
    # ========================================================

    for _, manifest_row in (
        manifest.iterrows()
    ):

        scenario_id = str(
            manifest_row[
                "scenario_id"
            ]
        )

        scenario_path = (
            SCENARIO_DIR
            / f"{scenario_id}.json"
        )


        scenario, feasible = (
            load_scenario(
                scenario_path
            )
        )


        workflow = (
            build_workflow(
                profiles,
                scenario_id,
            )
        )


        for scheduler_name in (
            args.schedulers
        ):

            run_index += 1


            print(
                f"[{run_index:04d}/"
                f"{total_runs:04d}] "
                f"{scheduler_name:8s} "
                f"| {scenario_id}"
            )


            result_path = (
                runs_dir
                / (
                    f"{scheduler_name}"
                    f"__"
                    f"{scenario_id}"
                    f".json"
                )
            )


            # ------------------------------------------------
            # Resume support
            # ------------------------------------------------

            if (
                result_path.exists()
                and not args.overwrite
            ):

                artifact = (
                    json.loads(
                        result_path
                        .read_text()
                    )
                )

                master.append(
                    artifact[
                        "summary"
                    ]
                )

                print(
                    "    reused"
                )

                continue


            state = build_state(
                resources
            )


            scheduler = (
                make_scheduler(
                    scheduler_name=
                        scheduler_name,

                    profiles=
                        profiles,

                    feasible=
                        feasible,

                    replica_engine=
                        args.replica_engine,

                    cpsat_time_limit=
                        args.cpsat_time_limit,
                )
            )


            wall_start = (
                time.perf_counter()
            )


            exception = None


            try:

                result = (
                    execute_scheduler(
                        scheduler_name=
                            scheduler_name,

                        scheduler=
                            scheduler,

                        workflow=
                            workflow,

                        resources=
                            resources,

                        state=
                            state,

                        profiles=
                            profiles,

                        feasible=
                            feasible,
                    )
                )


            except Exception as e:

                exception = (
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                result = {
                    "success":
                        False,

                    "makespan_sec":
                        None,

                    "scheduling_overhead_sec":
                        None,

                    "assignments":
                        [],

                    "solver_status":
                        None,
                }


            wall_sec = (
                time.perf_counter()
                - wall_start
            )


            validation = validate(
                result=result,
                workflow=workflow,
                feasible=feasible,
            )


            metrics = schedule_metrics(
                result=result,
                workflow=workflow,
            )


            summary = {
                "scheduler":
                    scheduler_name,

                "scenario_id":
                    scenario_id,

                "source_case_id":
                    scenario[
                        "source_case_id"
                    ],

                "source_regime":
                    scenario[
                        "source_regime"
                    ],

                "target_rho":
                    scenario[
                        "target_rho"
                    ],

                "actual_rho":
                    scenario[
                        "actual_rho"
                    ],

                "feasible_assignments":
                    scenario[
                        "feasible_assignments"
                    ],

                "removed_assignments":
                    scenario[
                        "removed_assignments"
                    ],

                "tasks_with_1_option":
                    scenario[
                        "tasks_with_1_option"
                    ],

                "tasks_with_2_options":
                    scenario[
                        "tasks_with_2_options"
                    ],

                "tasks_with_3_options":
                    scenario[
                        "tasks_with_3_options"
                    ],

                "mean_removed_max_stress":
                    scenario[
                        "mean_removed_max_stress"
                    ],

                "mean_retained_max_stress":
                    scenario[
                        "mean_retained_max_stress"
                    ],

                "success":
                    result[
                        "success"
                    ],

                "validation_pass":
                    validation[
                        "pass"
                    ],

                "validation_error_count":
                    len(
                        validation[
                            "errors"
                        ]
                    ),

                "makespan_sec":
                    result[
                        "makespan_sec"
                    ],

                "scheduling_overhead_sec":
                    result[
                        "scheduling_overhead_sec"
                    ],

                "run_wall_sec":
                    wall_sec,

                # Total scheduler computation is stored in
                # scheduling_overhead_sec for every scheduler.
                #
                # REPLICA additionally reports the portion spent
                # constructing/solving symbolic planning models.
                "symbolic_planning_overhead_sec":
                    result.get(
                        "symbolic_planning_overhead_sec"
                    ),

                "planner_calls":
                    result.get(
                        "planner_calls"
                    ),

                "decision_epochs":
                    result.get(
                        "decision_epochs"
                    ),

                "n_assignments":
                    len(
                        result[
                            "assignments"
                        ]
                    ),

                "solver_status":
                    result.get(
                        "solver_status"
                    ),

                "exception":
                    exception,

                **metrics,
            }


            artifact = {
                "summary":
                    summary,

                "result":
                    result,

                "validation":
                    validation,

                "scenario":
                    scenario,
            }


            result_path.write_text(
                json.dumps(
                    artifact,
                    indent=2,
                )
            )


            master.append(
                summary
            )


            print(
                "    "
                f"success={summary['success']} "
                f"valid={summary['validation_pass']} "
                f"makespan={summary['makespan_sec']} "
                f"overhead={summary['scheduling_overhead_sec']}"
            )


            # Incremental save
            save_csv(
                master,
                args.output_dir
                / "final_benchmark_master.csv",
            )


    # ========================================================
    # FINAL OUTPUTS
    # ========================================================

    save_csv(
        master,
        args.output_dir
        / "final_benchmark_master.csv",
    )


    (
        args.output_dir
        / "final_benchmark_master.json"
    ).write_text(
        json.dumps(
            master,
            indent=2,
        )
    )


    total_valid = sum(
        bool(
            row[
                "validation_pass"
            ]
        )
        for row in master
    )


    total_success = sum(
        bool(
            row[
                "success"
            ]
        )
        for row in master
    )


    print()
    print("=" * 100)
    print("FINAL BENCHMARK COMPLETE")
    print("=" * 100)

    print(
        "Runs:",
        len(master),
    )

    print(
        "Successful:",
        f"{total_success}/"
        f"{len(master)}",
    )

    print(
        "Validated:",
        f"{total_valid}/"
        f"{len(master)}",
    )

    print()
    print(
        "Master CSV:"
    )

    print(
        args.output_dir
        / "final_benchmark_master.csv"
    )

    print()
    print(
        "Metadata:"
    )

    print(
        args.output_dir
        / "benchmark_metadata.json"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
