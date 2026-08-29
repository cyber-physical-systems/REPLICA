#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiment2.common import (
    ExecutionProfile,
    Resource,
    ResourceState,
    SystemState,
    Task,
    Workflow,
)

from experiment2.heft_scheduler import HEFTScheduler
from experiment2.easy_scheduler import EASYBackfillScheduler
from experiment2.cpsat_scheduler import CPSATScheduler

from experiment2.replica_scheduler_optimizer_ablation import (
    ReplicaOptimizerAblationScheduler,
)

from experiment3.empirical_recovery_profiles import (
    load_execution_profiles,
)

from experiment3.replica_service_recovery_planner import (
    build_service_recovery_problem,
    build_background_preferred_repair_problem,
)

from unified_planning.shortcuts import OneshotPlanner


ROOT = Path("/workspace/sc26_rebuttal")

CASES_PATH = (
    ROOT
    / "experiment3/generated/rq3_final_multimask/"
      "rq3_final_cases.csv"
)

PROFILE_PATH = (
    ROOT
    / "experiment2/generated/execution_profiles.json"
)

SCENARIO_ROOT = (
    ROOT
    / "experiment2/generated/scenarios"
)

OUTDIR = (
    ROOT
    / "experiment3/generated/rq3_replica_ablation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUTDIR
    / "rq3_replica_ablation_results.csv"
)

FAILURES_PATH = (
    OUTDIR
    / "rq3_replica_ablation_exceptions.jsonl"
)

RESOURCES = [
    "A100",
    "RTX4090",
    "RTX5090",
]


# ======================================================================
# LOAD EXECUTION PROFILES
# ======================================================================

raw_profiles = json.loads(
    PROFILE_PATH.read_text()
)

profile_dict = {}
profile_list = []

for row in raw_profiles:

    p = ExecutionProfile(
        model=str(row["model"]),
        stage=str(row["stage"]),
        resource_id=str(row["resource_id"]),
        runtime_sec=float(row["runtime_sec"]),
        cpu_peak_cores=float(row.get("cpu_peak_cores", 0.0)),
        ram_peak_mb=float(row.get("ram_peak_mb", 0.0)),
        gpu_mean_pct=float(row.get("gpu_mean_pct", 0.0)),
        gpu_peak_pct=float(row.get("gpu_peak_pct", 0.0)),
        vram_peak_mb=float(row.get("vram_peak_mb", 0.0)),
        disk_read_mb=float(row.get("disk_read_mb", 0.0)),
        disk_write_mb=float(row.get("disk_write_mb", 0.0)),
        artifact_size_mb=float(row.get("artifact_size_mb", 0.0)),
    )

    profile_dict[
        (
            p.model,
            p.stage,
            p.resource_id,
        )
    ] = p

    profile_list.append(p)


# ======================================================================
# HARDWARE
# ======================================================================

resources = {
    "A100":
        Resource(
            resource_id="A100",
            resource_type="gpu",
            cpu_cores=128,
            ram_mb=1_000_000,
            has_gpu=True,
            gpu_name="A100",
            vram_mb=80_000,
        ),

    "RTX4090":
        Resource(
            resource_id="RTX4090",
            resource_type="gpu",
            cpu_cores=128,
            ram_mb=1_000_000,
            has_gpu=True,
            gpu_name="RTX4090",
            vram_mb=24_000,
        ),

    "RTX5090":
        Resource(
            resource_id="RTX5090",
            resource_type="gpu",
            cpu_cores=128,
            ram_mb=1_000_000,
            has_gpu=True,
            gpu_name="RTX5090",
            vram_mb=32_000,
        ),
}


def fresh_state():
    return SystemState(
        time_sec=0.0,
        resources={
            rid:
                ResourceState(
                    resource_id=rid,
                    available=True,
                    network_available=True,
                    busy_until_sec=0.0,
                )
            for rid in RESOURCES
        },
    )


# ======================================================================
# RHO SCENARIO
# ======================================================================

def load_scenario_feasible(
    scenario_path,
):
    """
    Load the exact frozen Experiment-2 resource world assigned
    to this RQ3 case.

    This deliberately uses scenario_path rather than rho because
    multiple masks at the same rho represent different feasible
    task-resource topologies.
    """

    path = ROOT / str(
        scenario_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Scenario does not exist: {path}"
        )

    raw = json.loads(
        path.read_text()
    )

    # Normal Experiment-2 scenario representation.
    if isinstance(raw, dict):

        entries = raw.get(
            "feasible",
            raw.get(
                "feasible_assignments",
                [],
            ),
        )

    elif isinstance(raw, list):

        entries = raw

    else:

        raise TypeError(
            f"Unsupported scenario format "
            f"{type(raw)} in {path}"
        )

    feasible = set()

    for item in entries:

        if isinstance(
            item,
            dict,
        ):

            feasible.add(
                (
                    str(item["model"]),
                    str(item["stage"]),
                    str(item["resource_id"]),
                )
            )

        elif (
            isinstance(
                item,
                (list, tuple),
            )
            and len(item) >= 3
        ):

            feasible.add(
                (
                    str(item[0]),
                    str(item[1]),
                    str(item[2]),
                )
            )

        else:

            raise TypeError(
                f"Unsupported feasible assignment "
                f"in {path}: {item!r}"
            )

    if not feasible:

        raise RuntimeError(
            f"No feasible assignments found "
            f"in scenario {path}"
        )

    return feasible


# ======================================================================
# WORKFLOW
# ======================================================================

def build_repair_workflow(model: str):

    stages = [
        "update",
        "evaluate",
        "package",
        "deploy",
        "validate",
    ]

    tasks = {}
    prev = None

    for stage in stages:

        tid = f"{model}:{stage}"

        tasks[tid] = Task(
            task_id=tid,
            model=model,
            stage=stage,
            predecessors=()
            if prev is None
            else (prev,),
            requires_gpu=True,
            allowed_resource_types=("gpu",),
        )

        prev = tid

    return Workflow(
        workflow_id=f"{model}_repair",
        tasks=tasks,
    )


# ======================================================================
# COMMON RESULT HELPERS
# ======================================================================

def quality_weighted_availability(
    clean_quality,
    attacked_quality,
    interim_quality,
    restored_quality,
    service_restore_sec,
    preferred_restore_sec,
    direction,
):

    horizon = max(
        preferred_restore_sec,
        1e-9,
    )

    if direction == "higher_is_better":

        q_attacked = (
            attacked_quality
            / clean_quality
            if clean_quality != 0
            else 0.0
        )

        q_interim = (
            interim_quality
            / clean_quality
            if clean_quality != 0
            else 0.0
        )

    else:

        q_attacked = (
            clean_quality
            / attacked_quality
            if attacked_quality > 0
            else 0.0
        )

        q_interim = (
            clean_quality
            / interim_quality
            if interim_quality > 0
            else 0.0
        )

    q_attacked = float(
        np.clip(
            q_attacked,
            0.0,
            1.0,
        )
    )

    q_interim = float(
        np.clip(
            q_interim,
            0.0,
            1.0,
        )
    )

    area = (
        q_attacked
        * service_restore_sec
        +
        q_interim
        * max(
            preferred_restore_sec
            - service_restore_sec,
            0.0,
        )
    )

    return area / horizon


def profile_metrics(model, stage, resource):

    p = profile_dict.get(
        (
            model,
            stage,
            resource,
        )
    )

    if p is None:
        return {}

    return {
        "gpu_mean_pct":
            p.gpu_mean_pct,

        "gpu_peak_pct":
            p.gpu_peak_pct,

        "ram_peak_mb":
            p.ram_peak_mb,

        "vram_peak_mb":
            p.vram_peak_mb,
    }


# ======================================================================
# BASELINE SCHEDULERS
# ======================================================================

def run_baseline(
    method,
    model,
    feasible,
):

    workflow = build_repair_workflow(
        model
    )

    state = fresh_state()

    if method == "heft":

        scheduler = HEFTScheduler(
            profiles=profile_dict,
            feasible_assignments=feasible,
        )

        result = scheduler.schedule(
            workflow,
            resources,
            state,
        )

        assignments = [
            {
                "stage":
                    a.task_id.split(
                        ":",
                        1,
                    )[1],

                "resource":
                    a.resource_id,

                "runtime_sec":
                    float(
                        a.estimated_runtime_sec
                    ),
            }
            for a in result.assignments
        ]

        return {
            "success":
                bool(result.success),

            "recovery_sec":
                float(
                    result.makespan_sec
                    or 0.0
                ),

            "overhead_sec":
                float(
                    result.scheduling_overhead_sec
                ),

            "assignments":
                assignments,
        }


    if method == "easy":

        scheduler = EASYBackfillScheduler(
            profiles=profile_dict,
            feasible_assignments=feasible,
        )

        result = scheduler.schedule(
            workflow,
            resources,
            state,
        )

        assignments = [
            {
                "stage":
                    a.task_id.split(
                        ":",
                        1,
                    )[1],

                "resource":
                    a.resource_id,

                "runtime_sec":
                    float(
                        a.estimated_runtime_sec
                    ),
            }
            for a in result.assignments
        ]

        return {
            "success":
                bool(result.success),

            "recovery_sec":
                float(
                    result.makespan_sec
                    or 0.0
                ),

            "overhead_sec":
                float(
                    result.scheduling_overhead_sec
                ),

            "assignments":
                assignments,
        }


    if method == "cpsat":

        scheduler = CPSATScheduler(
            time_limit_sec=60.0
        )

        result = scheduler.schedule(
            tasks=list(
                workflow.tasks.values()
            ),
            profiles=profile_list,
            feasible=feasible,
        )

        assignments = [
            {
                "stage":
                    a["stage"],

                "resource":
                    a["resource_id"],

                "runtime_sec":
                    float(
                        a["runtime_sec"]
                    ),
            }
            for a in result[
                "assignments"
            ]
        ]

        return {
            "success":
                bool(
                    result["success"]
                ),

            "recovery_sec":
                float(
                    result[
                        "makespan_sec"
                    ]
                ),

            "overhead_sec":
                float(
                    result[
                        "scheduling_overhead_sec"
                    ]
                ),

            "assignments":
                assignments,
        }


    raise ValueError(method)


# ======================================================================
# REPLICA
# ======================================================================

def run_replica_event_joint(
    model,
    feasible,
):
    """
    Schedule a direct model-repair workflow using the final
    Experiment-2 REPLICA policy.

    Final REPLICA configuration:
        joint_frontier=True
        event_driven=True

    This replaces the earlier HEFT fallback used inside REPLICA.
    """

    workflow = build_repair_workflow(
        model
    )

    state = fresh_state()

    scheduler = (
        ReplicaOptimizerAblationScheduler(
            profiles=profile_dict,
            feasible_assignments=feasible,
            engine_name="enhsp-opt",
            joint_frontier=True,
            event_driven=True,
        )
    )

    result = scheduler.schedule(
        workflow,
        resources,
        state,
    )

    assignments = [
        {
            "stage":
                a.task_id.split(
                    ":",
                    1,
                )[1],

            "resource":
                a.resource_id,

            "runtime_sec":
                float(
                    a.estimated_runtime_sec
                ),

            "start_sec":
                float(
                    a.start_sec
                ),

            "end_sec":
                float(
                    a.end_sec
                ),
        }
        for a in result.assignments
    ]

    return {
        "success":
            bool(
                result.success
            ),

        "recovery_sec":
            float(
                result.makespan_sec
                or 0.0
            ),

        "overhead_sec":
            float(
                result.scheduling_overhead_sec
            ),

        "symbolic_overhead_sec":
            float(
                getattr(
                    scheduler,
                    "last_symbolic_planning_overhead_sec",
                    0.0,
                )
                or 0.0
            ),

        "planner_calls":
            int(
                getattr(
                    scheduler,
                    "last_planner_calls",
                    0,
                )
                or 0
            ),

        "decision_epochs":
            int(
                getattr(
                    scheduler,
                    "last_decision_epochs",
                    0,
                )
                or 0
            ),

        "assignments":
            assignments,
    }


def solve_up(
    problem,
    metadata,
):

    t0 = time.perf_counter()

    with OneshotPlanner(
        name="enhsp-opt"
    ) as planner:

        result = planner.solve(
            problem
        )

    overhead = (
        time.perf_counter()
        - t0
    )

    steps = []
    execution_sec = 0.0

    if result.plan is not None:

        for ai in result.plan.actions:

            info = metadata[
                "action_metadata"
            ].get(
                ai.action.name,
                {},
            )

            cost_sec = (
                float(
                    info.get(
                        "cost_ms",
                        0,
                    )
                )
                / 1000.0
            )

            execution_sec += (
                cost_sec
            )

            steps.append(
                {
                    "stage":
                        info.get(
                            "stage"
                        ),

                    "resource":
                        info.get(
                            "resource"
                        ),

                    "runtime_sec":
                        cost_sec,

                    "strategy":
                        info.get(
                            "strategy"
                        ),
                }
            )

    return {
        "success":
            result.plan is not None,

        "execution_sec":
            execution_sec,

        "overhead_sec":
            overhead,

        "steps":
            steps,
    }


def run_replica_full(
    row,
    feasible,
):

    service = row["service"]

    if service == "detection":

        primary = "yolo11n"
        fallback = "yolov5s"

    else:

        primary = "lstm"
        fallback = str(
            row["fallback_model"]
        )


    # ----------------------------------------------------------
    # No recovery required
    # ----------------------------------------------------------

    if (
        row[
            "preferred_recovery_action"
        ]
        == "no_recovery_required"
    ):

        return {
            "success": True,
            "trusted_service_restore_sec": 0.0,
            "preferred_restore_sec": 0.0,
            "overhead_sec": 0.0,
            "repeated_work_sec": 0.0,
            "assignments": [],
        }


    # ----------------------------------------------------------
    # Primary repair without failover
    # ----------------------------------------------------------

    if (
        row[
            "preferred_recovery_action"
        ]
        == "repair_primary"
    ):

        baseline_like = run_replica_event_joint(
            primary,
            feasible,
        )

        return {
            "success":
                baseline_like[
                    "success"
                ],

            "trusted_service_restore_sec":
                baseline_like[
                    "recovery_sec"
                ],

            "preferred_restore_sec":
                baseline_like[
                    "recovery_sec"
                ],

            "overhead_sec":
                baseline_like[
                    "overhead_sec"
                ],

            "repeated_work_sec":
                0.0,

            "assignments":
                baseline_like[
                    "assignments"
                ],

            "recovery_strategy":
                "replica_event_joint_direct_repair",

            "replanned":
                False,
        }


    # ----------------------------------------------------------
    # Failover then repair
    # ----------------------------------------------------------

    resource_available = {
        rid: True
        for rid in RESOURCES
    }

    service_problem, service_meta = (
        build_service_recovery_problem(
            service_name=service,
            attacked_model=primary,
            substitute_model=fallback,
            substitute_quality=float(
                row["fallback_quality"]
            ),
            minimum_quality=0.0,
            resources=RESOURCES,
            resource_available=
                resource_available,
            feasible_assignments=
                feasible,
            profiles=
                load_execution_profiles(),
            substitute_available=True,
        )
    )

    # ----------------------------------------------------------
    # Preferred strategy:
    #     trusted failover -> background preferred-model repair.
    #
    # If that strategy is infeasible in the current resource
    # world, REPLICA replans to direct primary repair.
    # ----------------------------------------------------------

    immediate = solve_up(
        service_problem,
        service_meta,
    )

    total_overhead = float(
        immediate["overhead_sec"]
    )


    # ----------------------------------------------------------
    # Immediate failover itself is infeasible.
    # Replan to direct primary repair.
    # ----------------------------------------------------------

    if not immediate["success"]:

        direct = run_replica_event_joint(
            primary,
            feasible,
        )

        total_overhead += float(
            direct["overhead_sec"]
        )

        if not direct["success"]:

            return {
                "success": False,
                "trusted_service_restore_sec": np.nan,
                "preferred_restore_sec": np.nan,
                "overhead_sec": total_overhead,
                "repeated_work_sec": 0.0,
                "assignments": [],
                "recovery_strategy":
                    "failover_infeasible_direct_repair_failed",
                "replanned": True,
            }

        return {
            "success": True,

            "trusted_service_restore_sec":
                direct["recovery_sec"],

            "preferred_restore_sec":
                direct["recovery_sec"],

            "overhead_sec":
                total_overhead,

            "repeated_work_sec":
                0.0,

            "assignments":
                direct["assignments"],

            "recovery_strategy":
                "direct_repair_after_failover_infeasible",

            "replanned":
                True,
        }


    substitute_step = next(
        (
            step
            for step in immediate["steps"]
            if step.get("strategy") == "substitute"
        ),
        None,
    )


    # ----------------------------------------------------------
    # Defensive case: planner returned a plan but no substitute
    # action. Treat the failover strategy as invalid and replan.
    # ----------------------------------------------------------

    if substitute_step is None:

        direct = run_replica_event_joint(
            primary,
            feasible,
        )

        total_overhead += float(
            direct["overhead_sec"]
        )

        if not direct["success"]:

            return {
                "success": False,
                "trusted_service_restore_sec": np.nan,
                "preferred_restore_sec": np.nan,
                "overhead_sec": total_overhead,
                "repeated_work_sec": 0.0,
                "assignments": immediate["steps"],
                "recovery_strategy":
                    "invalid_failover_plan_direct_repair_failed",
                "replanned": True,
            }

        return {
            "success": True,

            "trusted_service_restore_sec":
                direct["recovery_sec"],

            "preferred_restore_sec":
                direct["recovery_sec"],

            "overhead_sec":
                total_overhead,

            "repeated_work_sec":
                0.0,

            "assignments":
                direct["assignments"],

            "recovery_strategy":
                "direct_repair_after_invalid_failover_plan",

            "replanned":
                True,
        }


    serving_resource = (
        substitute_step[
            "resource"
        ]
    )


    # ----------------------------------------------------------
    # Substitute is serving.
    # Attempt preferred-model repair in the background.
    # ----------------------------------------------------------

    background_problem, background_meta = (
        build_background_preferred_repair_problem(
            service_name=service,
            preferred_model=primary,
            active_substitute=fallback,
            active_substitute_resource=
                serving_resource,
            resources=RESOURCES,
            resource_available=
                resource_available,
            feasible_assignments=
                feasible,
            profiles=
                load_execution_profiles(),
            gpu_capacity_policy=
                "peak",
            completed_stages=set(),
        )
    )

    background = solve_up(
        background_problem,
        background_meta,
    )

    total_overhead += float(
        background["overhead_sec"]
    )


    # ----------------------------------------------------------
    # Failover worked but concurrent background repair is
    # infeasible. Keep substitute service and replan the
    # preferred model through direct repair.
    # ----------------------------------------------------------

    if not background["success"]:

        direct = run_replica_event_joint(
            primary,
            feasible,
        )

        total_overhead += float(
            direct["overhead_sec"]
        )

        if not direct["success"]:

            return {
                "success": False,

                "trusted_service_restore_sec":
                    immediate["execution_sec"],

                "preferred_restore_sec":
                    np.nan,

                "overhead_sec":
                    total_overhead,

                "repeated_work_sec":
                    0.0,

                "assignments":
                    immediate["steps"],

                "recovery_strategy":
                    "failover_succeeded_primary_repair_failed",

                "replanned":
                    True,
            }

        return {
            "success": True,

            "trusted_service_restore_sec":
                immediate["execution_sec"],

            "preferred_restore_sec":
                immediate["execution_sec"]
                + direct["recovery_sec"],

            "overhead_sec":
                total_overhead,

            "repeated_work_sec":
                0.0,

            "assignments":
                immediate["steps"]
                + direct["assignments"],

            "recovery_strategy":
                "failover_then_direct_repair_after_background_infeasible",

            "replanned":
                True,
        }


    # ----------------------------------------------------------
    # Preferred strategy succeeded normally.
    # ----------------------------------------------------------

    return {
        "success": True,

        "trusted_service_restore_sec":
            immediate["execution_sec"],

        "preferred_restore_sec":
            immediate["execution_sec"]
            + background["execution_sec"],

        "overhead_sec":
            total_overhead,

        "repeated_work_sec":
            0.0,

        "assignments":
            immediate["steps"]
            + background["steps"],

        "recovery_strategy":
            "failover_then_background_repair",

        "replanned":
            False,
    }



# ======================================================================
# REPLICA FRAMEWORK ABLATION
# ======================================================================

ABLATION_VARIANTS = [
    "full_replica",
    "no_strategy_selection",
    "no_state_reconstruction",
    "no_replanning",
    "no_closed_loop_adaptation",
]


def _models(row):
    service = str(row["service"])
    if service == "detection":
        return "yolo11n", "yolov5s"
    return "lstm", str(row["fallback_model"])


def _direct_primary(row, feasible, strategy):
    primary, _ = _models(row)
    direct = run_replica_event_joint(primary, feasible)
    return {
        "success": bool(direct["success"]),
        "trusted_service_restore_sec":
            float(direct["recovery_sec"]) if direct["success"] else np.nan,
        "preferred_restore_sec":
            float(direct["recovery_sec"]) if direct["success"] else np.nan,
        "overhead_sec": float(direct["overhead_sec"]),
        "repeated_work_sec": 0.0,
        "assignments": direct["assignments"],
        "recovery_strategy": strategy,
        "replanned": False,
        "ablation_failure_reason":
            "" if direct["success"] else "direct_primary_repair_infeasible",
    }


def _failover_once(row, feasible, allow_replan, attempt_background=True):
    """
    Use the observed case state once.  This is the shared primitive for
    ablations that remove later adaptation/replanning while preserving
    the same PDDL recovery model used by the final Experiment 3 driver.
    """
    service = str(row["service"])
    primary, fallback = _models(row)
    resource_available = {rid: True for rid in RESOURCES}

    service_problem, service_meta = build_service_recovery_problem(
        service_name=service,
        attacked_model=primary,
        substitute_model=fallback,
        substitute_quality=float(row["fallback_quality"]),
        minimum_quality=0.0,
        resources=RESOURCES,
        resource_available=resource_available,
        feasible_assignments=feasible,
        profiles=load_execution_profiles(),
        substitute_available=True,
    )

    immediate = solve_up(service_problem, service_meta)
    total_overhead = float(immediate["overhead_sec"])

    substitute_step = next(
        (s for s in immediate["steps"] if s.get("strategy") == "substitute"),
        None,
    )

    if (not immediate["success"]) or substitute_step is None:
        if allow_replan:
            direct = _direct_primary(
                row, feasible, "ablation_replan_to_direct_after_failover_failure"
            )
            direct["overhead_sec"] += total_overhead
            direct["replanned"] = True
            direct["ablation_failure_reason"] = (
                "failover_infeasible_replanned_to_direct"
                if direct["success"]
                else "failover_and_direct_repair_infeasible"
            )
            return direct

        return {
            "success": False,
            "trusted_service_restore_sec": np.nan,
            "preferred_restore_sec": np.nan,
            "overhead_sec": total_overhead,
            "repeated_work_sec": 0.0,
            "assignments": immediate["steps"],
            "recovery_strategy": "ablation_failover_plan_failed",
            "replanned": False,
            "ablation_failure_reason":
                "failover_infeasible_but_replanning_disabled",
        }

    T_service = float(immediate["execution_sec"])

    if not attempt_background:
        return {
            "success": True,
            "trusted_service_restore_sec": T_service,
            "preferred_restore_sec": T_service,
            "overhead_sec": total_overhead,
            "repeated_work_sec": 0.0,
            "assignments": immediate["steps"],
            "recovery_strategy": "ablation_failover_only",
            "replanned": False,
            "ablation_failure_reason": "",
        }

    serving_resource = substitute_step["resource"]

    background_problem, background_meta = (
        build_background_preferred_repair_problem(
            service_name=service,
            preferred_model=primary,
            active_substitute=fallback,
            active_substitute_resource=serving_resource,
            resources=RESOURCES,
            resource_available=resource_available,
            feasible_assignments=feasible,
            profiles=load_execution_profiles(),
            gpu_capacity_policy="peak",
            completed_stages=set(),
        )
    )

    background = solve_up(background_problem, background_meta)
    total_overhead += float(background["overhead_sec"])

    if background["success"]:
        return {
            "success": True,
            "trusted_service_restore_sec": T_service,
            "preferred_restore_sec":
                T_service + float(background["execution_sec"]),
            "overhead_sec": total_overhead,
            "repeated_work_sec": 0.0,
            "assignments": immediate["steps"] + background["steps"],
            "recovery_strategy": "ablation_failover_then_background_repair",
            "replanned": False,
            "ablation_failure_reason": "",
        }

    if not allow_replan:
        # Trusted substitute remains active, but preferred-model restoration
        # cannot finish under the frozen recovery strategy.
        return {
            "success": True,
            "trusted_service_restore_sec": T_service,
            "preferred_restore_sec": np.nan,
            "overhead_sec": total_overhead,
            "repeated_work_sec": 0.0,
            "assignments": immediate["steps"],
            "recovery_strategy":
                "ablation_failover_background_infeasible_no_replan",
            "replanned": False,
            "ablation_failure_reason":
                "background_repair_infeasible_but_replanning_disabled",
        }

    direct = run_replica_event_joint(primary, feasible)
    total_overhead += float(direct["overhead_sec"])

    return {
        "success": bool(direct["success"]),
        "trusted_service_restore_sec": T_service,
        "preferred_restore_sec":
            T_service + float(direct["recovery_sec"])
            if direct["success"] else np.nan,
        "overhead_sec": total_overhead,
        "repeated_work_sec": 0.0,
        "assignments":
            immediate["steps"] + (direct["assignments"] if direct["success"] else []),
        "recovery_strategy":
            "ablation_failover_then_replanned_direct_repair",
        "replanned": True,
        "ablation_failure_reason":
            "background_infeasible_replanned_to_direct"
            if direct["success"]
            else "background_and_direct_repair_infeasible",
    }


def run_ablation_variant(row, feasible, variant):
    """
    Component-removal study.

    full_replica:
        Final Experiment-3 REPLICA behavior.

    no_strategy_selection:
        Removes REPLICA's ability to choose failover versus direct repair.
        Every degraded case follows direct preferred-model repair.

    no_state_reconstruction:
        Removes use of the current resource topology for decision making.
        The policy is formed from the fully available assignment world,
        then executed against the actual contracted world.  If a selected
        assignment is unavailable, the frozen plan fails.

    no_replanning:
        Uses the observed state and initial recovery strategy, but cannot
        change path when failover/background repair becomes infeasible.

    no_closed_loop_adaptation:
        Makes one initial decision from the observed state and does not
        perform the subsequent background-repair/replanning feedback step.
        Trusted failover can restore service, but the framework does not
        adapt again to complete preferred-model recovery.
    """
    action = str(row["preferred_recovery_action"])

    if action == "no_recovery_required":
        return {
            "success": True,
            "trusted_service_restore_sec": 0.0,
            "preferred_restore_sec": 0.0,
            "overhead_sec": 0.0,
            "repeated_work_sec": 0.0,
            "assignments": [],
            "recovery_strategy": "no_recovery_required",
            "replanned": False,
            "ablation_failure_reason": "",
        }

    if variant == "full_replica":
        out = run_replica_full(row, feasible)
        out["ablation_failure_reason"] = ""
        return out

    if variant == "no_strategy_selection":
        return _direct_primary(
            row, feasible, "ablation_fixed_direct_primary_repair"
        )

    if variant == "no_replanning":
        if action == "repair_primary":
            return _direct_primary(
                row, feasible, "ablation_direct_primary_no_replanning"
            )
        return _failover_once(
            row, feasible, allow_replan=False, attempt_background=True
        )

    if variant == "no_closed_loop_adaptation":
        if action == "repair_primary":
            return _direct_primary(
                row, feasible, "ablation_one_shot_direct_primary"
            )
        return _failover_once(
            row, feasible, allow_replan=False, attempt_background=False
        )

    if variant == "no_state_reconstruction":
        # Build a "healthy/full" resource world from every empirically
        # profiled model-stage-resource tuple.  The ablated policy therefore
        # does not reconstruct the current contracted topology.
        full_feasible = set(profile_dict.keys())

        if action == "repair_primary":
            planned = _direct_primary(
                row, full_feasible, "ablation_stale_state_direct_primary"
            )
        else:
            planned = _failover_once(
                row,
                full_feasible,
                allow_replan=False,
                attempt_background=True,
            )

        # Validate the frozen assignments against the *actual* world.
        invalid = []
        primary, fallback = _models(row)
        for a in planned.get("assignments", []):
            stage = a.get("stage")
            resource = a.get("resource")
            if not stage or not resource:
                continue

            # Substitute/service actions may not correspond to primary repair
            # stages. Only enforce topology where an empirical task assignment
            # exists in the profile universe.
            candidates = [
                (primary, stage, resource),
                (fallback, stage, resource),
            ]
            profiled = [x for x in candidates if x in full_feasible]
            if profiled and not any(x in feasible for x in profiled):
                invalid.append(f"{stage}@{resource}")

        if invalid:
            planned["success"] = False
            planned["trusted_service_restore_sec"] = np.nan
            planned["preferred_restore_sec"] = np.nan
            planned["ablation_failure_reason"] = (
                "stale_state_selected_unavailable_assignments:"
                + "|".join(invalid)
            )
        else:
            planned["ablation_failure_reason"] = (
                planned.get("ablation_failure_reason", "")
            )
        return planned

    raise ValueError(f"Unknown ablation variant: {variant}")


def execute_ablation_case(row, variant):
    feasible = load_scenario_feasible(row["scenario_path"])
    result = run_ablation_variant(row, feasible, variant)

    success = bool(result["success"])
    if success:
        T_service = float(result["trusted_service_restore_sec"])
        T_pref_raw = result["preferred_restore_sec"]
        T_pref = float(T_pref_raw) if pd.notna(T_pref_raw) else np.nan
    else:
        T_service = np.nan
        T_pref = np.nan

    return {
        "case_id": str(row["case_id"]),
        "variant": variant,
        "service": row["service"],
        "attack_type": row["attack_type"],
        "attack_severity": row["attack_severity"],
        "rho": float(row["rho"]),
        "mask_id": row.get("mask_id", ""),
        "scenario_name": row.get("scenario_name", ""),
        "scenario_path": row["scenario_path"],
        "preferred_recovery_action": row["preferred_recovery_action"],
        "primary_model": row["primary_model"],
        "fallback_model": row["fallback_model"],
        "recovery_success": success,
        "recovery_strategy": result.get("recovery_strategy", ""),
        "replanned": bool(result.get("replanned", False)),
        "trusted_service_restore_sec": T_service,
        "preferred_restore_sec": T_pref,
        "planning_overhead_sec": float(result.get("overhead_sec", 0.0)),
        "repeated_work_sec": float(result.get("repeated_work_sec", 0.0)),
        "n_assignments": len(result.get("assignments", [])),
        "ablation_failure_reason": result.get("ablation_failure_reason", ""),
    }


# ======================================================================
# ABLATION MAIN
# ======================================================================

cases = pd.read_csv(CASES_PATH)

# Experiment 3 contains four scheduler rows per experimental condition.
# The ablation is REPLICA-only, so select the REPLICA member of each
# condition rather than rerunning identical conditions four times.
if "method" in cases.columns:
    cases = cases[cases["method"].astype(str) == "replica"].copy()

print("=" * 120)
print("RQ3 REPLICA FRAMEWORK ABLATION")
print("=" * 120)
print("Unique REPLICA conditions:", len(cases))
print("Variants:", len(ABLATION_VARIANTS))
print("Expected rows:", len(cases) * len(ABLATION_VARIANTS))
print("Output:", RESULTS_PATH)

if RESULTS_PATH.exists():
    done = pd.read_csv(RESULTS_PATH)
    completed = set(
        zip(
            done["case_id"].astype(str),
            done["variant"].astype(str),
        )
    )
else:
    completed = set()

failures = []

for variant in ABLATION_VARIANTS:
    print("\n" + "=" * 120)
    print("VARIANT:", variant)
    print("=" * 120)

    for i, (_, row) in enumerate(cases.iterrows(), start=1):
        key = (str(row["case_id"]), variant)
        if key in completed:
            continue

        print(
            f"[{i}/{len(cases)}] {row['case_id']} "
            f"{row['service']} {row['attack_type']} "
            f"sev={row['attack_severity']} rho={row['rho']} "
            f"{variant}"
        )

        try:
            out = execute_ablation_case(row, variant)
            pd.DataFrame([out]).to_csv(
                RESULTS_PATH,
                mode="a",
                header=not RESULTS_PATH.exists(),
                index=False,
            )
            print(
                f"  success={out['recovery_success']} "
                f"T_service={out['trusted_service_restore_sec']} "
                f"T_pref={out['preferred_restore_sec']} "
                f"reason={out['ablation_failure_reason'] or '-'}"
            )

        except Exception as e:
            failure = {
                "case_id": str(row["case_id"]),
                "variant": variant,
                "service": row["service"],
                "attack_type": row["attack_type"],
                "attack_severity": row["attack_severity"],
                "rho": float(row["rho"]),
                "error": repr(e),
            }
            failures.append(failure)
            with FAILURES_PATH.open("a") as f:
                f.write(json.dumps(failure) + "\n")
            print("  FAILED:", repr(e))


print("\n" + "=" * 120)
print("ABLATION SWEEP COMPLETE")
print("=" * 120)
print("Results:", RESULTS_PATH)
print("Failures:", FAILURES_PATH)

if RESULTS_PATH.exists():
    out = pd.read_csv(RESULTS_PATH)

    print("\n" + "=" * 120)
    print("QUICK COMPONENT SIGNIFICANCE SUMMARY")
    print("=" * 120)

    summary = (
        out.groupby("variant", dropna=False)
        .agg(
            n=("case_id", "size"),
            success_rate_pct=("recovery_success", lambda x: 100.0 * x.mean()),
            trusted_restore_mean_sec=("trusted_service_restore_sec", "mean"),
            preferred_restore_mean_sec=("preferred_restore_sec", "mean"),
            planning_overhead_mean_sec=("planning_overhead_sec", "mean"),
            replans=("replanned", "sum"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))
    summary.to_csv(
        OUTDIR / "rq3_replica_ablation_summary.csv",
        index=False,
    )

    reasons = (
        out[out["ablation_failure_reason"].fillna("") != ""]
        .groupby(["variant", "ablation_failure_reason"])
        .size()
        .reset_index(name="count")
        .sort_values(["variant", "count"], ascending=[True, False])
    )
    reasons.to_csv(
        OUTDIR / "rq3_replica_ablation_failure_reasons.csv",
        index=False,
    )

    print("\nSaved summary:",
          OUTDIR / "rq3_replica_ablation_summary.csv")
    print("Saved failure reasons:",
          OUTDIR / "rq3_replica_ablation_failure_reasons.csv")
