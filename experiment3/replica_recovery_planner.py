#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time

from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
    IntType,
    Object,
    Problem,
    UserType,
    Not,
    OneshotPlanner,
    get_environment,
)

from experiment3.empirical_recovery_profiles import (
    load_execution_profiles,
)


# ============================================================
# TYPES
# ============================================================

ModelType = UserType("Model")
ResourceType = UserType("Resource")


# ============================================================
# RESULT
# ============================================================

@dataclass
class RecoveryPlanStep:
    index: int
    action: str
    model: str
    resource: str | None
    stage: str
    cost_ms: int


@dataclass
class RecoveryPlanResult:
    success: bool
    status: str
    steps: List[RecoveryPlanStep]
    total_cost_sec: float
    planning_overhead_sec: float


# ============================================================
# RECOVERY STAGES
# ============================================================

RESOURCE_STAGES = [
    "update",
    "evaluate",
    "package",
    "deploy",
    "validate",
]

CONTROL_STAGES = [
    "quarantine",
    "reactivate",
]


# ============================================================
# BUILD WORLD
# ============================================================

def build_recovery_problem(
    *,
    model_name: str,
    resources: List[str],
    resource_available: Dict[str, bool],
    feasible_assignments: set[Tuple[str, str, str]],
    profiles,
) -> tuple[Problem, dict]:
    """
    Build a symbolic recovery problem.

    Initial security state:
        model available
        model active
        model degraded
        model untrusted

    Goal:
        trusted
        validated
        active

    The planner must discover the legal recovery path.
    """

    problem = Problem(
        f"recovery_{model_name}"
    )

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    model_obj = Object(
        model_name,
        ModelType,
    )

    problem.add_object(
        model_obj
    )

    resource_objs = {}

    for rid in resources:

        obj = Object(
            rid,
            ResourceType,
        )

        resource_objs[rid] = obj

        problem.add_object(
            obj
        )


    # --------------------------------------------------------
    # Fluents
    # --------------------------------------------------------

    available = Fluent(
        "available",
        BoolType(),
        r=ResourceType,
    )

    model_degraded = Fluent(
        "model_degraded",
        BoolType(),
        m=ModelType,
    )

    model_trusted = Fluent(
        "model_trusted",
        BoolType(),
        m=ModelType,
    )

    model_active = Fluent(
        "model_active",
        BoolType(),
        m=ModelType,
    )

    model_quarantined = Fluent(
        "model_quarantined",
        BoolType(),
        m=ModelType,
    )

    candidate_available = Fluent(
        "candidate_available",
        BoolType(),
        m=ModelType,
    )

    candidate_evaluated = Fluent(
        "candidate_evaluated",
        BoolType(),
        m=ModelType,
    )

    candidate_valid = Fluent(
        "candidate_valid",
        BoolType(),
        m=ModelType,
    )

    candidate_packaged = Fluent(
        "candidate_packaged",
        BoolType(),
        m=ModelType,
    )

    candidate_deployed = Fluent(
        "candidate_deployed",
        BoolType(),
        m=ModelType,
    )

    model_validated = Fluent(
        "model_validated",
        BoolType(),
        m=ModelType,
    )

    total_cost = Fluent(
        "total_cost",
        IntType(0),
    )


    for fluent in [
        available,
        model_degraded,
        model_trusted,
        model_active,
        model_quarantined,
        candidate_available,
        candidate_evaluated,
        candidate_valid,
        candidate_packaged,
        candidate_deployed,
        model_validated,
        total_cost,
    ]:
        problem.add_fluent(
            fluent,
            default_initial_value=False
            if fluent != total_cost
            else 0,
        )


    # --------------------------------------------------------
    # Initial world
    # --------------------------------------------------------

    for rid, obj in resource_objs.items():

        problem.set_initial_value(
            available(obj),
            bool(
                resource_available[
                    rid
                ]
            ),
        )


    problem.set_initial_value(
        model_degraded(
            model_obj
        ),
        True,
    )

    problem.set_initial_value(
        model_trusted(
            model_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_active(
            model_obj
        ),
        True,
    )

    problem.set_initial_value(
        model_quarantined(
            model_obj
        ),
        False,
    )

    problem.set_initial_value(
        candidate_available(
            model_obj
        ),
        False,
    )

    problem.set_initial_value(
        candidate_evaluated(
            model_obj
        ),
        False,
    )

    problem.set_initial_value(
        candidate_valid(
            model_obj
        ),
        False,
    )

    problem.set_initial_value(
        candidate_packaged(
            model_obj
        ),
        False,
    )

    problem.set_initial_value(
        candidate_deployed(
            model_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_validated(
            model_obj
        ),
        False,
    )

    problem.set_initial_value(
        total_cost,
        0,
    )


    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    action_metadata = {}


    def runtime_ms(
        stage: str,
        rid: str,
    ) -> int:

        key = (
            model_name,
            stage,
            rid,
        )

        if key not in profiles:
            raise KeyError(
                f"Missing empirical profile: "
                f"{key}"
            )

        return max(
            1,
            int(
                round(
                    profiles[key][
                        "runtime_sec"
                    ]
                    * 1000.0
                )
            ),
        )


    # --------------------------------------------------------
    # QUARANTINE
    #
    # Control-plane state transition.
    # Zero empirical workload runtime.
    # --------------------------------------------------------

    quarantine = InstantaneousAction(
        f"quarantine__{model_name}"
    )

    quarantine.add_precondition(
        model_degraded(
            model_obj
        )
    )

    quarantine.add_precondition(
        Not(
            model_trusted(
                model_obj
            )
        )
    )

    quarantine.add_precondition(
        Not(
            model_quarantined(
                model_obj
            )
        )
    )

    quarantine.add_effect(
        model_quarantined(
            model_obj
        ),
        True,
    )

    quarantine.add_effect(
        model_active(
            model_obj
        ),
        False,
    )

    problem.add_action(
        quarantine
    )

    action_metadata[
        quarantine.name
    ] = {
        "stage":
            "quarantine",

        "resource":
            None,

        "cost_ms":
            0,
    }


    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            model_name,
            "update",
            rid,
        )

        if key not in feasible_assignments:
            continue

        if key not in profiles:
            continue

        cost = runtime_ms(
            "update",
            rid,
        )

        action = InstantaneousAction(
            f"update__{model_name}__to__{rid}"
        )

        action.add_precondition(
            model_quarantined(
                model_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )

        action.add_effect(
            candidate_available(
                model_obj
            ),
            True,
        )

        action.add_increase_effect(
            total_cost,
            cost,
        )

        problem.add_action(
            action
        )

        action_metadata[
            action.name
        ] = {
            "stage":
                "update",

            "resource":
                rid,

            "cost_ms":
                cost,
        }


    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            model_name,
            "evaluate",
            rid,
        )

        if key not in feasible_assignments:
            continue

        if key not in profiles:
            continue

        cost = runtime_ms(
            "evaluate",
            rid,
        )

        action = InstantaneousAction(
            f"evaluate__{model_name}__to__{rid}"
        )

        action.add_precondition(
            candidate_available(
                model_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )

        action.add_effect(
            candidate_evaluated(
                model_obj
            ),
            True,
        )

        action.add_effect(
            candidate_valid(
                model_obj
            ),
            True,
        )

        action.add_increase_effect(
            total_cost,
            cost,
        )

        problem.add_action(
            action
        )

        action_metadata[
            action.name
        ] = {
            "stage":
                "evaluate",

            "resource":
                rid,

            "cost_ms":
                cost,
        }


    # --------------------------------------------------------
    # PACKAGE
    # --------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            model_name,
            "package",
            rid,
        )

        if key not in feasible_assignments:
            continue

        if key not in profiles:
            continue

        cost = runtime_ms(
            "package",
            rid,
        )

        action = InstantaneousAction(
            f"package__{model_name}__to__{rid}"
        )

        action.add_precondition(
            candidate_evaluated(
                model_obj
            )
        )

        action.add_precondition(
            candidate_valid(
                model_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )

        action.add_effect(
            candidate_packaged(
                model_obj
            ),
            True,
        )

        action.add_increase_effect(
            total_cost,
            cost,
        )

        problem.add_action(
            action
        )

        action_metadata[
            action.name
        ] = {
            "stage":
                "package",

            "resource":
                rid,

            "cost_ms":
                cost,
        }


    # --------------------------------------------------------
    # DEPLOY
    # --------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            model_name,
            "deploy",
            rid,
        )

        if key not in feasible_assignments:
            continue

        if key not in profiles:
            continue

        cost = runtime_ms(
            "deploy",
            rid,
        )

        action = InstantaneousAction(
            f"deploy__{model_name}__to__{rid}"
        )

        action.add_precondition(
            candidate_packaged(
                model_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )

        action.add_effect(
            candidate_deployed(
                model_obj
            ),
            True,
        )

        action.add_effect(
            model_validated(
                model_obj
            ),
            False,
        )

        action.add_increase_effect(
            total_cost,
            cost,
        )

        problem.add_action(
            action
        )

        action_metadata[
            action.name
        ] = {
            "stage":
                "deploy",

            "resource":
                rid,

            "cost_ms":
                cost,
        }


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            model_name,
            "validate",
            rid,
        )

        if key not in feasible_assignments:
            continue

        if key not in profiles:
            continue

        cost = runtime_ms(
            "validate",
            rid,
        )

        action = InstantaneousAction(
            f"validate__{model_name}__to__{rid}"
        )

        action.add_precondition(
            candidate_deployed(
                model_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )

        action.add_effect(
            model_validated(
                model_obj
            ),
            True,
        )

        action.add_effect(
            model_trusted(
                model_obj
            ),
            True,
        )

        action.add_effect(
            model_degraded(
                model_obj
            ),
            False,
        )

        action.add_increase_effect(
            total_cost,
            cost,
        )

        problem.add_action(
            action
        )

        action_metadata[
            action.name
        ] = {
            "stage":
                "validate",

            "resource":
                rid,

            "cost_ms":
                cost,
        }


    # --------------------------------------------------------
    # REACTIVATE
    # --------------------------------------------------------

    reactivate = InstantaneousAction(
        f"reactivate__{model_name}"
    )

    reactivate.add_precondition(
        model_validated(
            model_obj
        )
    )

    reactivate.add_precondition(
        model_trusted(
            model_obj
        )
    )

    reactivate.add_precondition(
        Not(
            model_active(
                model_obj
            )
        )
    )

    reactivate.add_effect(
        model_active(
            model_obj
        ),
        True,
    )

    reactivate.add_effect(
        model_quarantined(
            model_obj
        ),
        False,
    )

    problem.add_action(
        reactivate
    )

    action_metadata[
        reactivate.name
    ] = {
        "stage":
            "reactivate",

        "resource":
            None,

        "cost_ms":
            0,
    }


    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    problem.add_goal(
        model_trusted(
            model_obj
        )
    )

    problem.add_goal(
        model_validated(
            model_obj
        )
    )

    problem.add_goal(
        model_active(
            model_obj
        )
    )

    problem.add_goal(
        Not(
            model_degraded(
                model_obj
            )
        )
    )


    # Minimize empirical recovery execution cost.
    problem.add_quality_metric(
        __import__(
            "unified_planning.shortcuts",
            fromlist=[
                "MinimizeExpressionOnFinalState"
            ]
        ).MinimizeExpressionOnFinalState(
            total_cost
        )
    )


    metadata = {
        "model_obj":
            model_obj,

        "resources":
            resource_objs,

        "action_metadata":
            action_metadata,

        "total_cost":
            total_cost,
    }

    return (
        problem,
        metadata,
    )


# ============================================================
# SOLVE
# ============================================================

def solve_recovery_problem(
    *,
    problem: Problem,
    metadata,
    engine_name: str = "enhsp-opt",
) -> RecoveryPlanResult:

    t0 = time.perf_counter()

    with OneshotPlanner(
        name=engine_name
    ) as planner:

        result = planner.solve(
            problem
        )

    overhead = (
        time.perf_counter()
        - t0
    )


    if (
        result.plan is None
    ):

        return RecoveryPlanResult(
            success=False,
            status=str(
                result.status
            ),
            steps=[],
            total_cost_sec=0.0,
            planning_overhead_sec=
                overhead,
        )


    steps = []

    total_cost_ms = 0


    for i, action_instance in enumerate(
        result.plan.actions
    ):

        action_name = (
            action_instance
            .action
            .name
        )

        info = (
            metadata[
                "action_metadata"
            ].get(
                action_name,
                {
                    "stage":
                        action_name,

                    "resource":
                        None,

                    "cost_ms":
                        0,
                },
            )
        )

        total_cost_ms += int(
            info["cost_ms"]
        )

        steps.append(
            RecoveryPlanStep(
                index=i + 1,

                action=
                    action_name,

                model=
                    problem.name.replace(
                        "recovery_",
                        "",
                        1,
                    ),

                resource=
                    info[
                        "resource"
                    ],

                stage=
                    info[
                        "stage"
                    ],

                cost_ms=int(
                    info[
                        "cost_ms"
                    ]
                ),
            )
        )


    return RecoveryPlanResult(
        success=True,

        status=str(
            result.status
        ),

        steps=steps,

        total_cost_sec=(
            total_cost_ms
            / 1000.0
        ),

        planning_overhead_sec=
            overhead,
    )
