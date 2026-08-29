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
    And,
    Or,
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


# =============================================================================
# SERVICE-LEVEL RECOVERY
# =============================================================================

def build_service_recovery_problem(
    *,
    service_name: str,
    attacked_model: str,
    substitute_model: str,
    substitute_quality: float,
    minimum_quality: float,
    resources: List[str],
    resource_available: Dict[str, bool],
    feasible_assignments: set[Tuple[str, str, str]],
    profiles,
    substitute_available: bool = True,
):
    """
    Build a service-level trusted recovery problem.

    The goal is NOT necessarily to restore the attacked model.

    The planner may satisfy the service goal through either:

      1. repair attacked_model:
           quarantine
           -> update
           -> evaluate
           -> package
           -> deploy
           -> validate
           -> reactivate

      2. substitute:
           quarantine attacked_model
           -> activate already-trusted substitute_model

    The substitute route only exists when:
      - the substitute is available,
      - its measured quality meets the mission threshold,
      - a feasible empirical activation mapping exists.

    This lets the planner choose a recovery STRATEGY rather than
    merely a resource assignment.
    """

    problem = Problem(
        f"service_recovery_{service_name}"
    )


    # -------------------------------------------------------------------------
    # Objects
    # -------------------------------------------------------------------------

    attacked_obj = Object(
        attacked_model,
        ModelType,
    )

    substitute_obj = Object(
        substitute_model,
        ModelType,
    )

    problem.add_object(
        attacked_obj
    )

    problem.add_object(
        substitute_obj
    )


    resource_objs = {}

    for rid in resources:

        robj = Object(
            rid,
            ResourceType,
        )

        resource_objs[rid] = robj

        problem.add_object(
            robj
        )


    # -------------------------------------------------------------------------
    # Fluents
    # -------------------------------------------------------------------------

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


    bool_fluents = [
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
    ]


    for fluent in bool_fluents:

        problem.add_fluent(
            fluent,
            default_initial_value=False,
        )


    problem.add_fluent(
        total_cost,
        default_initial_value=0,
    )


    # -------------------------------------------------------------------------
    # Resource state
    # -------------------------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        problem.set_initial_value(
            available(
                robj
            ),
            bool(
                resource_available.get(
                    rid,
                    False,
                )
            ),
        )


    # -------------------------------------------------------------------------
    # Attacked provider state
    #
    # Physically present and active, but degraded + untrusted.
    # -------------------------------------------------------------------------

    problem.set_initial_value(
        model_degraded(
            attacked_obj
        ),
        True,
    )

    problem.set_initial_value(
        model_trusted(
            attacked_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_validated(
            attacked_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_active(
            attacked_obj
        ),
        True,
    )

    problem.set_initial_value(
        model_quarantined(
            attacked_obj
        ),
        False,
    )


    # -------------------------------------------------------------------------
    # Substitute state
    #
    # The standby is assumed to be a previously validated, trusted model.
    # It begins inactive.
    # -------------------------------------------------------------------------

    substitute_quality_ok = (
        float(
            substitute_quality
        )
        >=
        float(
            minimum_quality
        )
    )


    substitute_usable = (
        bool(
            substitute_available
        )
        and substitute_quality_ok
    )


    problem.set_initial_value(
        model_degraded(
            substitute_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_trusted(
            substitute_obj
        ),
        substitute_usable,
    )

    problem.set_initial_value(
        model_validated(
            substitute_obj
        ),
        substitute_usable,
    )

    problem.set_initial_value(
        model_active(
            substitute_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_quarantined(
            substitute_obj
        ),
        False,
    )


    problem.set_initial_value(
        total_cost,
        0,
    )


    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    action_metadata = {}


    def runtime_ms(
        model_name: str,
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
                    float(
                        profiles[
                            key
                        ][
                            "runtime_sec"
                        ]
                    )
                    * 1000.0
                )
            ),
        )


    # =========================================================================
    # STRATEGY A — REPAIR ATTACKED MODEL
    # =========================================================================

    quarantine = InstantaneousAction(
        f"quarantine__{attacked_model}"
    )

    quarantine.add_precondition(
        model_degraded(
            attacked_obj
        )
    )

    quarantine.add_precondition(
        Not(
            model_trusted(
                attacked_obj
            )
        )
    )

    quarantine.add_precondition(
        Not(
            model_quarantined(
                attacked_obj
            )
        )
    )

    quarantine.add_effect(
        model_quarantined(
            attacked_obj
        ),
        True,
    )

    quarantine.add_effect(
        model_active(
            attacked_obj
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

        "model":
            attacked_model,

        "strategy":
            "shared",

        "cost_ms":
            0,
    }


    # -------------------------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            attacked_model,
            "update",
            rid,
        )

        if (
            key
            not in feasible_assignments
        ):
            continue

        if key not in profiles:
            continue


        cost = runtime_ms(
            attacked_model,
            "update",
            rid,
        )


        action = InstantaneousAction(
            f"update__{attacked_model}__to__{rid}"
        )


        action.add_precondition(
            model_quarantined(
                attacked_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            candidate_available(
                attacked_obj
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

            "model":
                attacked_model,

            "strategy":
                "repair",

            "cost_ms":
                cost,
        }


    # -------------------------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            attacked_model,
            "evaluate",
            rid,
        )

        if (
            key
            not in feasible_assignments
        ):
            continue

        if key not in profiles:
            continue


        cost = runtime_ms(
            attacked_model,
            "evaluate",
            rid,
        )


        action = InstantaneousAction(
            f"evaluate__{attacked_model}__to__{rid}"
        )


        action.add_precondition(
            candidate_available(
                attacked_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            candidate_evaluated(
                attacked_obj
            ),
            True,
        )

        action.add_effect(
            candidate_valid(
                attacked_obj
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

            "model":
                attacked_model,

            "strategy":
                "repair",

            "cost_ms":
                cost,
        }


    # -------------------------------------------------------------------------
    # PACKAGE
    # -------------------------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            attacked_model,
            "package",
            rid,
        )

        if (
            key
            not in feasible_assignments
        ):
            continue

        if key not in profiles:
            continue


        cost = runtime_ms(
            attacked_model,
            "package",
            rid,
        )


        action = InstantaneousAction(
            f"package__{attacked_model}__to__{rid}"
        )


        action.add_precondition(
            candidate_evaluated(
                attacked_obj
            )
        )

        action.add_precondition(
            candidate_valid(
                attacked_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            candidate_packaged(
                attacked_obj
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

            "model":
                attacked_model,

            "strategy":
                "repair",

            "cost_ms":
                cost,
        }


    # -------------------------------------------------------------------------
    # DEPLOY
    # -------------------------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            attacked_model,
            "deploy",
            rid,
        )

        if (
            key
            not in feasible_assignments
        ):
            continue

        if key not in profiles:
            continue


        cost = runtime_ms(
            attacked_model,
            "deploy",
            rid,
        )


        action = InstantaneousAction(
            f"deploy__{attacked_model}__to__{rid}"
        )


        action.add_precondition(
            candidate_packaged(
                attacked_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            candidate_deployed(
                attacked_obj
            ),
            True,
        )

        action.add_effect(
            model_validated(
                attacked_obj
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

            "model":
                attacked_model,

            "strategy":
                "repair",

            "cost_ms":
                cost,
        }


    # -------------------------------------------------------------------------
    # VALIDATE
    # -------------------------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            attacked_model,
            "validate",
            rid,
        )

        if (
            key
            not in feasible_assignments
        ):
            continue

        if key not in profiles:
            continue


        cost = runtime_ms(
            attacked_model,
            "validate",
            rid,
        )


        action = InstantaneousAction(
            f"validate__{attacked_model}__to__{rid}"
        )


        action.add_precondition(
            candidate_deployed(
                attacked_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            model_validated(
                attacked_obj
            ),
            True,
        )

        action.add_effect(
            model_trusted(
                attacked_obj
            ),
            True,
        )

        action.add_effect(
            model_degraded(
                attacked_obj
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

            "model":
                attacked_model,

            "strategy":
                "repair",

            "cost_ms":
                cost,
        }


    # -------------------------------------------------------------------------
    # REACTIVATE REPAIRED MODEL
    # -------------------------------------------------------------------------

    reactivate = InstantaneousAction(
        f"reactivate__{attacked_model}"
    )


    reactivate.add_precondition(
        model_validated(
            attacked_obj
        )
    )

    reactivate.add_precondition(
        model_trusted(
            attacked_obj
        )
    )

    reactivate.add_precondition(
        Not(
            model_active(
                attacked_obj
            )
        )
    )


    reactivate.add_effect(
        model_active(
            attacked_obj
        ),
        True,
    )

    reactivate.add_effect(
        model_quarantined(
            attacked_obj
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

        "model":
            attacked_model,

        "strategy":
            "repair",

        "cost_ms":
            0,
    }


    # =========================================================================
    # STRATEGY B — ACTIVATE TRUSTED SUBSTITUTE
    #
    # The substitute is already validated. We therefore use its measured
    # reactivate cost as the empirical service activation cost.
    # =========================================================================

    substitute_action_count = 0


    if substitute_usable:

        for rid, robj in (
            resource_objs.items()
        ):

            key = (
                substitute_model,
                "reactivate",
                rid,
            )


            if (
                key
                not in feasible_assignments
            ):
                continue

            if key not in profiles:
                continue


            cost = runtime_ms(
                substitute_model,
                "reactivate",
                rid,
            )


            action = InstantaneousAction(
                f"activate_substitute__"
                f"{substitute_model}__to__{rid}"
            )


            # The attacked provider must first be isolated.
            action.add_precondition(
                model_quarantined(
                    attacked_obj
                )
            )

            action.add_precondition(
                model_trusted(
                    substitute_obj
                )
            )

            action.add_precondition(
                model_validated(
                    substitute_obj
                )
            )

            action.add_precondition(
                Not(
                    model_degraded(
                        substitute_obj
                    )
                )
            )

            action.add_precondition(
                Not(
                    model_active(
                        substitute_obj
                    )
                )
            )

            action.add_precondition(
                available(
                    robj
                )
            )


            action.add_effect(
                model_active(
                    substitute_obj
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
                    "activate_substitute",

                "resource":
                    rid,

                "model":
                    substitute_model,

                "strategy":
                    "substitute",

                "cost_ms":
                    cost,
            }


            substitute_action_count += 1


    # =========================================================================
    # SERVICE-LEVEL GOAL
    #
    # Detection service is restored when EITHER provider is:
    #
    #       trusted AND validated AND active
    #
    # This is intentionally provider-independent.
    # =========================================================================

    attacked_service_ok = And(
        model_trusted(
            attacked_obj
        ),
        model_validated(
            attacked_obj
        ),
        model_active(
            attacked_obj
        ),
    )


    substitute_service_ok = And(
        model_trusted(
            substitute_obj
        ),
        model_validated(
            substitute_obj
        ),
        model_active(
            substitute_obj
        ),
    )


    problem.add_goal(
        Or(
            attacked_service_ok,
            substitute_service_ok,
        )
    )


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
        "service_name":
            service_name,

        "attacked_model":
            attacked_model,

        "substitute_model":
            substitute_model,

        "substitute_quality":
            float(
                substitute_quality
            ),

        "minimum_quality":
            float(
                minimum_quality
            ),

        "substitute_quality_ok":
            bool(
                substitute_quality_ok
            ),

        "substitute_available":
            bool(
                substitute_available
            ),

        "substitute_usable":
            bool(
                substitute_usable
            ),

        "substitute_action_count":
            int(
                substitute_action_count
            ),

        "resource_objects":
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


# =============================================================================
# BACKGROUND REPAIR AFTER TRUSTED FAILOVER
# =============================================================================

def build_background_preferred_repair_problem(
    *,
    service_name: str,
    preferred_model: str,
    active_substitute: str,
    active_substitute_resource: str,
    resources: List[str],
    resource_available: Dict[str, bool],
    feasible_assignments: set[Tuple[str, str, str]],
    profiles,
    max_gpu_fraction: float = 1.0,
    gpu_capacity_policy: str = "mean",
    completed_stages: set[str] | None = None,
):
    """
    Repair the preferred model while a trusted substitute
    continues providing the service.

    Initial world
    -------------
    preferred:
        degraded
        untrusted
        quarantined
        inactive

    substitute:
        trusted
        validated
        active

    Goal
    ----
    preferred becomes:
        trusted
        validated
        active

    and the substitute is released after the preferred model
    is safely restored.

    The service therefore remains available during the repair
    process.
    """

    problem = Problem(
        f"background_repair_{service_name}_{preferred_model}"
    )


    # -------------------------------------------------------------------------
    # Objects
    # -------------------------------------------------------------------------

    preferred_obj = Object(
        preferred_model,
        ModelType,
    )

    substitute_obj = Object(
        active_substitute,
        ModelType,
    )

    problem.add_object(
        preferred_obj
    )

    problem.add_object(
        substitute_obj
    )


    resource_objs = {}

    for rid in resources:

        robj = Object(
            rid,
            ResourceType,
        )

        resource_objs[rid] = robj

        problem.add_object(
            robj
        )


    # -------------------------------------------------------------------------
    # Fluents
    # -------------------------------------------------------------------------

    available = Fluent(
        "bg_available",
        BoolType(),
        r=ResourceType,
    )

    model_degraded = Fluent(
        "bg_model_degraded",
        BoolType(),
        m=ModelType,
    )

    model_trusted = Fluent(
        "bg_model_trusted",
        BoolType(),
        m=ModelType,
    )

    model_active = Fluent(
        "bg_model_active",
        BoolType(),
        m=ModelType,
    )

    model_quarantined = Fluent(
        "bg_model_quarantined",
        BoolType(),
        m=ModelType,
    )

    candidate_available = Fluent(
        "bg_candidate_available",
        BoolType(),
        m=ModelType,
    )

    candidate_evaluated = Fluent(
        "bg_candidate_evaluated",
        BoolType(),
        m=ModelType,
    )

    candidate_valid = Fluent(
        "bg_candidate_valid",
        BoolType(),
        m=ModelType,
    )

    candidate_packaged = Fluent(
        "bg_candidate_packaged",
        BoolType(),
        m=ModelType,
    )

    candidate_deployed = Fluent(
        "bg_candidate_deployed",
        BoolType(),
        m=ModelType,
    )

    model_validated = Fluent(
        "bg_model_validated",
        BoolType(),
        m=ModelType,
    )

    total_cost = Fluent(
        "bg_total_cost",
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
    ]:

        problem.add_fluent(
            fluent,
            default_initial_value=False,
        )


    problem.add_fluent(
        total_cost,
        default_initial_value=0,
    )


    # -------------------------------------------------------------------------
    # Initial resource world
    # -------------------------------------------------------------------------

    for rid, robj in (
        resource_objs.items()
    ):

        problem.set_initial_value(
            available(
                robj
            ),
            bool(
                resource_available.get(
                    rid,
                    False,
                )
            ),
        )


    # -------------------------------------------------------------------------
    # Preferred model after failover
    # -------------------------------------------------------------------------

    problem.set_initial_value(
        model_degraded(
            preferred_obj
        ),
        True,
    )

    problem.set_initial_value(
        model_trusted(
            preferred_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_validated(
            preferred_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_active(
            preferred_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_quarantined(
            preferred_obj
        ),
        True,
    )


    # -------------------------------------------------------------------------
    # Trusted substitute currently maintaining the service
    # -------------------------------------------------------------------------

    problem.set_initial_value(
        model_degraded(
            substitute_obj
        ),
        False,
    )

    problem.set_initial_value(
        model_trusted(
            substitute_obj
        ),
        True,
    )

    problem.set_initial_value(
        model_validated(
            substitute_obj
        ),
        True,
    )

    problem.set_initial_value(
        model_active(
            substitute_obj
        ),
        True,
    )

    problem.set_initial_value(
        model_quarantined(
            substitute_obj
        ),
        False,
    )


    problem.set_initial_value(
        total_cost,
        0,
    )


    # -------------------------------------------------------------------------
    # Preserve valid work completed before a state change.
    #
    # Example:
    #   update completed under the old world
    #   -> candidate_available remains True after replanning
    #   -> the update does not need to be repeated
    # -------------------------------------------------------------------------

    completed_stages = set(
        completed_stages or ()
    )

    valid_stage_names = {
        "update",
        "evaluate",
        "package",
        "deploy",
        "validate",
    }

    unknown = (
        completed_stages
        - valid_stage_names
    )

    if unknown:
        raise ValueError(
            f"Unknown completed stages: {sorted(unknown)}"
        )


    if "update" in completed_stages:

        problem.set_initial_value(
            candidate_available(
                preferred_obj
            ),
            True,
        )


    if "evaluate" in completed_stages:

        problem.set_initial_value(
            candidate_available(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_evaluated(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_valid(
                preferred_obj
            ),
            True,
        )


    if "package" in completed_stages:

        problem.set_initial_value(
            candidate_available(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_evaluated(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_valid(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_packaged(
                preferred_obj
            ),
            True,
        )


    if "deploy" in completed_stages:

        problem.set_initial_value(
            candidate_available(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_evaluated(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_valid(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_packaged(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_deployed(
                preferred_obj
            ),
            True,
        )


    if "validate" in completed_stages:

        problem.set_initial_value(
            candidate_available(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_evaluated(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_valid(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_packaged(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            candidate_deployed(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            model_validated(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            model_trusted(
                preferred_obj
            ),
            True,
        )

        problem.set_initial_value(
            model_degraded(
                preferred_obj
            ),
            False,
        )


    action_metadata = {}


    def runtime_ms(
        model_name: str,
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
                f"Missing empirical profile: {key}"
            )

        return max(
            1,
            int(
                round(
                    float(
                        profiles[
                            key
                        ]["runtime_sec"]
                    )
                    * 1000.0
                )
            ),
        )


    # -------------------------------------------------------------------------
    # Active-service reservation
    #
    # The substitute is actively serving while the preferred model
    # is repaired. If a repair stage uses that SAME GPU, it must fit
    # within the remaining measured GPU/VRAM capacity.
    # -------------------------------------------------------------------------

    substitute_key = (
        active_substitute,
        "reactivate",
        active_substitute_resource,
    )

    if substitute_key not in profiles:
        raise KeyError(
            f"Missing active substitute profile: {substitute_key}"
        )

    substitute_profile = profiles[
        substitute_key
    ]

    policy = str(
        gpu_capacity_policy
    ).strip().lower()

    if policy not in {
        "mean",
        "peak",
    }:
        raise ValueError(
            "gpu_capacity_policy must be "
            "'mean' or 'peak'"
        )

    gpu_field = (
        "gpu_mean_pct"
        if policy == "mean"
        else "gpu_peak_pct"
    )

    serving_gpu_fraction = min(
        1.0,
        max(
            0.0,
            float(
                substitute_profile.get(
                    gpu_field,
                    0.0,
                )
            ) / 100.0,
        ),
    )

    serving_vram_mb = max(
        0.0,
        float(
            substitute_profile.get(
                "vram_peak_mb",
                0.0,
            )
        ),
    )


    resource_vram_capacity_mb = {
        "A100": 80_000.0,
        "RTX4090": 24_000.0,
        "RTX5090": 32_000.0,
    }


    def background_mapping_safe(
        stage: str,
        rid: str,
    ) -> tuple[bool, str]:
        """
        Check whether one background-repair placement can coexist
        with the active substitute service.

        Different GPU:
            no direct serving contention is charged here.

        Same GPU:
            serving GPU + repair GPU must remain within capacity,
            and serving VRAM + repair VRAM must fit device VRAM.
        """

        key = (
            preferred_model,
            stage,
            rid,
        )

        if key not in profiles:
            return False, "missing_profile"

        if rid != active_substitute_resource:
            return True, "different_resource"

        repair_profile = profiles[
            key
        ]

        repair_gpu_fraction = min(
            1.0,
            max(
                0.0,
                float(
                    repair_profile.get(
                        gpu_field,
                        0.0,
                    )
                ) / 100.0,
            ),
        )

        combined_gpu = (
            serving_gpu_fraction
            + repair_gpu_fraction
        )

        if combined_gpu > (
            float(max_gpu_fraction)
            + 1e-12
        ):
            return (
                False,
                f"gpu_contention:"
                f"{combined_gpu:.3f}>"
                f"{float(max_gpu_fraction):.3f}",
            )

        repair_vram_mb = max(
            0.0,
            float(
                repair_profile.get(
                    "vram_peak_mb",
                    0.0,
                )
            ),
        )

        capacity = (
            resource_vram_capacity_mb.get(
                rid
            )
        )

        if (
            capacity is not None
            and (
                serving_vram_mb
                + repair_vram_mb
            ) > capacity
        ):
            return (
                False,
                f"vram_contention:"
                f"{serving_vram_mb + repair_vram_mb:.1f}>"
                f"{capacity:.1f}",
            )

        return True, "same_resource_safe"


    # -------------------------------------------------------------------------
    # Helper for metadata
    # -------------------------------------------------------------------------

    def remember(
        action,
        *,
        stage,
        resource,
        cost_ms,
    ):

        action_metadata[
            action.name
        ] = {
            "stage":
                stage,

            "resource":
                resource,

            "model":
                preferred_model,

            "strategy":
                "background_repair",

            "cost_ms":
                int(
                    cost_ms
                ),
        }


    # =========================================================================
    # UPDATE
    # =========================================================================

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            preferred_model,
            "update",
            rid,
        )

        if (
            key not in feasible_assignments
            or key not in profiles
        ):
            continue

        safe, reason = (
            background_mapping_safe(
                "update",
                rid,
            )
        )

        if not safe:
            print(
                f"[bg-capacity] reject "
                f"update->{rid}: {reason}"
            )
            continue


        cost = runtime_ms(
            preferred_model,
            "update",
            rid,
        )


        action = InstantaneousAction(
            f"bg_update__{preferred_model}__to__{rid}"
        )


        action.add_precondition(
            model_quarantined(
                preferred_obj
            )
        )

        action.add_precondition(
            Not(
                candidate_available(
                    preferred_obj
                )
            )
        )

        # Trusted service must still be running while repair occurs.
        action.add_precondition(
            model_trusted(
                substitute_obj
            )
        )

        action.add_precondition(
            model_validated(
                substitute_obj
            )
        )

        action.add_precondition(
            model_active(
                substitute_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            candidate_available(
                preferred_obj
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

        remember(
            action,
            stage="update",
            resource=rid,
            cost_ms=cost,
        )


    # =========================================================================
    # EVALUATE
    # =========================================================================

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            preferred_model,
            "evaluate",
            rid,
        )

        if (
            key not in feasible_assignments
            or key not in profiles
        ):
            continue

        safe, reason = (
            background_mapping_safe(
                "evaluate",
                rid,
            )
        )

        if not safe:
            print(
                f"[bg-capacity] reject "
                f"evaluate->{rid}: {reason}"
            )
            continue


        cost = runtime_ms(
            preferred_model,
            "evaluate",
            rid,
        )


        action = InstantaneousAction(
            f"bg_evaluate__{preferred_model}__to__{rid}"
        )


        action.add_precondition(
            candidate_available(
                preferred_obj
            )
        )

        action.add_precondition(
            Not(
                candidate_evaluated(
                    preferred_obj
                )
            )
        )

        action.add_precondition(
            model_active(
                substitute_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            candidate_evaluated(
                preferred_obj
            ),
            True,
        )

        action.add_effect(
            candidate_valid(
                preferred_obj
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

        remember(
            action,
            stage="evaluate",
            resource=rid,
            cost_ms=cost,
        )


    # =========================================================================
    # PACKAGE
    # =========================================================================

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            preferred_model,
            "package",
            rid,
        )

        if (
            key not in feasible_assignments
            or key not in profiles
        ):
            continue

        safe, reason = (
            background_mapping_safe(
                "package",
                rid,
            )
        )

        if not safe:
            print(
                f"[bg-capacity] reject "
                f"package->{rid}: {reason}"
            )
            continue


        cost = runtime_ms(
            preferred_model,
            "package",
            rid,
        )


        action = InstantaneousAction(
            f"bg_package__{preferred_model}__to__{rid}"
        )


        action.add_precondition(
            candidate_evaluated(
                preferred_obj
            )
        )

        action.add_precondition(
            candidate_valid(
                preferred_obj
            )
        )

        action.add_precondition(
            Not(
                candidate_packaged(
                    preferred_obj
                )
            )
        )

        action.add_precondition(
            model_active(
                substitute_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            candidate_packaged(
                preferred_obj
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

        remember(
            action,
            stage="package",
            resource=rid,
            cost_ms=cost,
        )


    # =========================================================================
    # DEPLOY
    # =========================================================================

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            preferred_model,
            "deploy",
            rid,
        )

        if (
            key not in feasible_assignments
            or key not in profiles
        ):
            continue

        safe, reason = (
            background_mapping_safe(
                "deploy",
                rid,
            )
        )

        if not safe:
            print(
                f"[bg-capacity] reject "
                f"deploy->{rid}: {reason}"
            )
            continue


        cost = runtime_ms(
            preferred_model,
            "deploy",
            rid,
        )


        action = InstantaneousAction(
            f"bg_deploy__{preferred_model}__to__{rid}"
        )


        action.add_precondition(
            candidate_packaged(
                preferred_obj
            )
        )

        action.add_precondition(
            Not(
                candidate_deployed(
                    preferred_obj
                )
            )
        )

        action.add_precondition(
            model_active(
                substitute_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            candidate_deployed(
                preferred_obj
            ),
            True,
        )

        action.add_effect(
            model_validated(
                preferred_obj
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

        remember(
            action,
            stage="deploy",
            resource=rid,
            cost_ms=cost,
        )


    # =========================================================================
    # VALIDATE
    # =========================================================================

    for rid, robj in (
        resource_objs.items()
    ):

        key = (
            preferred_model,
            "validate",
            rid,
        )

        if (
            key not in feasible_assignments
            or key not in profiles
        ):
            continue

        safe, reason = (
            background_mapping_safe(
                "validate",
                rid,
            )
        )

        if not safe:
            print(
                f"[bg-capacity] reject "
                f"validate->{rid}: {reason}"
            )
            continue


        cost = runtime_ms(
            preferred_model,
            "validate",
            rid,
        )


        action = InstantaneousAction(
            f"bg_validate__{preferred_model}__to__{rid}"
        )


        action.add_precondition(
            candidate_deployed(
                preferred_obj
            )
        )

        action.add_precondition(
            Not(
                model_validated(
                    preferred_obj
                )
            )
        )

        action.add_precondition(
            model_active(
                substitute_obj
            )
        )

        action.add_precondition(
            available(
                robj
            )
        )


        action.add_effect(
            model_validated(
                preferred_obj
            ),
            True,
        )

        action.add_effect(
            model_trusted(
                preferred_obj
            ),
            True,
        )

        action.add_effect(
            model_degraded(
                preferred_obj
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

        remember(
            action,
            stage="validate",
            resource=rid,
            cost_ms=cost,
        )


    # =========================================================================
    # SWITCH BACK TO PREFERRED PROVIDER
    #
    # This is a zero-time control-plane transition.
    #
    # Critically, it is illegal until the repaired preferred model
    # has been validated and trusted.
    # =========================================================================

    switch_back = InstantaneousAction(
        f"switch_back__{preferred_model}"
    )


    switch_back.add_precondition(
        model_trusted(
            preferred_obj
        )
    )

    switch_back.add_precondition(
        model_validated(
            preferred_obj
        )
    )

    switch_back.add_precondition(
        Not(
            model_degraded(
                preferred_obj
            )
        )
    )

    switch_back.add_precondition(
        model_active(
            substitute_obj
        )
    )


    switch_back.add_effect(
        model_active(
            preferred_obj
        ),
        True,
    )

    switch_back.add_effect(
        model_quarantined(
            preferred_obj
        ),
        False,
    )

    switch_back.add_effect(
        model_active(
            substitute_obj
        ),
        False,
    )


    problem.add_action(
        switch_back
    )


    action_metadata[
        switch_back.name
    ] = {
        "stage":
            "switch_back",

        "resource":
            None,

        "model":
            preferred_model,

        "strategy":
            "background_repair",

        "cost_ms":
            0,
    }


    # =========================================================================
    # GOAL
    # =========================================================================

    problem.add_goal(
        model_trusted(
            preferred_obj
        )
    )

    problem.add_goal(
        model_validated(
            preferred_obj
        )
    )

    problem.add_goal(
        model_active(
            preferred_obj
        )
    )

    problem.add_goal(
        Not(
            model_degraded(
                preferred_obj
            )
        )
    )

    problem.add_goal(
        Not(
            model_active(
                substitute_obj
            )
        )
    )


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
        "service_name":
            service_name,

        "gpu_capacity_policy":
            policy,

        "serving_gpu_fraction":
            serving_gpu_fraction,

        "serving_vram_mb":
            serving_vram_mb,

        "preferred_model":
            preferred_model,

        "active_substitute":
            active_substitute,

        "action_metadata":
            action_metadata,

        "total_cost":
            total_cost,
    }


    return (
        problem,
        metadata,
    )
