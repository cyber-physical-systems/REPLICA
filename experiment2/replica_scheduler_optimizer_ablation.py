from __future__ import annotations

import re
import time
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Tuple

from up_model_builder import build_planning_problem
from up_planner_runner import plan_assignments

from experiment2.common import (
    Assignment,
    ExecutionProfile,
    Resource,
    ScheduleResult,
    SystemState,
    Workflow,
    task_resource_feasible,
)

from experiment2.base_scheduler import BaseScheduler


# ============================================================
# HELPERS
# ============================================================

def safe_name(x: str) -> str:
    x = x.lower()
    x = re.sub(r"[^a-z0-9_]+", "_", x)
    return x.strip("_")


class ReplicaOptimizerAblationScheduler(BaseScheduler):
    """
    REPLICA scheduler for Experiment 2.

    Design
    ------
    REPLICA separates three responsibilities:

      1. symbolic state / feasibility:
         determine and validate what task-resource mappings
         are legal in the current reconstructed world;

      2. scheduling:
         rank ready work by remaining critical-path pressure
         and place it using earliest projected finish time;

      3. replanning:
         after a decision epoch changes execution state,
         reconstruct the world and solve again.

    The symbolic planner is therefore NOT asked to implement
    HEFT or CP-SAT internally. It validates the scheduler's
    proposed actions against the current symbolic world.

    This preserves REPLICA's planning/replanning architecture
    while giving the execution layer explicit awareness of
    resource contention and heterogeneous execution cost.
    """

    name = "replica_optimizer_ablation"

    def __init__(
        self,
        profiles: Dict[
            Tuple[str, str, str],
            ExecutionProfile,
        ],
        feasible_assignments: set[
            Tuple[str, str, str]
        ],
        engine_name: str = "enhsp-opt",

        # ----------------------------------------------------
        # Experimental ablation switches.
        #
        # IMPORTANT:
        # All default to False so the scheduler reproduces
        # Lookahead-128 behavior unless one mechanism is
        # explicitly enabled.
        # ----------------------------------------------------

        joint_frontier: bool = False,
        conditional_sufferage: bool = False,
        scarcity: bool = False,
        dominance_pruning: bool = False,
        memoization: bool = False,
        adaptive_beam: bool = False,
        adaptive_depth: bool = False,
        resource_reservation: bool = False,
        event_driven: bool = False,
    ):
        super().__init__(profiles)

        self.feasible_assignments = (
            feasible_assignments
        )

        self.engine_name = engine_name

        # Experimental configuration.
        #
        # These are intentionally inert for now.
        # We will implement each mechanism independently.
        self.joint_frontier = bool(
            joint_frontier
        )

        self.conditional_sufferage = bool(
            conditional_sufferage
        )

        self.scarcity = bool(
            scarcity
        )

        self.dominance_pruning = bool(
            dominance_pruning
        )

        self.memoization = bool(
            memoization
        )

        self.adaptive_beam = bool(
            adaptive_beam
        )

        self.adaptive_depth = bool(
            adaptive_depth
        )

        self.resource_reservation = bool(
            resource_reservation
        )

        self.event_driven = bool(
            event_driven
        )


    # ========================================================
    # CURRENT FEASIBILITY
    # ========================================================

    def _feasible_resources(
        self,
        task,
        resources,
        state,
    ):
        """
        Return resources that are valid for this task in the
        currently reconstructed world.
        """

        allowed = []

        for rid, resource in resources.items():

            key = (
                task.model,
                task.stage,
                rid,
            )

            # Dataset-driven Experiment-2 feasibility.
            if key not in self.feasible_assignments:
                continue

            if rid not in state.resources:
                continue

            rs = state.resources[rid]

            if not rs.available:
                continue

            # Shared physical resource constraints.
            if not task_resource_feasible(
                task,
                resource,
                rs,
            ):
                continue

            # Must have empirical Experiment-1 cost.
            if key not in self.profiles:
                continue

            allowed.append(rid)

        return sorted(allowed)


    # ========================================================
    # SYMBOLIC VALIDATION
    # ========================================================

    def _validate_symbolic_batch(
        self,
        batch,
        resources,
        state,
    ):
        """
        Validate the scheduler-proposed batch through the
        REPLICA symbolic planning model.

        Each task is temporarily restricted to the resource
        selected by the scheduling layer. The planner therefore
        answers:

            "Is this proposed set of actions valid in the
             currently reconstructed world?"

        Returns
        -------
        elapsed_sec, selected, status
        """

        planner_to_item = {}

        capabilities = {}

        telemetry = {
            rid: {
                "available":
                    bool(
                        state.resources[
                            rid
                        ].available
                    ),

                "cpu_load":
                    0.0,

                "gpu_util":
                    0.0,

                "free_ram_gb":
                    (
                        state.resources[
                            rid
                        ].ram_available_mb
                        / 1024.0
                        if state.resources[
                            rid
                        ].ram_available_mb
                        is not None
                        else resources[
                            rid
                        ].ram_mb
                        / 1024.0
                    ),

                "latency_ms":
                    0.0,

                "reliability":
                    1.0,

                "expected_runtime_sec":
                    {},
            }
            for rid in resources
        }


        for item in batch:

            task = item["task"]
            rid = item["resource_id"]

            pname = (
                f"{safe_name(task.model)}__"
                f"{safe_name(task.stage)}"
            )

            planner_to_item[pname] = item

            # Restrict symbolic validation to the resource
            # already selected by REPLICA's scheduling layer.
            capabilities[pname] = [rid]

            # Integer empirical planning cost.
            telemetry[rid][
                "expected_runtime_sec"
            ][pname] = int(
                round(
                    1000.0
                    * item["runtime"]
                )
            )


        planner_stages = list(
            planner_to_item
        )


        thresholds = {
            pname: {
                "min_free_ram_gb":
                    0.0,

                "max_cpu_load":
                    1.0,

                "max_gpu_util":
                    1.0,

                "max_latency_ms":
                    100000.0,

                "min_reliability":
                    0.0,
            }
            for pname in planner_stages
        }


        t0 = time.perf_counter()

        problem, _, _ = (
            build_planning_problem(
                telemetry=telemetry,
                stages=planner_stages,
                capabilities=capabilities,
                thresholds=thresholds,
                completed_stages=[],
                use_empirical_runtime_cost=True,
            )
        )


        result, selected = (
            plan_assignments(
                problem,
                engine_name=
                    self.engine_name,
            )
        )


        elapsed = (
            time.perf_counter()
            - t0
        )


        missing = [
            pname
            for pname
            in planner_stages
            if pname not in selected
        ]


        if missing:

            return (
                elapsed,
                selected,
                (
                    "symbolic_incomplete:"
                    + ",".join(missing)
                ),
            )


        # Ensure the symbolic planner validated exactly the
        # resource proposed by the scheduler.
        mismatched = []

        for pname, item in (
            planner_to_item.items()
        ):

            if (
                selected.get(pname)
                != item["resource_id"]
            ):

                mismatched.append(
                    pname
                )


        if mismatched:

            return (
                elapsed,
                selected,
                (
                    "symbolic_mismatch:"
                    + ",".join(
                        mismatched
                    )
                ),
            )


        return (
            elapsed,
            selected,
            "ok",
        )


    # ========================================================
    # MAIN SCHEDULER
    # ========================================================

    def schedule(
        self,
        workflow: Workflow,
        resources: Dict[
            str,
            Resource,
        ],
        state: SystemState,
    ) -> ScheduleResult:

        # ====================================================
        # TIMING
        #
        # scheduler_wall_start measures the COMPLETE REPLICA
        # scheduling procedure, making scheduling_overhead_sec
        # directly comparable with HEFT, EASY, and CP-SAT.
        #
        # total_planning_overhead remains a separate measure of
        # symbolic model construction + planner solve time.
        # ====================================================

        scheduler_wall_start = time.perf_counter()

        self.last_symbolic_planning_overhead_sec = None
        self.last_planner_calls = None
        self.last_decision_epochs = None

        # Decision-state instrumentation.
        #
        # Diagnostic only. This trace does NOT participate in
        # scheduler scoring, beam pruning, symbolic planning,
        # or assignment selection.
        self.last_decision_trace = []

        # Counterfactual candidate instrumentation.
        #
        # Diagnostic only. Candidate records are copied from
        # the beam AFTER normal beam search has finished.
        # They do not alter ranking, pruning, or selection.
        self.last_candidate_trace = []

        # Joint-frontier beam after each expansion depth.
        # Diagnostic only.
        self.last_joint_depth_trace = []


        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        resource_free = {
            rid: max(
                state.time_sec,
                state.resources[
                    rid
                ].busy_until_sec,
            )
            for rid in resources
        }


        completed = set()

        task_end = {}

        assignments = []


        total_planning_overhead = 0.0

        planner_calls = 0

        decision_epochs = 0


        # ----------------------------------------------------
        # Event-driven execution state.
        #
        # In normal mode, the scheduler preserves the existing
        # full-frontier commitment behavior.
        #
        # In event-driven mode:
        #
        #   completed     = tasks that have actually finished
        #   running_tasks = committed tasks still executing
        #   current_time  = simulated scheduling-event clock
        # ----------------------------------------------------

        current_time = float(
            state.time_sec
        )

        running_tasks = {}


        # ----------------------------------------------------
        # DAG successor map
        # ----------------------------------------------------

        successors = defaultdict(
            list
        )


        for task in (
            workflow.tasks.values()
        ):

            for pred in (
                task.predecessors
            ):

                successors[pred].append(
                    task.task_id
                )


        # ----------------------------------------------------
        # Current-world minimum runtime
        #
        # Used in remaining critical-path calculation.
        #
        # Optional memoization keeps these invariant values
        # across decision epochs within this schedule call.
        # ----------------------------------------------------

        persistent_min_runtime_cache = {}
        persistent_cp_cache = {}


        def min_runtime(task):

            if (
                self.memoization
                and task.task_id
                in persistent_min_runtime_cache
            ):
                return persistent_min_runtime_cache[
                    task.task_id
                ]

            allowed = (
                self._feasible_resources(
                    task,
                    resources,
                    state,
                )
            )

            if not allowed:
                return None

            value = min(
                self.runtime(
                    task.model,
                    task.stage,
                    rid,
                )
                for rid in allowed
            )

            if self.memoization:

                persistent_min_runtime_cache[
                    task.task_id
                ] = value

            return value


        # ----------------------------------------------------
        # Remaining critical path lower bound.
        #
        # Recomputed each decision epoch because feasibility
        # may change after reconstruction/replanning.
        # ----------------------------------------------------

        def build_criticality():

            # ------------------------------------------------
            # Memoized path
            #
            # The workflow DAG, empirical runtimes, feasibility,
            # and reconstructed resource state are constant
            # during this schedule() invocation. Therefore each
            # task's downstream critical-path lower bound is
            # invariant across decision epochs.
            # ------------------------------------------------

            if self.memoization:

                def cp(task_id):

                    if (
                        task_id
                        in persistent_cp_cache
                    ):
                        return persistent_cp_cache[
                            task_id
                        ]


                    task = (
                        workflow.tasks[
                            task_id
                        ]
                    )

                    own = min_runtime(
                        task
                    )


                    if own is None:

                        value = float(
                            "inf"
                        )

                    else:

                        children = (
                            successors[
                                task_id
                            ]
                        )


                        if not children:

                            value = float(
                                own
                            )

                        else:

                            value = float(
                                own
                                +
                                max(
                                    cp(child)
                                    for child
                                    in children
                                )
                            )


                    persistent_cp_cache[
                        task_id
                    ] = value

                    return value


            # ------------------------------------------------
            # Exact Lookahead-128 baseline path.
            #
            # Cache exists only for this decision epoch.
            # ------------------------------------------------

            else:

                @lru_cache(None)
                def cp(task_id):

                    task = (
                        workflow.tasks[
                            task_id
                        ]
                    )

                    own = min_runtime(
                        task
                    )

                    if own is None:
                        return float(
                            "inf"
                        )

                    children = (
                        successors[
                            task_id
                        ]
                    )

                    if not children:
                        return float(
                            own
                        )

                    return float(
                        own
                        +
                        max(
                            cp(child)
                            for child
                            in children
                        )
                    )


            return {
                task_id: cp(task_id)
                for task_id
                in workflow.tasks
                if task_id
                not in completed
            }


        # ====================================================
        # REPLAN UNTIL WORKFLOW COMPLETE
        # ====================================================

        while (
            len(completed)
            <
            len(workflow.tasks)
        ):

            decision_epochs += 1


            # =================================================
            # EVENT-DRIVEN COMPLETION
            #
            # A committed task becomes completed only when the
            # simulated clock reaches its finish time.
            # =================================================

            if self.event_driven:

                newly_completed = [
                    task_id
                    for task_id, end_sec
                    in list(
                        running_tasks.items()
                    )

                    if (
                        float(end_sec)
                        <=
                        current_time + 1e-12
                    )
                ]


                for task_id in newly_completed:

                    completed.add(
                        task_id
                    )

                    del running_tasks[
                        task_id
                    ]


                # The final running task may have completed at
                # this event. Exit before reconstructing a ready
                # set, because an empty ready set is correct when
                # the workflow is finished.
                if (
                    len(completed)
                    >=
                    len(workflow.tasks)
                ):
                    break


            # =================================================
            # RECONSTRUCT CURRENT READY SET
            # =================================================

            ready = [
                task
                for task
                in workflow.tasks.values()

                if (
                    task.task_id
                    not in completed
                )

                and (
                    task.task_id
                    not in running_tasks
                )

                and all(
                    pred in completed
                    for pred
                    in task.predecessors
                )
            ]


            if not ready:

                if (
                    self.event_driven
                    and running_tasks
                ):

                    current_time = min(
                        float(end_sec)
                        for end_sec
                        in running_tasks.values()
                    )

                    continue


                remaining = sorted(
                    set(
                        workflow.tasks
                    )
                    - completed
                )

                return ScheduleResult(
                    scheduler=
                        self.name,

                    success=
                        False,

                    assignments=
                        assignments,

                    scheduling_overhead_sec=(
                        time.perf_counter()
                        - scheduler_wall_start
                    ),

                    makespan_sec=
                        None,

                    infeasible_tasks=
                        remaining,

                    notes=(
                        "No ready tasks; "
                        "workflow dependency unresolved."
                    ),
                )


            # =================================================
            # RECONSTRUCT SYMBOLIC FEASIBILITY
            # =================================================

            capabilities = {}


            for task in ready:

                allowed = (
                    self._feasible_resources(
                        task,
                        resources,
                        state,
                    )
                )

                capabilities[
                    task.task_id
                ] = allowed


            impossible = [
                task.task_id
                for task
                in ready
                if not capabilities[
                    task.task_id
                ]
            ]


            if impossible:

                return ScheduleResult(
                    scheduler=
                        self.name,

                    success=
                        False,

                    assignments=
                        assignments,

                    scheduling_overhead_sec=(
                        time.perf_counter()
                        - scheduler_wall_start
                    ),

                    makespan_sec=
                        None,

                    infeasible_tasks=
                        impossible,

                    notes=(
                        "Ready task has no "
                        "feasible resource."
                    ),
                )


            # =================================================
            # CRITICALITY
            # =================================================

            criticality = (
                build_criticality()
            )


            # =================================================
            # OPTIONAL CONDITIONAL SUFFERAGE
            #
            # Baseline Lookahead-128 ranks ready work by
            # remaining critical-path pressure.
            #
            # When enabled, add a sufferage signal ONLY when:
            #
            #   1. the task has at least two feasible resources;
            #   2. its best resource is also desired by another
            #      currently ready task;
            #   3. losing that resource increases earliest finish.
            #
            # Sufferage:
            #
            #   second-best EFT - best EFT
            #
            # This avoids replacing REPLICA's critical-path
            # policy with a pure sufferage scheduler.
            # =================================================

            conditional_sufferage = {
                task.task_id: 0.0
                for task in ready
            }


            if self.conditional_sufferage:

                # First determine the locally best resource for
                # every ready task under the CURRENT queue state.
                best_resource_by_task = {}

                eft_options_by_task = {}


                for task in ready:

                    if task.predecessors:

                        pred_ready = max(
                            task_end[p]
                            for p in task.predecessors
                        )

                    else:

                        pred_ready = (
                            state.time_sec
                        )


                    options = []


                    for rid in capabilities[
                        task.task_id
                    ]:

                        runtime = self.runtime(
                            task.model,
                            task.stage,
                            rid,
                        )

                        start = max(
                            pred_ready,
                            resource_free[rid],
                        )

                        finish = (
                            start
                            + runtime
                        )


                        options.append(
                            (
                                float(finish),
                                rid,
                            )
                        )


                    options.sort(
                        key=lambda x: (
                            x[0],
                            x[1],
                        )
                    )


                    eft_options_by_task[
                        task.task_id
                    ] = options


                    if options:

                        best_resource_by_task[
                            task.task_id
                        ] = options[0][1]


                # Number of ready tasks whose local first choice
                # is each resource.
                best_resource_demand = {
                    rid: 0
                    for rid in resources
                }


                for rid in (
                    best_resource_by_task.values()
                ):

                    best_resource_demand[
                        rid
                    ] += 1


                for task in ready:

                    options = (
                        eft_options_by_task[
                            task.task_id
                        ]
                    )


                    if len(options) < 2:
                        continue


                    best_finish = (
                        options[0][0]
                    )

                    best_rid = (
                        options[0][1]
                    )

                    second_finish = (
                        options[1][0]
                    )


                    # Activate sufferage only when the preferred
                    # resource is actually contested.
                    if (
                        best_resource_demand[
                            best_rid
                        ]
                        <= 1
                    ):
                        continue


                    regret = max(
                        0.0,
                        second_finish
                        -
                        best_finish,
                    )


                    conditional_sufferage[
                        task.task_id
                    ] = float(
                        regret
                    )


                # Critical-path pressure remains the dominant
                # scheduling signal. Sufferage contributes only
                # where current resource competition makes the
                # alternative placement materially worse.
                ready.sort(
                    key=lambda task: (
                        -(
                            criticality[
                                task.task_id
                            ]
                            +
                            conditional_sufferage[
                                task.task_id
                            ]
                        ),
                        -criticality[
                            task.task_id
                        ],
                        task.task_id,
                    )
                )


            else:

                # Exact Lookahead-128 baseline behavior.
                ready.sort(
                    key=lambda task: (
                        -criticality[
                            task.task_id
                        ],
                        task.task_id,
                    )
                )


            # =================================================
            # DECISION-STATE INSTRUMENTATION
            #
            # Everything in this block is observational only.
            # None of these variables influence the scheduler.
            # =================================================

            remaining_tasks = [
                task
                for task
                in workflow.tasks.values()
                if task.task_id
                not in completed
            ]

            ready_count = len(
                ready
            )

            remaining_count = len(
                remaining_tasks
            )

            completed_count = len(
                completed
            )


            # -----------------------------------------------
            # Current feasible-choice geometry
            # -----------------------------------------------

            option_counts = [
                len(
                    capabilities[
                        task.task_id
                    ]
                )
                for task
                in ready
            ]

            forced_ready_tasks = sum(
                1
                for n
                in option_counts
                if n == 1
            )

            flexible_ready_tasks = sum(
                1
                for n
                in option_counts
                if n > 1
            )

            total_ready_feasible_assignments = sum(
                option_counts
            )

            mean_ready_options = (
                sum(
                    option_counts
                )
                / len(
                    option_counts
                )
                if option_counts
                else 0.0
            )


            # -----------------------------------------------
            # Resource competition
            #
            # Number of currently ready tasks that could use
            # each resource.
            # -----------------------------------------------

            resource_competition = {
                rid: sum(
                    1
                    for task
                    in ready
                    if rid
                    in capabilities[
                        task.task_id
                    ]
                )
                for rid
                in resources
            }

            competition_values = list(
                resource_competition.values()
            )

            competition_max = (
                max(
                    competition_values
                )
                if competition_values
                else 0
            )

            competition_min = (
                min(
                    competition_values
                )
                if competition_values
                else 0
            )

            competition_range = (
                competition_max
                -
                competition_min
            )


            # -----------------------------------------------
            # Resource queue state before decision
            # -----------------------------------------------

            free_values = [
                float(
                    resource_free[
                        rid
                    ]
                )
                for rid
                in resources
            ]

            resource_free_min = min(
                free_values
            )

            resource_free_max = max(
                free_values
            )

            resource_free_range = (
                resource_free_max
                -
                resource_free_min
            )

            resource_free_mean = (
                sum(
                    free_values
                )
                / len(
                    free_values
                )
            )


            # -----------------------------------------------
            # Runtime heterogeneity / resource sensitivity
            #
            # For each ready task:
            #   fastest runtime
            #   second-fastest runtime
            #   slowest runtime
            #   ratios and regret of losing best resource
            # -----------------------------------------------

            runtime_features = {}

            runtime_best_values = []
            runtime_ratio_values = []
            runtime_regret_values = []

            for task in ready:

                vals = sorted(
                    (
                        float(
                            self.runtime(
                                task.model,
                                task.stage,
                                rid,
                            )
                        ),
                        rid,
                    )
                    for rid
                    in capabilities[
                        task.task_id
                    ]
                )

                best_runtime = (
                    vals[0][0]
                )

                best_resource = (
                    vals[0][1]
                )

                worst_runtime = (
                    vals[-1][0]
                )

                worst_resource = (
                    vals[-1][1]
                )

                if len(vals) > 1:

                    second_runtime = (
                        vals[1][0]
                    )

                    second_resource = (
                        vals[1][1]
                    )

                    second_best_regret = (
                        second_runtime
                        -
                        best_runtime
                    )

                else:

                    second_runtime = None
                    second_resource = None
                    second_best_regret = None


                runtime_ratio = (
                    worst_runtime
                    /
                    best_runtime
                    if best_runtime > 0
                    else 1.0
                )


                runtime_features[
                    task.task_id
                ] = {
                    "best_runtime_sec":
                        float(
                            best_runtime
                        ),

                    "best_resource":
                        best_resource,

                    "second_runtime_sec":
                        (
                            float(
                                second_runtime
                            )
                            if second_runtime
                            is not None
                            else None
                        ),

                    "second_resource":
                        second_resource,

                    "worst_runtime_sec":
                        float(
                            worst_runtime
                        ),

                    "worst_resource":
                        worst_resource,

                    "runtime_ratio_worst_to_best":
                        float(
                            runtime_ratio
                        ),

                    "second_best_regret_sec":
                        (
                            float(
                                second_best_regret
                            )
                            if second_best_regret
                            is not None
                            else None
                        ),
                }


                runtime_best_values.append(
                    best_runtime
                )

                runtime_ratio_values.append(
                    runtime_ratio
                )

                if (
                    second_best_regret
                    is not None
                ):
                    runtime_regret_values.append(
                        second_best_regret
                    )


            mean_runtime_ratio = (
                sum(
                    runtime_ratio_values
                )
                /
                len(
                    runtime_ratio_values
                )
                if runtime_ratio_values
                else 1.0
            )

            max_runtime_ratio = (
                max(
                    runtime_ratio_values
                )
                if runtime_ratio_values
                else 1.0
            )

            mean_second_best_regret = (
                sum(
                    runtime_regret_values
                )
                /
                len(
                    runtime_regret_values
                )
                if runtime_regret_values
                else 0.0
            )

            max_second_best_regret = (
                max(
                    runtime_regret_values
                )
                if runtime_regret_values
                else 0.0
            )


            # -----------------------------------------------
            # Criticality geometry
            # -----------------------------------------------

            ready_criticalities = [
                float(
                    criticality[
                        task.task_id
                    ]
                )
                for task
                in ready
            ]

            criticality_max = max(
                ready_criticalities
            )

            criticality_min = min(
                ready_criticalities
            )

            criticality_range = (
                criticality_max
                -
                criticality_min
            )

            criticality_mean = (
                sum(
                    ready_criticalities
                )
                /
                len(
                    ready_criticalities
                )
            )


            # Snapshot BEFORE bounded lookahead chooses action.
            decision_state_before = {
                "epoch":
                    int(
                        decision_epochs
                    ),

                "system_time_sec":
                    float(
                        state.time_sec
                    ),

                "completed_count":
                    int(
                        completed_count
                    ),

                "remaining_count":
                    int(
                        remaining_count
                    ),

                "ready_count":
                    int(
                        ready_count
                    ),

                "ready_task_ids":
                    [
                        task.task_id
                        for task
                        in ready
                    ],

                "forced_ready_tasks":
                    int(
                        forced_ready_tasks
                    ),

                "flexible_ready_tasks":
                    int(
                        flexible_ready_tasks
                    ),

                "total_ready_feasible_assignments":
                    int(
                        total_ready_feasible_assignments
                    ),

                "mean_ready_options":
                    float(
                        mean_ready_options
                    ),

                "option_counts_by_task":
                    {
                        task.task_id:
                            int(
                                len(
                                    capabilities[
                                        task.task_id
                                    ]
                                )
                            )
                        for task
                        in ready
                    },

                "feasible_resources_by_task":
                    {
                        task.task_id:
                            list(
                                capabilities[
                                    task.task_id
                                ]
                            )
                        for task
                        in ready
                    },

                "resource_competition":
                    {
                        rid:
                            int(
                                resource_competition[
                                    rid
                                ]
                            )
                        for rid
                        in resources
                    },

                "competition_max":
                    int(
                        competition_max
                    ),

                "competition_min":
                    int(
                        competition_min
                    ),

                "competition_range":
                    int(
                        competition_range
                    ),

                "resource_free_sec":
                    {
                        rid:
                            float(
                                resource_free[
                                    rid
                                ]
                            )
                        for rid
                        in resources
                    },

                "resource_free_min_sec":
                    float(
                        resource_free_min
                    ),

                "resource_free_max_sec":
                    float(
                        resource_free_max
                    ),

                "resource_free_mean_sec":
                    float(
                        resource_free_mean
                    ),

                "resource_free_range_sec":
                    float(
                        resource_free_range
                    ),

                "criticality_by_task":
                    {
                        task.task_id:
                            float(
                                criticality[
                                    task.task_id
                                ]
                            )
                        for task
                        in ready
                    },

                "criticality_max":
                    float(
                        criticality_max
                    ),

                "criticality_min":
                    float(
                        criticality_min
                    ),

                "criticality_mean":
                    float(
                        criticality_mean
                    ),

                "criticality_range":
                    float(
                        criticality_range
                    ),

                "runtime_features_by_task":
                    runtime_features,

                "mean_runtime_ratio":
                    float(
                        mean_runtime_ratio
                    ),

                "max_runtime_ratio":
                    float(
                        max_runtime_ratio
                    ),

                "mean_second_best_regret_sec":
                    float(
                        mean_second_best_regret
                    ),

                "max_second_best_regret_sec":
                    float(
                        max_second_best_regret
                    ),
            }


            # =================================================
            # OPTIONAL FUTURE RESOURCE SCARCITY
            #
            # Estimate how strongly FUTURE unfinished work
            # depends on each resource.
            #
            # A future task contributes:
            #
            #     1 / number_of_feasible_resources
            #
            # to every resource it can use.
            #
            # Therefore:
            #   - exclusive task -> pressure 1.0
            #   - 2 choices      -> pressure 0.5 each
            #   - 3 choices      -> pressure 0.333 each
            #
            # Current ready tasks are excluded because the beam
            # is already explicitly assigning them.
            #
            # When scarcity=False this signal is inert and the
            # original Lookahead-128 ranking is preserved.
            # =================================================

            scarcity_pressure = {
                rid: 0.0
                for rid in resources
            }

            normalized_scarcity_pressure = {
                rid: 0.0
                for rid in resources
            }


            if self.scarcity:

                current_ready_ids = {
                    task.task_id
                    for task in ready
                }


                for future_task in remaining_tasks:

                    if (
                        future_task.task_id
                        in current_ready_ids
                    ):
                        continue


                    allowed_future = (
                        self._feasible_resources(
                            future_task,
                            resources,
                            state,
                        )
                    )


                    if not allowed_future:
                        continue


                    contribution = (
                        1.0
                        /
                        float(
                            len(
                                allowed_future
                            )
                        )
                    )


                    for rid in allowed_future:

                        scarcity_pressure[
                            rid
                        ] += contribution


                max_pressure = max(
                    scarcity_pressure.values(),
                    default=0.0,
                )


                if max_pressure > 0.0:

                    normalized_scarcity_pressure = {
                        rid:
                            float(
                                scarcity_pressure[
                                    rid
                                ]
                                /
                                max_pressure
                            )
                        for rid in resources
                    }


            # =================================================
            # BOUNDED LOOKAHEAD ASSIGNMENT
            #
            # Legacy REPLICA committed each ready task to the
            # locally earliest-finishing resource immediately.
            #
            # This variant preserves critical-path ordering but
            # evaluates multiple resource-placement combinations
            # for the entire ready batch before committing.
            #
            # It is intentionally NOT a global optimizer such as
            # CP-SAT. Search is bounded to the current decision
            # epoch and is followed by the same symbolic
            # validation used by REPLICA.
            # =================================================

            # =================================================
            # OPTIONAL ADAPTIVE BEAM WIDTH
            #
            # Baseline Lookahead-128 always retains up to 128
            # partial schedules.
            #
            # When enabled, spend search effort according to the
            # ambiguity of the CURRENT ready frontier.
            #
            # branching_excess:
            #
            #     sum(max(0, feasible_options - 1))
            #
            # Forced tasks therefore add no search pressure,
            # while flexible tasks increase the beam budget.
            #
            # Maximum width remains 128, so this ablation never
            # searches more broadly than Lookahead-128.
            # =================================================

            if self.adaptive_beam:

                branching_excess = sum(
                    max(
                        0,
                        int(n) - 1,
                    )
                    for n in option_counts
                )


                if branching_excess <= 2:

                    BEAM_WIDTH = 16

                elif branching_excess <= 5:

                    BEAM_WIDTH = 32

                elif branching_excess <= 8:

                    BEAM_WIDTH = 64

                else:

                    BEAM_WIDTH = 128

            else:

                # Exact Lookahead-128 baseline behavior.
                BEAM_WIDTH = 128

            # -------------------------------------------------
            # Tail lower bound after each ready task.
            #
            # criticality(task) contains the minimum runtime of
            # this task plus the minimum downstream critical path.
            # Once a resource-specific runtime is chosen, the
            # remaining tail is therefore:
            #
            #     criticality - minimum current-world runtime
            #
            # -------------------------------------------------

            min_runtime_cache = {
                task.task_id:
                    min_runtime(task)
                for task in ready
            }

            tail_lb = {
                task.task_id:
                    max(
                        0.0,
                        criticality[
                            task.task_id
                        ]
                        -
                        min_runtime_cache[
                            task.task_id
                        ],
                    )
                for task in ready
            }


            # -------------------------------------------------
            # Beam state:
            #
            # {
            #   "free": projected resource-free times,
            #   "items": proposed task assignments,
            #   "projected_cp_finish": lower bound on final
            #                          workflow completion,
            #   "batch_finish": current batch completion,
            #   "sum_finish": tie breaker
            # }
            # -------------------------------------------------

            beam = [
                {
                    "free":
                        dict(
                            resource_free
                        ),

                    "items":
                        [],

                    "projected_cp_finish":
                        0.0,

                    "batch_finish":
                        0.0,

                    "sum_finish":
                        0.0,

                    "scarcity_penalty":
                        0.0,
                }
            ]


            # =================================================
            # OPTIONAL JOINT FRONTIER SEARCH
            #
            # Baseline Lookahead-128 fixes the task order first
            # using criticality, then explores only resource
            # assignments.
            #
            # When enabled, jointly explore:
            #
            #     next ready task
            #         x
            #     feasible resource
            #
            # while keeping the same bounded beam width.
            #
            # This is still local to the CURRENT ready frontier.
            # It does not globally optimize the full workflow.
            # =================================================

            if self.joint_frontier:

                ready_by_id = {
                    task.task_id: task
                    for task in ready
                }


                # Track which ready tasks are still unassigned
                # in each partial beam state.
                for partial in beam:

                    partial[
                        "remaining_ready"
                    ] = tuple(
                        task.task_id
                        for task in ready
                    )


                # One expansion level per task in the current
                # ready frontier.
                for _ in range(
                    len(ready)
                ):

                    expanded = []


                    for partial in beam:

                        remaining_ids = (
                            partial[
                                "remaining_ready"
                            ]
                        )


                        for task_id in remaining_ids:

                            task = (
                                ready_by_id[
                                    task_id
                                ]
                            )


                            if task.predecessors:

                                pred_ready = max(
                                    task_end[p]
                                    for p
                                    in task.predecessors
                                )

                            else:

                                pred_ready = (
                                    state.time_sec
                                )


                            for rid in (
                                capabilities[
                                    task_id
                                ]
                            ):

                                runtime = self.runtime(
                                    task.model,
                                    task.stage,
                                    rid,
                                )


                                start = max(
                                    pred_ready,
                                    partial[
                                        "free"
                                    ][rid],
                                )

                                end = (
                                    start
                                    + runtime
                                )


                                new_free = dict(
                                    partial[
                                        "free"
                                    ]
                                )

                                new_free[
                                    rid
                                ] = end


                                item = {
                                    "task":
                                        task,

                                    "resource_id":
                                        rid,

                                    "start":
                                        float(
                                            start
                                        ),

                                    "end":
                                        float(
                                            end
                                        ),

                                    "runtime":
                                        float(
                                            runtime
                                        ),

                                    "criticality":
                                        float(
                                            criticality[
                                                task_id
                                            ]
                                        ),
                                }


                                task_projected_finish = (
                                    end
                                    +
                                    tail_lb[
                                        task_id
                                    ]
                                )


                                new_remaining = tuple(
                                    x
                                    for x
                                    in remaining_ids
                                    if x != task_id
                                )


                                expanded.append(
                                    {
                                        "free":
                                            new_free,

                                        "items":
                                            partial[
                                                "items"
                                            ]
                                            +
                                            [
                                                item
                                            ],

                                        "projected_cp_finish":
                                            max(
                                                partial[
                                                    "projected_cp_finish"
                                                ],
                                                task_projected_finish,
                                            ),

                                        "batch_finish":
                                            max(
                                                partial[
                                                    "batch_finish"
                                                ],
                                                end,
                                            ),

                                        "sum_finish":
                                            partial[
                                                "sum_finish"
                                            ]
                                            +
                                            end,

                                        "scarcity_penalty":
                                            0.0,

                                        "remaining_ready":
                                            new_remaining,
                                    }
                                )


                    # =========================================
                    # JOINT-FRONTIER LOWER BOUND
                    #
                    # Different partial states may have scheduled
                    # different subsets of the ready frontier.
                    #
                    # Comparing only projected_cp_finish would
                    # unfairly favor states that postpone long or
                    # critical ready tasks, because those tasks
                    # would not yet contribute to the score.
                    #
                    # Therefore include an optimistic lower bound
                    # for EVERY ready task that remains unassigned.
                    #
                    # This changes only joint-frontier pruning.
                    # Baseline Lookahead-128 remains untouched.
                    # =========================================

                    for candidate in expanded:

                        joint_bound = float(
                            candidate[
                                "projected_cp_finish"
                            ]
                        )


                        for remaining_id in (
                            candidate[
                                "remaining_ready"
                            ]
                        ):

                            remaining_task = (
                                ready_by_id[
                                    remaining_id
                                ]
                            )


                            if (
                                remaining_task.predecessors
                            ):

                                remaining_pred_ready = max(
                                    task_end[p]
                                    for p
                                    in remaining_task.predecessors
                                )

                            else:

                                remaining_pred_ready = (
                                    state.time_sec
                                )


                            optimistic_finish = float(
                                "inf"
                            )


                            for remaining_rid in (
                                capabilities[
                                    remaining_id
                                ]
                            ):

                                remaining_runtime = (
                                    self.runtime(
                                        remaining_task.model,
                                        remaining_task.stage,
                                        remaining_rid,
                                    )
                                )


                                remaining_start = max(
                                    remaining_pred_ready,
                                    candidate[
                                        "free"
                                    ][remaining_rid],
                                )


                                remaining_finish = (
                                    remaining_start
                                    +
                                    remaining_runtime
                                    +
                                    tail_lb[
                                        remaining_id
                                    ]
                                )


                                optimistic_finish = min(
                                    optimistic_finish,
                                    remaining_finish,
                                )


                            joint_bound = max(
                                joint_bound,
                                optimistic_finish,
                            )


                        candidate[
                            "joint_frontier_bound"
                        ] = float(
                            joint_bound
                        )


                    # =========================================
                    # EVENT-DRIVEN TIE BREAK
                    #
                    # For normal Joint Frontier, preserve the
                    # existing Lookahead objective exactly.
                    #
                    # In event-driven mode, only actions that can
                    # begin at current_time become binding.
                    #
                    # Therefore, when candidates are otherwise
                    # equal on projected workflow finish and
                    # current batch finish, prefer the candidate
                    # that dispatches greater critical-path
                    # pressure NOW.
                    #
                    # sum_finish remains the next tie breaker.
                    # =========================================

                    if self.event_driven:

                        for candidate in expanded:

                            candidate[
                                "immediate_criticality"
                            ] = sum(
                                float(
                                    item[
                                        "criticality"
                                    ]
                                )

                                for item
                                in candidate[
                                    "items"
                                ]

                                if (
                                    float(
                                        item[
                                            "start"
                                        ]
                                    )
                                    <=
                                    current_time + 1e-12
                                )
                            )


                        expanded.sort(
                            key=lambda x: (
                                x[
                                    "joint_frontier_bound"
                                ],

                                x[
                                    "projected_cp_finish"
                                ],

                                x[
                                    "batch_finish"
                                ],

                                -x[
                                    "immediate_criticality"
                                ],

                                x[
                                    "sum_finish"
                                ],

                                tuple(
                                    item[
                                        "resource_id"
                                    ]
                                    for item
                                    in x[
                                        "items"
                                    ]
                                ),

                                tuple(
                                    item[
                                        "task"
                                    ].task_id
                                    for item
                                    in x[
                                        "items"
                                    ]
                                ),
                            )
                        )

                    else:

                        expanded.sort(
                            key=lambda x: (
                                x[
                                    "joint_frontier_bound"
                                ],

                                x[
                                    "projected_cp_finish"
                                ],

                                x[
                                    "batch_finish"
                                ],

                                x[
                                    "sum_finish"
                                ],

                                tuple(
                                    item[
                                        "resource_id"
                                    ]
                                    for item
                                    in x[
                                        "items"
                                    ]
                                ),

                                tuple(
                                    item[
                                        "task"
                                    ].task_id
                                    for item
                                    in x[
                                        "items"
                                    ]
                                ),
                            )
                        )


                    beam = expanded[
                        :BEAM_WIDTH
                    ]


                    # -----------------------------------------
                    # Diagnostic snapshot after this joint
                    # expansion depth.
                    # -----------------------------------------

                    self.last_joint_depth_trace.append(
                        {
                            "epoch":
                                int(
                                    decision_epochs
                                ),

                            "depth":
                                int(
                                    len(ready)
                                    -
                                    len(
                                        beam[0][
                                            "remaining_ready"
                                        ]
                                    )
                                )
                                if beam
                                else None,

                            "beam_count":
                                int(
                                    len(beam)
                                ),

                            "states":
                                [
                                    {
                                        "projected_cp_finish":
                                            float(
                                                b[
                                                    "projected_cp_finish"
                                                ]
                                            ),

                                        "batch_finish":
                                            float(
                                                b[
                                                    "batch_finish"
                                                ]
                                            ),

                                        "sum_finish":
                                            float(
                                                b[
                                                    "sum_finish"
                                                ]
                                            ),

                                        "remaining_ready":
                                            list(
                                                b[
                                                    "remaining_ready"
                                                ]
                                            ),

                                        "assignments":
                                            [
                                                {
                                                    "task_id":
                                                        item[
                                                            "task"
                                                        ].task_id,

                                                    "resource_id":
                                                        item[
                                                            "resource_id"
                                                        ],

                                                    "start_sec":
                                                        float(
                                                            item[
                                                                "start"
                                                            ]
                                                        ),

                                                    "end_sec":
                                                        float(
                                                            item[
                                                                "end"
                                                            ]
                                                        ),
                                                }

                                                for item
                                                in b[
                                                    "items"
                                                ]
                                            ],
                                    }

                                    for b in beam
                                ],
                        }
                    )


                # Joint-frontier search is complete.
                #
                # Skip the existing fixed-order expansion below.
                joint_frontier_complete = True

            else:

                joint_frontier_complete = False


            if not joint_frontier_complete:

                for task in ready:

                    if task.predecessors:

                        pred_ready = max(
                            task_end[p]
                            for p
                            in task.predecessors
                        )

                    else:

                        pred_ready = (
                            state.time_sec
                        )


                    expanded = []


                    for partial in beam:

                        for rid in (
                            capabilities[
                                task.task_id
                            ]
                        ):

                            runtime = self.runtime(
                                task.model,
                                task.stage,
                                rid,
                            )

                            start = max(
                                pred_ready,
                                partial[
                                    "free"
                                ][rid],
                            )

                            end = (
                                start
                                + runtime
                            )


                            new_free = dict(
                                partial[
                                    "free"
                                ]
                            )

                            new_free[
                                rid
                            ] = end


                            item = {
                                "task":
                                    task,

                                "resource_id":
                                    rid,

                                "start":
                                    float(start),

                                "end":
                                    float(end),

                                "runtime":
                                    float(runtime),

                                "criticality":
                                    float(
                                        criticality[
                                            task.task_id
                                        ]
                                    ),
                            }


                            # Projected completion lower bound if
                            # this task follows its minimum-runtime
                            # downstream critical path from here.
                            task_projected_finish = (
                                end
                                +
                                tail_lb[
                                    task.task_id
                                ]
                            )


                            # -------------------------------------
                            # Optional scarcity consequence.
                            #
                            # Measure how much this partial schedule
                            # extends occupation of each resource
                            # beyond the resource-free time observed
                            # at the beginning of this decision epoch.
                            #
                            # Units remain seconds because normalized
                            # scarcity pressure is dimensionless.
                            # -------------------------------------

                            if self.scarcity:

                                scarcity_penalty = sum(
                                    max(
                                        0.0,
                                        float(
                                            new_free[
                                                scarcity_rid
                                            ]
                                        )
                                        -
                                        float(
                                            resource_free[
                                                scarcity_rid
                                            ]
                                        ),
                                    )
                                    *
                                    normalized_scarcity_pressure[
                                        scarcity_rid
                                    ]

                                    for scarcity_rid
                                    in resources
                                )

                            else:

                                scarcity_penalty = 0.0


                            expanded.append(
                                {
                                    "free":
                                        new_free,

                                    "items":
                                        partial[
                                            "items"
                                        ]
                                        +
                                        [
                                            item
                                        ],

                                    "projected_cp_finish":
                                        max(
                                            partial[
                                                "projected_cp_finish"
                                            ],
                                            task_projected_finish,
                                        ),

                                    "batch_finish":
                                        max(
                                            partial[
                                                "batch_finish"
                                            ],
                                            end,
                                        ),

                                    "sum_finish":
                                        partial[
                                            "sum_finish"
                                        ]
                                        +
                                        end,

                                    "scarcity_penalty":
                                        float(
                                            scarcity_penalty
                                        ),
                                }
                            )


                    # =================================================
                    # OPTIONAL PARETO DOMINANCE PRUNING
                    #
                    # Every state in `expanded` has assigned the same
                    # prefix of ready tasks.
                    #
                    # State A dominates State B when:
                    #
                    #   1. every resource in A becomes free no later
                    #      than the same resource in B;
                    #
                    #   2. every already-assigned task in A finishes
                    #      no later than that task in B;
                    #
                    #   3. at least one of those values is strictly
                    #      better.
                    #
                    # Such a B state cannot create an earlier feasible
                    # continuation than A, so retaining B only consumes
                    # bounded beam capacity.
                    #
                    # IMPORTANT:
                    # This does NOT introduce a new scheduling score.
                    # With dominance_pruning=False, baseline behavior
                    # is unchanged.
                    # =================================================

                    if self.dominance_pruning and len(expanded) > 1:

                        DOM_EPS = 1e-9


                        def dominates(a, b):
                            """
                            Return True iff beam state `a` Pareto-dominates
                            beam state `b` under resource availability and
                            completion times of the identical task prefix.
                            """

                            strictly_better = False


                            # -----------------------------------------
                            # Resource-free vector
                            # -----------------------------------------

                            for rid in resources:

                                av = float(
                                    a["free"][rid]
                                )

                                bv = float(
                                    b["free"][rid]
                                )


                                if av > bv + DOM_EPS:
                                    return False


                                if av < bv - DOM_EPS:
                                    strictly_better = True


                            # -----------------------------------------
                            # End time of every task in the common
                            # assigned prefix.
                            #
                            # Items are generated in the same ready-task
                            # order, so their positions correspond.
                            # -----------------------------------------

                            if len(a["items"]) != len(b["items"]):
                                return False


                            for a_item, b_item in zip(
                                a["items"],
                                b["items"],
                            ):

                                # Defensive check that we really are
                                # comparing the same task prefix.
                                if (
                                    a_item["task"].task_id
                                    !=
                                    b_item["task"].task_id
                                ):
                                    return False


                                ae = float(
                                    a_item["end"]
                                )

                                be = float(
                                    b_item["end"]
                                )


                                if ae > be + DOM_EPS:
                                    return False


                                if ae < be - DOM_EPS:
                                    strictly_better = True


                            return strictly_better


                        # ---------------------------------------------
                        # Pareto filter.
                        #
                        # expanded is normally small because the parent
                        # beam is bounded to 128 states and the system
                        # has only a few resources. O(n^2) comparison is
                        # therefore acceptable for this development
                        # ablation and keeps the implementation simple.
                        # ---------------------------------------------

                        keep = [
                            True
                            for _ in expanded
                        ]


                        for i in range(
                            len(expanded)
                        ):

                            if not keep[i]:
                                continue


                            for j in range(
                                len(expanded)
                            ):

                                if i == j:
                                    continue


                                if not keep[i]:
                                    break


                                if dominates(
                                    expanded[j],
                                    expanded[i],
                                ):

                                    keep[i] = False
                                    break


                        expanded = [
                            state_i
                            for state_i, keep_i
                            in zip(
                                expanded,
                                keep,
                            )
                            if keep_i
                        ]


                    # ---------------------------------------------
                    # Keep only the best bounded number of partial
                    # assignments.
                    #
                    # Primary objective:
                    #     projected remaining critical-path finish
                    #
                    # Tie breakers:
                    #     batch finish,
                    #     total completion time,
                    #     deterministic resource sequence
                    # ---------------------------------------------

                    if self.scarcity:

                        expanded.sort(
                            key=lambda x: (
                                x[
                                    "projected_cp_finish"
                                ]
                                +
                                x[
                                    "scarcity_penalty"
                                ],

                                x[
                                    "projected_cp_finish"
                                ],

                                x[
                                    "batch_finish"
                                ],

                                x[
                                    "sum_finish"
                                ],

                                tuple(
                                    item[
                                        "resource_id"
                                    ]
                                    for item
                                    in x[
                                        "items"
                                    ]
                                ),
                            )
                        )

                    else:

                        # Exact Lookahead-128 baseline ranking.
                        expanded.sort(
                            key=lambda x: (
                                x[
                                    "projected_cp_finish"
                                ],

                                x[
                                    "batch_finish"
                                ],

                                x[
                                    "sum_finish"
                                ],

                                tuple(
                                    item[
                                        "resource_id"
                                    ]
                                    for item
                                    in x[
                                        "items"
                                    ]
                                ),
                            )
                        )


                    beam = expanded[
                        :BEAM_WIDTH
                    ]


            # =================================================
            # FUTURE RESOURCE OPTION VALUE
            #
            # Diagnostic only.
            #
            # Reconstruct feasible resources for ALL unfinished
            # tasks using the same scheduler feasibility logic.
            #
            # The goal is to estimate how strategically valuable
            # each resource is to work that remains after the
            # current decision.
            # =================================================

            future_resource_value = {
                rid: 0.0
                for rid in resources
            }

            future_criticality_value = {
                rid: 0.0
                for rid in resources
            }

            future_best_task_count = {
                rid: 0
                for rid in resources
            }

            future_feasible_task_count = {
                rid: 0
                for rid in resources
            }

            future_exclusive_task_count = {
                rid: 0
                for rid in resources
            }

            future_max_regret = {
                rid: 0.0
                for rid in resources
            }

            future_task_features = {}


            # Current ready batch will be committed by one of the
            # candidates below. We are interested in option value
            # for work AFTER this batch.
            ready_ids = {
                task.task_id
                for task in ready
            }

            future_tasks = [
                task
                for task in remaining_tasks
                if task.task_id
                not in ready_ids
            ]


            for future_task in future_tasks:

                allowed_future = (
                    self._feasible_resources(
                        future_task,
                        resources,
                        state,
                    )
                )

                if not allowed_future:
                    continue


                runtime_options = sorted(
                    (
                        float(
                            self.runtime(
                                future_task.model,
                                future_task.stage,
                                rid,
                            )
                        ),
                        rid,
                    )
                    for rid
                    in allowed_future
                )


                best_runtime = (
                    runtime_options[0][0]
                )

                best_resource = (
                    runtime_options[0][1]
                )


                if len(runtime_options) > 1:

                    second_runtime = (
                        runtime_options[1][0]
                    )

                    best_regret = (
                        second_runtime
                        -
                        best_runtime
                    )

                else:

                    second_runtime = None

                    # Keep exclusivity explicit rather than
                    # inventing an arbitrary infinite regret.
                    best_regret = 0.0


                for rid in allowed_future:

                    future_feasible_task_count[
                        rid
                    ] += 1


                if len(allowed_future) == 1:

                    future_exclusive_task_count[
                        best_resource
                    ] += 1


                future_best_task_count[
                    best_resource
                ] += 1


                future_resource_value[
                    best_resource
                ] += float(
                    best_regret
                )


                task_criticality = float(
                    criticality.get(
                        future_task.task_id,
                        0.0,
                    )
                )


                future_criticality_value[
                    best_resource
                ] += (
                    float(
                        best_regret
                    )
                    *
                    max(
                        1.0,
                        task_criticality,
                    )
                )


                future_max_regret[
                    best_resource
                ] = max(
                    future_max_regret[
                        best_resource
                    ],
                    float(
                        best_regret
                    ),
                )


                future_task_features[
                    future_task.task_id
                ] = {
                    "feasible_resources":
                        list(
                            allowed_future
                        ),

                    "best_resource":
                        best_resource,

                    "best_runtime_sec":
                        float(
                            best_runtime
                        ),

                    "second_runtime_sec":
                        (
                            float(
                                second_runtime
                            )
                            if second_runtime
                            is not None
                            else None
                        ),

                    "best_resource_regret_sec":
                        float(
                            best_regret
                        ),

                    "exclusive":
                        bool(
                            len(
                                allowed_future
                            )
                            == 1
                        ),

                    "criticality":
                        float(
                            task_criticality
                        ),
                }


            # =================================================
            # COUNTERFACTUAL CANDIDATE TRACE
            #
            # At this point, beam contains the surviving complete
            # ready-batch assignments under the ORIGINAL
            # Lookahead-128 objective.
            #
            # We only COPY information from these candidates.
            # No candidate is reranked or modified here.
            # =================================================

            candidate_records = []

            current_free_values = [
                float(
                    resource_free[rid]
                )
                for rid in resources
            ]

            current_free_range = (
                max(current_free_values)
                -
                min(current_free_values)
            )


            for candidate_rank, candidate in enumerate(
                beam,
                start=1,
            ):

                projected_free = {
                    rid:
                        float(
                            candidate[
                                "free"
                            ][rid]
                        )
                    for rid in resources
                }

                projected_values = list(
                    projected_free.values()
                )

                projected_range = (
                    max(projected_values)
                    -
                    min(projected_values)
                )


                candidate_items = []

                for item in candidate[
                    "items"
                ]:

                    candidate_items.append(
                        {
                            "task_id":
                                item[
                                    "task"
                                ].task_id,

                            "model":
                                item[
                                    "task"
                                ].model,

                            "stage":
                                item[
                                    "task"
                                ].stage,

                            "resource_id":
                                item[
                                    "resource_id"
                                ],

                            "start_sec":
                                float(
                                    item[
                                        "start"
                                    ]
                                ),

                            "end_sec":
                                float(
                                    item[
                                        "end"
                                    ]
                                ),

                            "runtime_sec":
                                float(
                                    item[
                                        "runtime"
                                    ]
                                ),

                            "criticality":
                                float(
                                    item[
                                        "criticality"
                                    ]
                                ),
                        }
                    )


                # ---------------------------------------------
                # Candidate-specific future reservation conflict
                # ---------------------------------------------

                busy_extension = {
                    rid:
                        max(
                            0.0,
                            float(
                                projected_free[
                                    rid
                                ]
                            )
                            -
                            float(
                                resource_free[
                                    rid
                                ]
                            ),
                        )
                    for rid in resources
                }


                reservation_conflict = sum(
                    busy_extension[
                        rid
                    ]
                    *
                    future_resource_value[
                        rid
                    ]

                    for rid in resources
                )


                criticality_reservation_conflict = sum(
                    busy_extension[
                        rid
                    ]
                    *
                    future_criticality_value[
                        rid
                    ]

                    for rid in resources
                )


                exclusivity_conflict = sum(
                    busy_extension[
                        rid
                    ]
                    *
                    future_exclusive_task_count[
                        rid
                    ]

                    for rid in resources
                )


                best_task_conflict = sum(
                    busy_extension[
                        rid
                    ]
                    *
                    future_best_task_count[
                        rid
                    ]

                    for rid in resources
                )


                feasible_demand_conflict = sum(
                    busy_extension[
                        rid
                    ]
                    *
                    future_feasible_task_count[
                        rid
                    ]

                    for rid in resources
                )


                # Store candidate-level reservation metrics
                # directly on the beam state so the optional
                # reservation policy can rerank candidates later.
                candidate[
                    "reservation_conflict"
                ] = float(
                    reservation_conflict
                )

                candidate[
                    "criticality_reservation_conflict"
                ] = float(
                    criticality_reservation_conflict
                )

                candidate[
                    "exclusivity_conflict"
                ] = float(
                    exclusivity_conflict
                )

                candidate[
                    "best_task_conflict"
                ] = float(
                    best_task_conflict
                )

                candidate[
                    "feasible_demand_conflict"
                ] = float(
                    feasible_demand_conflict
                )


                candidate_records.append(
                    {
                        "candidate_rank":
                            int(
                                candidate_rank
                            ),

                        "chosen_by_original_objective":
                            bool(
                                candidate_rank == 1
                            ),

                        "projected_cp_finish":
                            float(
                                candidate[
                                    "projected_cp_finish"
                                ]
                            ),

                        "batch_finish":
                            float(
                                candidate[
                                    "batch_finish"
                                ]
                            ),

                        "sum_finish":
                            float(
                                candidate[
                                    "sum_finish"
                                ]
                            ),

                        "resource_free_before":
                            {
                                rid:
                                    float(
                                        resource_free[
                                            rid
                                        ]
                                    )
                                for rid
                                in resources
                            },

                        "resource_free_after":
                            projected_free,

                        "resource_free_range_before":
                            float(
                                current_free_range
                            ),

                        "resource_free_range_after":
                            float(
                                projected_range
                            ),

                        "delta_resource_free_range":
                            float(
                                projected_range
                                -
                                current_free_range
                            ),

                        "busy_extension_by_resource":
                            {
                                rid:
                                    float(
                                        busy_extension[
                                            rid
                                        ]
                                    )
                                for rid
                                in resources
                            },

                        "future_resource_value":
                            {
                                rid:
                                    float(
                                        future_resource_value[
                                            rid
                                        ]
                                    )
                                for rid
                                in resources
                            },

                        "future_criticality_value":
                            {
                                rid:
                                    float(
                                        future_criticality_value[
                                            rid
                                        ]
                                    )
                                for rid
                                in resources
                            },

                        "future_best_task_count":
                            {
                                rid:
                                    int(
                                        future_best_task_count[
                                            rid
                                        ]
                                    )
                                for rid
                                in resources
                            },

                        "future_feasible_task_count":
                            {
                                rid:
                                    int(
                                        future_feasible_task_count[
                                            rid
                                        ]
                                    )
                                for rid
                                in resources
                            },

                        "future_exclusive_task_count":
                            {
                                rid:
                                    int(
                                        future_exclusive_task_count[
                                            rid
                                        ]
                                    )
                                for rid
                                in resources
                            },

                        "reservation_conflict":
                            float(
                                reservation_conflict
                            ),

                        "criticality_reservation_conflict":
                            float(
                                criticality_reservation_conflict
                            ),

                        "exclusivity_conflict":
                            float(
                                exclusivity_conflict
                            ),

                        "best_task_conflict":
                            float(
                                best_task_conflict
                            ),

                        "feasible_demand_conflict":
                            float(
                                feasible_demand_conflict
                            ),

                        "assignments":
                            candidate_items,
                    }
                )


            self.last_candidate_trace.append(
                {
                    "epoch":
                        int(
                            decision_epochs
                        ),

                    "state_before":
                        decision_state_before,

                    "candidate_count":
                        int(
                            len(
                                candidate_records
                            )
                        ),

                    "candidates":
                        candidate_records,
                }
            )


            # =================================================
            # OPTIONAL CRITICAL-RESOURCE RESERVATION
            #
            # The normal Lookahead-128 objective remains primary.
            #
            # Reservation is allowed to influence the decision
            # ONLY among candidates that are already within 1%
            # of the best projected critical-path finish.
            #
            # This prevents the reservation heuristic from
            # repeating the failure mode of the earlier generic
            # scarcity penalty, which could sacrifice too much
            # current makespan for speculative future benefit.
            #
            # Within the near-optimal candidate set prefer:
            #
            #   1. less occupation of resources required
            #      exclusively by future work;
            #   2. less criticality-weighted future regret;
            #   3. less ordinary future-resource regret;
            #   4. the original Lookahead-128 objective.
            #
            # When resource_reservation=False, beam ordering is
            # unchanged.
            # =================================================

            if self.resource_reservation and beam:

                RESERVATION_SLACK = 0.01

                best_projected_cp = min(
                    float(
                        candidate[
                            "projected_cp_finish"
                        ]
                    )
                    for candidate
                    in beam
                )


                reservation_limit = (
                    best_projected_cp
                    *
                    (
                        1.0
                        +
                        RESERVATION_SLACK
                    )
                )


                near_optimal = [
                    candidate
                    for candidate
                    in beam
                    if (
                        float(
                            candidate[
                                "projected_cp_finish"
                            ]
                        )
                        <=
                        reservation_limit
                        + 1e-9
                    )
                ]


                # There should always be at least the original
                # best candidate, but keep this defensive.
                if near_optimal:

                    near_optimal.sort(
                        key=lambda x: (
                            x.get(
                                "exclusivity_conflict",
                                0.0,
                            ),

                            x.get(
                                "criticality_reservation_conflict",
                                0.0,
                            ),

                            x.get(
                                "reservation_conflict",
                                0.0,
                            ),

                            x[
                                "projected_cp_finish"
                            ],

                            x[
                                "batch_finish"
                            ],

                            x[
                                "sum_finish"
                            ],

                            tuple(
                                item[
                                    "resource_id"
                                ]
                                for item
                                in x[
                                    "items"
                                ]
                            ),
                        )
                    )


                    reservation_choice = (
                        near_optimal[0]
                    )


                    # Move chosen reservation-aware candidate to
                    # the front. Preserve all other beam states
                    # and their existing order.
                    beam = [
                        reservation_choice
                    ] + [
                        candidate
                        for candidate
                        in beam
                        if candidate
                        is not reservation_choice
                    ]


            # =================================================
            # OPTIONAL ADAPTIVE DEPTH
            #
            # Lookahead-128 already searches the entire CURRENT
            # ready frontier. "Depth" therefore means looking one
            # frontier beyond the current decision.
            #
            # For each surviving complete beam candidate:
            #
            #   1. assume the current ready batch is completed;
            #   2. identify tasks whose predecessors would then
            #      all be complete;
            #   3. estimate each next-frontier task's earliest
            #      finish using the candidate's projected resource
            #      availability;
            #   4. combine that finish with its downstream tail
            #      lower bound.
            #
            # This gives REPLICA limited awareness of the NEXT
            # decision epoch without turning the scheduler into
            # a global CP-SAT-style optimizer.
            #
            # When adaptive_depth=False, no reranking occurs.
            # =================================================

            if self.adaptive_depth and beam:

                completed_after_batch = (
                    set(completed)
                    |
                    {
                        item["task"].task_id
                        for item
                        in beam[0]["items"]
                    }
                )


                # Candidate-independent identity of tasks that
                # become ready after this frontier is committed.
                next_frontier = [
                    future_task
                    for future_task
                    in workflow.tasks.values()

                    if (
                        future_task.task_id
                        not in completed_after_batch
                    )

                    and all(
                        pred
                        in completed_after_batch
                        for pred
                        in future_task.predecessors
                    )
                ]


                def candidate_next_frontier_score(
                    candidate,
                ):

                    # If no new frontier is exposed, the current
                    # objective remains sufficient.
                    if not next_frontier:
                        return float(
                            candidate[
                                "projected_cp_finish"
                            ]
                        )


                    next_pressure = 0.0


                    for future_task in next_frontier:

                        allowed_future = (
                            self._feasible_resources(
                                future_task,
                                resources,
                                state,
                            )
                        )


                        if not allowed_future:
                            return float("inf")


                        # All predecessors should belong either to
                        # previously completed work or the current
                        # candidate batch.
                        pred_ready = state.time_sec

                        if future_task.predecessors:

                            pred_times = []

                            for pred in (
                                future_task.predecessors
                            ):

                                if pred in task_end:

                                    pred_times.append(
                                        float(
                                            task_end[pred]
                                        )
                                    )

                                else:

                                    matched = [
                                        item
                                        for item
                                        in candidate[
                                            "items"
                                        ]
                                        if (
                                            item[
                                                "task"
                                            ].task_id
                                            == pred
                                        )
                                    ]

                                    if matched:

                                        pred_times.append(
                                            float(
                                                matched[0][
                                                    "end"
                                                ]
                                            )
                                        )


                            if pred_times:

                                pred_ready = max(
                                    pred_times
                                )


                        best_finish = float("inf")


                        for rid in allowed_future:

                            runtime = self.runtime(
                                future_task.model,
                                future_task.stage,
                                rid,
                            )


                            start = max(
                                float(
                                    pred_ready
                                ),
                                float(
                                    candidate[
                                        "free"
                                    ][rid]
                                ),
                            )


                            finish = (
                                start
                                + runtime
                            )


                            best_finish = min(
                                best_finish,
                                finish,
                            )


                        # Remaining downstream tail after this
                        # future task.
                        future_cp = float(
                            criticality.get(
                                future_task.task_id,
                                min_runtime(
                                    future_task
                                )
                                or 0.0,
                            )
                        )

                        future_min = (
                            min_runtime(
                                future_task
                            )
                            or 0.0
                        )

                        future_tail = max(
                            0.0,
                            future_cp
                            -
                            float(
                                future_min
                            ),
                        )


                        next_pressure = max(
                            next_pressure,
                            best_finish
                            +
                            future_tail,
                        )


                    return max(
                        float(
                            candidate[
                                "projected_cp_finish"
                            ]
                        ),
                        float(
                            next_pressure
                        ),
                    )


                # Compute one-frontier-deeper score for each
                # candidate and rerank the surviving beam.
                for candidate in beam:

                    candidate[
                        "adaptive_depth_score"
                    ] = (
                        candidate_next_frontier_score(
                            candidate
                        )
                    )


                beam.sort(
                    key=lambda x: (
                        x[
                            "adaptive_depth_score"
                        ],

                        x[
                            "projected_cp_finish"
                        ],

                        x[
                            "batch_finish"
                        ],

                        x[
                            "sum_finish"
                        ],

                        tuple(
                            item[
                                "resource_id"
                            ]
                            for item
                            in x[
                                "items"
                            ]
                        ),
                    )
                )


            if not beam:

                return ScheduleResult(
                    scheduler=
                        self.name,

                    success=
                        False,

                    assignments=
                        assignments,

                    scheduling_overhead_sec=(
                        time.perf_counter()
                        -
                        scheduler_wall_start
                    ),

                    makespan_sec=
                        None,

                    infeasible_tasks=[
                        task.task_id
                        for task in ready
                    ],

                    notes=(
                        "Bounded lookahead could not "
                        "construct a feasible ready batch."
                    ),
                )


            best_batch = beam[0]

            proposed_batch = (
                best_batch[
                    "items"
                ]
            )


            # =================================================
            # EVENT-DRIVEN PARTIAL COMMIT
            #
            # Joint Frontier may project future placements for
            # scoring, but only tasks that can actually dispatch
            # at the current event time become binding actions.
            #
            # This leaves future queued work available for
            # reconsideration after the next completion event.
            # =================================================

            if self.event_driven:

                commit_batch = [
                    item
                    for item
                    in proposed_batch

                    if (
                        float(
                            item[
                                "start"
                            ]
                        )
                        <=
                        current_time + 1e-12
                    )
                ]


                # If the current projection contains no action
                # that can begin now, advance to the next running
                # completion and reconstruct the world.
                if not commit_batch:

                    if running_tasks:

                        current_time = min(
                            float(end_sec)
                            for end_sec
                            in running_tasks.values()
                        )

                        continue


                    return ScheduleResult(
                        scheduler=
                            self.name,

                        success=
                            False,

                        assignments=
                            assignments,

                        scheduling_overhead_sec=(
                            time.perf_counter()
                            -
                            scheduler_wall_start
                        ),

                        makespan_sec=
                            None,

                        infeasible_tasks=[
                            task.task_id
                            for task
                            in ready
                        ],

                        notes=(
                            "Event-driven scheduler found "
                            "no executable action and no "
                            "running task to advance."
                        ),
                    )

            else:

                commit_batch = (
                    proposed_batch
                )


            # =================================================
            # SYMBOLIC VALIDATION OF COMMITTED ACTIONS
            # =================================================

            (
                elapsed,
                selected,
                validation_status,
            ) = (
                self._validate_symbolic_batch(
                    batch=
                        commit_batch,

                    resources=
                        resources,

                    state=
                        state,
                )
            )


            total_planning_overhead += (
                elapsed
            )

            # =================================================
            # RECORD ACTION SELECTED BY LOOKAHEAD
            #
            # Diagnostic only.
            # =================================================

            selected_action = [
                {
                    "task_id":
                        item[
                            "task"
                        ].task_id,

                    "model":
                        item[
                            "task"
                        ].model,

                    "stage":
                        item[
                            "task"
                        ].stage,

                    "resource_id":
                        item[
                            "resource_id"
                        ],

                    "start_sec":
                        float(
                            item[
                                "start"
                            ]
                        ),

                    "end_sec":
                        float(
                            item[
                                "end"
                            ]
                        ),

                    "runtime_sec":
                        float(
                            item[
                                "runtime"
                            ]
                        ),

                    "criticality":
                        float(
                            item[
                                "criticality"
                            ]
                        ),
                }
                for item
                in commit_batch
            ]


            projected_free_after = dict(
                resource_free
            )

            for item in commit_batch:

                rid = item[
                    "resource_id"
                ]

                projected_free_after[
                    rid
                ] = max(
                    projected_free_after[
                        rid
                    ],
                    float(
                        item[
                            "end"
                        ]
                    ),
                )


            projected_free_values = [
                float(
                    projected_free_after[
                        rid
                    ]
                )
                for rid
                in resources
            ]


            self.last_decision_trace.append(
                {
                    "state_before":
                        decision_state_before,

                    "action":
                        selected_action,

                    "projected_state_after":
                        {
                            "resource_free_sec":
                                {
                                    rid:
                                        float(
                                            projected_free_after[
                                                rid
                                            ]
                                        )
                                    for rid
                                    in resources
                                },

                            "resource_free_min_sec":
                                float(
                                    min(
                                        projected_free_values
                                    )
                                ),

                            "resource_free_max_sec":
                                float(
                                    max(
                                        projected_free_values
                                    )
                                ),

                            "resource_free_range_sec":
                                float(
                                    max(
                                        projected_free_values
                                    )
                                    -
                                    min(
                                        projected_free_values
                                    )
                                ),

                            "batch_finish_sec":
                                float(
                                    max(
                                        item[
                                            "end"
                                        ]
                                        for item
                                        in commit_batch
                                    )
                                ),

                            "committed_task_count":
                                int(
                                    len(
                                        commit_batch
                                    )
                                ),
                        },
                }
            )


            planner_calls += 1


            if validation_status != "ok":

                return ScheduleResult(
                    scheduler=
                        self.name,

                    success=
                        False,

                    assignments=
                        assignments,

                    scheduling_overhead_sec=(
                        time.perf_counter()
                        - scheduler_wall_start
                    ),

                    makespan_sec=
                        None,

                    infeasible_tasks=[
                        item[
                            "task"
                        ].task_id
                        for item
                        in commit_batch
                    ],

                    notes=(
                        "Symbolic validation "
                        "failed: "
                        f"{validation_status}"
                    ),
                )


            # =================================================
            # COMMIT CURRENT DECISION EPOCH
            #
            # Every proposed task was ready in the same
            # reconstructed world and its resource placement
            # has been symbolically validated.
            #
            # Multiple tasks may be queued on one resource,
            # but their projected start/end intervals do not
            # overlap because provisional_free was updated
            # during list scheduling.
            # =================================================

            for item in (
                commit_batch
            ):

                task = item[
                    "task"
                ]

                rid = item[
                    "resource_id"
                ]


                assignment = (
                    Assignment(
                        task_id=
                            task.task_id,

                        resource_id=
                            rid,

                        start_sec=
                            item["start"],

                        end_sec=
                            item["end"],

                        estimated_runtime_sec=
                            item["runtime"],
                    )
                )


                assignments.append(
                    assignment
                )


                resource_free[rid] = max(
                    resource_free[
                        rid
                    ],
                    item["end"],
                )


                task_end[
                    task.task_id
                ] = item[
                    "end"
                ]


                if self.event_driven:

                    running_tasks[
                        task.task_id
                    ] = float(
                        item[
                            "end"
                        ]
                    )

                else:

                    completed.add(
                        task.task_id
                    )


            if (
                self.event_driven
                and running_tasks
            ):

                current_time = min(
                    float(end_sec)
                    for end_sec
                    in running_tasks.values()
                )


            print(
                f"[replica] epoch={decision_epochs} "
                f"ready={len(ready)} "
                f"committed={len(commit_batch)} "
                f"running={len(running_tasks)} "
                f"completed={len(completed)}/"
                f"{len(workflow.tasks)}"
            )


        # ====================================================
        # FINALIZE
        # ====================================================

        assignments.sort(
            key=lambda a: (
                a.start_sec,
                a.task_id,
            )
        )


        makespan = (
            max(
                a.end_sec
                for a
                in assignments
            )
            if assignments
            else 0.0
        )


        # ====================================================
        # FINAL TIMING
        # ====================================================

        total_scheduler_overhead = (
            time.perf_counter()
            - scheduler_wall_start
        )

        # Expose REPLICA-specific diagnostics to the benchmark
        # runner without changing the shared ScheduleResult API.
        self.last_symbolic_planning_overhead_sec = float(
            total_planning_overhead
        )
        self.last_planner_calls = int(
            planner_calls
        )
        self.last_decision_epochs = int(
            decision_epochs
        )

        print(
            f"[replica] planner_calls={planner_calls} "
            f"decision_epochs={decision_epochs} "
            f"symbolic_overhead="
            f"{total_planning_overhead:.3f}s "
            f"total_overhead="
            f"{total_scheduler_overhead:.3f}s"
        )


        return ScheduleResult(
            scheduler=
                self.name,

            success=
                True,

            assignments=
                assignments,

            scheduling_overhead_sec=
                total_scheduler_overhead,

            makespan_sec=
                makespan,

            infeasible_tasks=
                [],

            notes=(
                "REPLICA symbolic-validation + "
                "critical-path/EFT scheduler. "
                f"planner_calls={planner_calls}; "
                f"decision_epochs={decision_epochs}."
            ),
        )
