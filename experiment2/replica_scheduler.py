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


class ReplicaScheduler(BaseScheduler):
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

    name = "replica"

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
    ):
        super().__init__(profiles)

        self.feasible_assignments = (
            feasible_assignments
        )

        self.engine_name = engine_name


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
        task_release_times=None,
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

        task_release_times = dict(
            task_release_times
            or {}
        )


        completed = set()

        task_end = {}

        assignments = []


        total_planning_overhead = 0.0

        planner_calls = 0

        decision_epochs = 0


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
        # ----------------------------------------------------

        def min_runtime(task):

            allowed = (
                self._feasible_resources(
                    task,
                    resources,
                    state,
                )
            )

            if not allowed:
                return None

            return min(
                self.runtime(
                    task.model,
                    task.stage,
                    rid,
                )
                for rid in allowed
            )


        # ----------------------------------------------------
        # Remaining critical path lower bound.
        #
        # Recomputed each decision epoch because feasibility
        # may change after reconstruction/replanning.
        # ----------------------------------------------------

        def build_criticality():

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
                    return float("inf")

                children = (
                    successors[
                        task_id
                    ]
                )

                if not children:
                    return float(own)

                return float(
                    own
                    + max(
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

                and all(
                    pred in completed
                    for pred
                    in task.predecessors
                )
            ]


            if not ready:

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


            # Highest remaining critical path first.
            ready.sort(
                key=lambda task: (
                    -criticality[
                        task.task_id
                    ],
                    task.task_id,
                )
            )


            # =================================================
            # CONTENTION-AWARE LIST SCHEDULING
            #
            # Use a provisional resource queue so every ready
            # task sees the assignments already proposed during
            # this SAME decision epoch.
            #
            # This is the key distinction from the legacy
            # implementation.
            # =================================================

            provisional_free = dict(
                resource_free
            )


            proposed_batch = []


            for task in ready:

                release_ready = float(
                    task_release_times.get(
                        task.task_id,
                        state.time_sec,
                    )
                )

                if task.predecessors:

                    pred_ready = max(
                        release_ready,
                        max(
                            task_end[p]
                            for p
                            in task.predecessors
                        ),
                    )

                else:

                    pred_ready = max(
                        state.time_sec,
                        release_ready,
                    )


                candidates = []


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
                        provisional_free[
                            rid
                        ],
                    )


                    end = (
                        start
                        + runtime
                    )


                    candidates.append(
                        (
                            end,
                            start,
                            runtime,
                            rid,
                        )
                    )


                # Earliest projected finish.
                #
                # Stable tie break:
                # start, runtime, resource id.
                candidates.sort(
                    key=lambda x: (
                        x[0],
                        x[1],
                        x[2],
                        x[3],
                    )
                )


                (
                    end,
                    start,
                    runtime,
                    rid,
                ) = candidates[0]


                proposed_batch.append(
                    {
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
                )


                # IMPORTANT:
                #
                # Later ready tasks now see the queue created
                # by earlier critical tasks in this SAME epoch.
                provisional_free[
                    rid
                ] = end


            # =================================================
            # SYMBOLIC VALIDATION OF PROPOSED BATCH
            # =================================================

            (
                elapsed,
                selected,
                validation_status,
            ) = (
                self._validate_symbolic_batch(
                    batch=
                        proposed_batch,

                    resources=
                        resources,

                    state=
                        state,
                )
            )


            total_planning_overhead += (
                elapsed
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
                        in proposed_batch
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
                proposed_batch
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


                completed.add(
                    task.task_id
                )


            print(
                f"[replica] epoch={decision_epochs} "
                f"ready={len(ready)} "
                f"committed={len(proposed_batch)} "
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
