from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Tuple

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


class HEFTScheduler(BaseScheduler):
    """
    Empirical HEFT baseline.

    Standard HEFT structure:
      1. compute upward rank
      2. process tasks in descending rank
      3. assign each task to the feasible resource producing
         the earliest finish time

    Difference from stock SAGA HEFT:
      computation cost comes directly from the Experiment-1
      task x resource runtime matrix rather than task_cost/node_speed.

    Communication cost is zero in Experiment 2 so that all methods
    receive the same timing model.
    """

    name = "heft"

    def __init__(
        self,
        profiles: Dict[Tuple[str, str, str], ExecutionProfile],
        feasible_assignments: set[Tuple[str, str, str]],
    ):
        super().__init__(profiles)
        self.feasible_assignments = feasible_assignments


    def schedule(
        self,
        workflow: Workflow,
        resources: Dict[str, Resource],
        state: SystemState,
        task_release_times=None,
    ) -> ScheduleResult:

        t0 = time.perf_counter()

        tasks = workflow.tasks

        task_release_times = dict(
            task_release_times
            or {}
        )


        # ====================================================
        # Feasible resources per task
        # ====================================================

        feasible_resources = {}

        for task_id, task in tasks.items():

            choices = []

            for rid, resource in resources.items():

                key = (
                    task.model,
                    task.stage,
                    rid,
                )

                if key not in self.feasible_assignments:
                    continue

                if key not in self.profiles:
                    continue

                if rid not in state.resources:
                    continue

                if not task_resource_feasible(
                    task,
                    resource,
                    state.resources[rid],
                ):
                    continue

                choices.append(rid)

            if not choices:
                overhead = time.perf_counter() - t0

                return ScheduleResult(
                    scheduler=self.name,
                    success=False,
                    assignments=[],
                    scheduling_overhead_sec=overhead,
                    makespan_sec=None,
                    infeasible_tasks=[task_id],
                    notes="Task has no feasible resource.",
                )

            feasible_resources[task_id] = choices


        # ====================================================
        # Successor graph
        # ====================================================

        successors = {
            task_id: []
            for task_id in tasks
        }

        for task_id, task in tasks.items():

            for pred in task.predecessors:
                successors[pred].append(task_id)


        # ====================================================
        # Average computation cost
        #
        # HEFT upward rank traditionally uses average execution
        # cost across processors.
        # ====================================================

        average_runtime = {}

        for task_id, task in tasks.items():

            vals = [
                self.runtime(
                    task.model,
                    task.stage,
                    rid,
                )
                for rid in feasible_resources[task_id]
            ]

            average_runtime[task_id] = (
                sum(vals) / len(vals)
            )


        # ====================================================
        # Upward rank
        #
        # rank_u(i) =
        #   avg_comp(i) +
        #   max_j [ comm(i,j) + rank_u(j) ]
        #
        # Communication cost = 0 for Experiment 2.
        # ====================================================

        rank_cache = {}


        def upward_rank(task_id):

            if task_id in rank_cache:
                return rank_cache[task_id]

            succ = successors[task_id]

            if not succ:
                rank = average_runtime[task_id]

            else:
                rank = (
                    average_runtime[task_id]
                    + max(
                        upward_rank(child)
                        for child in succ
                    )
                )

            rank_cache[task_id] = rank
            return rank


        for task_id in tasks:
            upward_rank(task_id)


        priority_order = sorted(
            tasks.keys(),
            key=lambda tid: (
                -rank_cache[tid],
                tid,
            ),
        )


        # ====================================================
        # Scheduling state
        # ====================================================

        resource_jobs = defaultdict(list)

        task_end = {}

        assignments_by_task = {}


        def earliest_slot(
            rid: str,
            ready_time: float,
            duration: float,
        ):
            """
            Find earliest non-overlapping interval on resource.
            HEFT insertion-based scheduling.
            """

            jobs = sorted(
                resource_jobs[rid],
                key=lambda x: x.start_sec,
            )

            candidate = max(
                ready_time,
                state.resources[rid].busy_until_sec,
                state.time_sec,
            )

            for job in jobs:

                # Fits before this existing job.
                if candidate + duration <= job.start_sec:
                    return candidate

                # Otherwise move after this job if necessary.
                if candidate < job.end_sec:
                    candidate = job.end_sec

            return candidate


        # ====================================================
        # HEFT placement
        # ====================================================

        for task_id in priority_order:

            task = tasks[task_id]

            release_ready = float(
                task_release_times.get(
                    task_id,
                    state.time_sec,
                )
            )

            if task.predecessors:
                predecessor_ready = max(
                    release_ready,
                    max(
                        task_end[p]
                        for p in task.predecessors
                    ),
                )
            else:
                predecessor_ready = max(
                    state.time_sec,
                    release_ready,
                )


            candidates = []

            for rid in feasible_resources[task_id]:

                runtime = self.runtime(
                    task.model,
                    task.stage,
                    rid,
                )

                start = earliest_slot(
                    rid=rid,
                    ready_time=predecessor_ready,
                    duration=runtime,
                )

                end = start + runtime

                candidates.append(
                    (
                        end,
                        start,
                        runtime,
                        rid,
                    )
                )


            # Earliest finish time.
            # Deterministic tie-breaking.
            candidates.sort(
                key=lambda x: (
                    x[0],
                    x[1],
                    x[3],
                )
            )

            end, start, runtime, rid = candidates[0]

            assignment = Assignment(
                task_id=task_id,
                resource_id=rid,
                start_sec=start,
                end_sec=end,
                estimated_runtime_sec=runtime,
            )

            assignments_by_task[task_id] = assignment
            task_end[task_id] = end

            resource_jobs[rid].append(assignment)


        assignments = sorted(
            assignments_by_task.values(),
            key=lambda a: (
                a.start_sec,
                a.end_sec,
                a.task_id,
            )
        )

        makespan = (
            max(a.end_sec for a in assignments)
            - state.time_sec
            if assignments
            else 0.0
        )

        overhead = time.perf_counter() - t0


        return ScheduleResult(
            scheduler=self.name,
            success=(
                len(assignments)
                == len(tasks)
            ),
            assignments=assignments,
            scheduling_overhead_sec=overhead,
            makespan_sec=makespan,
            infeasible_tasks=[],
            notes=(
                "Empirical HEFT; "
                "Experiment-1 task-resource runtimes; "
                "zero communication cost"
            ),
        )
