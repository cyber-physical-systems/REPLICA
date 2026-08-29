from __future__ import annotations

import time
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


class EASYBackfillScheduler(BaseScheduler):
    """
    Generalized EASY Backfill baseline for heterogeneous resources.

    Policy
    ------
    1. Tasks enter a deterministic FCFS queue when their predecessors
       have completed.
    2. The first queued task receives a reservation on the resource
       giving its earliest feasible execution opportunity.
    3. Later ready tasks may backfill onto currently free resources
       only if they do not delay the head-of-line reservation.
    4. Each resource executes at most one task at a time.

    Experiment-1 measured task-resource runtimes are used directly.
    """

    name = "easy_backfill"

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

        wall_start = time.perf_counter()

        tasks = workflow.tasks

        task_release_times = dict(
            task_release_times
            or {}
        )

        # Preserve workload insertion order as deterministic FCFS tie-break.
        priority = {
            task_id: i
            for i, task_id in enumerate(tasks.keys())
        }

        # ------------------------------------------------------------
        # Feasible resource set per task
        # ------------------------------------------------------------

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
                overhead = time.perf_counter() - wall_start

                return ScheduleResult(
                    scheduler=self.name,
                    success=False,
                    assignments=[],
                    scheduling_overhead_sec=overhead,
                    makespan_sec=None,
                    infeasible_tasks=[task_id],
                    notes="Task has no feasible resource.",
                )

            feasible_resources[task_id] = sorted(choices)


        # ------------------------------------------------------------
        # Simulation state
        # ------------------------------------------------------------

        now = state.time_sec

        completed = set()
        started = set()

        assignments = []

        # resource -> currently running Assignment or None
        running = {
            rid: None
            for rid in resources
        }

        # Respect any initial busy-until values.
        initially_busy_until = {
            rid: max(
                state.time_sec,
                state.resources[rid].busy_until_sec,
            )
            for rid in resources
        }


        def ready_task_ids():
            """
            FCFS queue of tasks whose predecessors are complete.
            """
            ready = []

            for tid, task in tasks.items():

                if tid in completed or tid in started:
                    continue

                release_time = float(
                    task_release_times.get(
                        tid,
                        state.time_sec,
                    )
                )

                if (
                    all(
                        pred in completed
                        for pred in task.predecessors
                    )
                    and
                    release_time <= now + 1e-12
                ):
                    ready.append(tid)

            ready.sort(key=lambda tid: priority[tid])
            return ready


        def resource_available_time(rid):
            """
            Earliest time resource can begin another task.
            """

            job = running[rid]

            if job is not None:
                return job.end_sec

            return max(
                now,
                initially_busy_until[rid],
            )


        def reservation_for(task_id):
            """
            EASY reservation for head-of-line task.

            Choose resource by:
              1. earliest start
              2. earliest finish
              3. deterministic resource id
            """

            task = tasks[task_id]

            candidates = []

            release_time = float(
                task_release_times.get(
                    task_id,
                    state.time_sec,
                )
            )

            for rid in feasible_resources[task_id]:

                start = max(
                    resource_available_time(rid),
                    release_time,
                )

                runtime = self.runtime(
                    task.model,
                    task.stage,
                    rid,
                )

                end = start + runtime

                candidates.append(
                    (
                        start,
                        end,
                        rid,
                        runtime,
                    )
                )

            candidates.sort(
                key=lambda x: (
                    x[0],
                    x[1],
                    x[2],
                )
            )

            return candidates[0]


        def start_task(task_id, rid, start_time):
            task = tasks[task_id]

            runtime = self.runtime(
                task.model,
                task.stage,
                rid,
            )

            assignment = Assignment(
                task_id=task_id,
                resource_id=rid,
                start_sec=start_time,
                end_sec=start_time + runtime,
                estimated_runtime_sec=runtime,
            )

            started.add(task_id)
            assignments.append(assignment)
            running[rid] = assignment

            return assignment


        # ------------------------------------------------------------
        # Main EASY simulation
        # ------------------------------------------------------------

        while len(completed) < len(tasks):

            # --------------------------------------------------------
            # Complete jobs ending at current time.
            # --------------------------------------------------------

            for rid in list(running):

                job = running[rid]

                if (
                    job is not None
                    and job.end_sec <= now + 1e-12
                ):
                    completed.add(job.task_id)
                    running[rid] = None


            # --------------------------------------------------------
            # Repeatedly dispatch at this same event time while
            # legitimate EASY starts are available.
            # --------------------------------------------------------

            dispatched_at_this_time = True

            while dispatched_at_this_time:

                dispatched_at_this_time = False

                queue = ready_task_ids()

                if not queue:
                    break

                head = queue[0]

                (
                    reserved_start,
                    reserved_end,
                    reserved_resource,
                    reserved_runtime,
                ) = reservation_for(head)


                # ====================================================
                # HEAD-OF-LINE CAN START NOW
                # ====================================================

                if reserved_start <= now + 1e-12:

                    start_task(
                        head,
                        reserved_resource,
                        now,
                    )

                    dispatched_at_this_time = True
                    continue


                # ====================================================
                # HEAD IS RESERVED IN THE FUTURE.
                #
                # Try EASY backfill candidates.
                # ====================================================

                free_resources = [
                    rid
                    for rid in resources
                    if running[rid] is None
                    and initially_busy_until[rid]
                        <= now + 1e-12
                ]

                if not free_resources:
                    break


                backfilled = False

                # FCFS through jobs behind head.
                for candidate_id in queue[1:]:

                    candidate = tasks[candidate_id]

                    candidate_options = []

                    for rid in free_resources:

                        if rid not in feasible_resources[candidate_id]:
                            continue

                        runtime = self.runtime(
                            candidate.model,
                            candidate.stage,
                            rid,
                        )

                        finish = now + runtime

                        # EASY rule:
                        # If candidate uses the resource reserved
                        # for the queue head, it must finish before
                        # the head's reservation starts.
                        #
                        # On another independent resource it cannot
                        # delay that reservation.
                        if (
                            rid == reserved_resource
                            and finish > reserved_start + 1e-12
                        ):
                            continue

                        candidate_options.append(
                            (
                                finish,
                                rid,
                                runtime,
                            )
                        )


                    if not candidate_options:
                        continue

                    # Among allowed backfill placements choose
                    # earliest finish.
                    candidate_options.sort(
                        key=lambda x: (
                            x[0],
                            x[1],
                        )
                    )

                    _, rid, _ = candidate_options[0]

                    start_task(
                        candidate_id,
                        rid,
                        now,
                    )

                    backfilled = True
                    dispatched_at_this_time = True
                    break


                if not backfilled:
                    break


            # --------------------------------------------------------
            # Finished?
            # --------------------------------------------------------

            if len(completed) == len(tasks):
                break


            # --------------------------------------------------------
            # Advance to next event.
            # --------------------------------------------------------

            next_times = []

            # Running job completions
            for job in running.values():

                if (
                    job is not None
                    and job.end_sec > now + 1e-12
                ):
                    next_times.append(job.end_sec)

            # Resources that are initially unavailable.
            for rid, t in initially_busy_until.items():

                if t > now + 1e-12:
                    next_times.append(t)

            # Future task releases are also simulation events.
            #
            # This matters when a predecessor was already running
            # at the disruption and survives outside the residual
            # DAG. Its successor cannot become ready until that
            # surviving work actually completes.
            for tid, release_time in task_release_times.items():

                if (
                    tid not in completed
                    and
                    tid not in started
                    and
                    float(release_time)
                    > now + 1e-12
                ):
                    next_times.append(
                        float(release_time)
                    )


            if not next_times:

                remaining = [
                    tid
                    for tid in tasks
                    if tid not in completed
                ]

                overhead = time.perf_counter() - wall_start

                return ScheduleResult(
                    scheduler=self.name,
                    success=False,
                    assignments=sorted(
                        assignments,
                        key=lambda a: (
                            a.start_sec,
                            a.task_id,
                        ),
                    ),
                    scheduling_overhead_sec=overhead,
                    makespan_sec=None,
                    infeasible_tasks=remaining,
                    notes="EASY simulation reached deadlock.",
                )


            now = min(next_times)


        # ------------------------------------------------------------
        # Final result
        # ------------------------------------------------------------

        assignments.sort(
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

        overhead = time.perf_counter() - wall_start

        return ScheduleResult(
            scheduler=self.name,
            success=(
                len(completed) == len(tasks)
                and len(assignments) == len(tasks)
            ),
            assignments=assignments,
            scheduling_overhead_sec=overhead,
            makespan_sec=makespan,
            infeasible_tasks=[],
            notes=(
                "Generalized EASY Backfill; deterministic FCFS; "
                "Experiment-1 task-resource runtimes; "
                "heterogeneous resource eligibility"
            ),
        )
