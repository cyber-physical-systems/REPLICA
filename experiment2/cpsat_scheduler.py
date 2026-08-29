from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model


@dataclass
class CPSATAssignment:
    task_id: str
    model: str
    stage: str
    resource_id: str
    start_sec: float
    end_sec: float
    runtime_sec: float


class CPSATScheduler:
    """
    CP-SAT baseline for Experiment 2.

    Inputs:
      tasks:
        objects/dicts containing:
          task_id, model, stage

      profiles:
        empirical Experiment-1 execution profiles containing:
          model, stage, resource_id, runtime_sec

      feasible:
        optional set of allowed (model, stage, resource_id) tuples.

    Objective:
      minimize workflow makespan.

    Constraints:
      - exactly one resource per task
      - only empirically feasible assignments
      - one task at a time per resource
      - workflow stage precedence
    """

    STAGE_ORDER = [
        "update",
        "evaluate",
        "package",
        "deploy",
        "validate",
        "reactivate",
    ]

    # CP-SAT uses integers. Millisecond resolution preserves our measured
    # runtimes sufficiently well for this experiment.
    SCALE = 1000

    def __init__(self, time_limit_sec: float = 60.0):
        self.time_limit_sec = time_limit_sec

    @staticmethod
    def _get(obj: Any, key: str):
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    def schedule(
        self,
        tasks,
        profiles,
        feasible=None,
        **kwargs,
    ):
        wall_start = time.perf_counter()

        # ---------------------------------------------------------
        # Optional runtime state
        #
        # Static behavior is unchanged when these kwargs are
        # omitted.
        #
        # busy-until values use the absolute execution clock.
        # CP-SAT converts them to offsets from state_time_sec.
        # ---------------------------------------------------------

        state_time_sec = float(
            kwargs.get(
                "state_time_sec",
                0.0,
            )
        )

        resource_busy_until = dict(
            kwargs.get(
                "resource_busy_until",
                {},
            )
            or {}
        )

        task_release_times = dict(
            kwargs.get(
                "task_release_times",
                {},
            )
            or {}
        )

        # ---------------------------------------------------------
        # Normalize tasks
        # ---------------------------------------------------------
        normalized_tasks = []

        for t in tasks:
            model_name = self._get(t, "model")
            stage = self._get(t, "stage")

            try:
                task_id = self._get(t, "task_id")
            except (KeyError, AttributeError):
                task_id = f"{model_name}:{stage}"

            normalized_tasks.append(
                {
                    "task_id": str(task_id),
                    "model": str(model_name),
                    "stage": str(stage),
                }
            )

        # ---------------------------------------------------------
        # Build empirical runtime lookup
        # ---------------------------------------------------------
        runtime = {}

        for p in profiles:
            model_name = self._get(p, "model")
            stage = self._get(p, "stage")
            resource = self._get(p, "resource_id")
            runtime_sec = float(self._get(p, "runtime_sec"))

            runtime[
                (
                    str(model_name),
                    str(stage),
                    str(resource),
                )
            ] = runtime_sec

        resources = sorted(
            {
                resource
                for _, _, resource in runtime.keys()
            }
        )

        if not resources:
            raise ValueError("No resources found in empirical profiles.")

        # ---------------------------------------------------------
        # Initial resource reservations
        #
        # CP-SAT's recovery solve starts at local t=0.
        #
        # Example:
        #
        # event = 199.948
        # resource busy until = 212.015
        #
        # local busy offset = 12.067 s
        # ---------------------------------------------------------

        busy_offset_sec = {
            resource: max(
                0.0,
                float(
                    resource_busy_until.get(
                        resource,
                        state_time_sec,
                    )
                )
                - state_time_sec,
            )
            for resource in resources
        }

        busy_offset_ms = {
            resource: max(
                0,
                int(
                    round(
                        busy_offset_sec[
                            resource
                        ]
                        * self.SCALE
                    )
                ),
            )
            for resource in resources
        }

        # ---------------------------------------------------------
        # Task release offsets
        #
        # Input release times are absolute world times.
        # CP-SAT's recovery solve uses local t=0.
        # ---------------------------------------------------------

        release_offset_ms = {}

        for task in normalized_tasks:

            tid = task["task_id"]

            release_absolute = float(
                task_release_times.get(
                    tid,
                    state_time_sec,
                )
            )

            release_offset_ms[tid] = max(
                0,
                int(
                    round(
                        max(
                            0.0,
                            release_absolute
                            - state_time_sec,
                        )
                        * self.SCALE
                    )
                ),
            )

        # ---------------------------------------------------------
        # Determine allowed placements
        # ---------------------------------------------------------
        allowed = {}

        for task in normalized_tasks:
            tid = task["task_id"]
            model_name = task["model"]
            stage = task["stage"]

            choices = []

            for resource in resources:
                key = (model_name, stage, resource)

                if key not in runtime:
                    continue

                if feasible is not None and key not in feasible:
                    continue

                choices.append(resource)

            if not choices:
                raise ValueError(
                    f"No feasible resource for task {tid}"
                )

            allowed[tid] = choices

        # ---------------------------------------------------------
        # Integer time horizon
        # ---------------------------------------------------------
        duration_ms = {}

        for task in normalized_tasks:
            tid = task["task_id"]

            for resource in allowed[tid]:
                key = (
                    task["model"],
                    task["stage"],
                    resource,
                )

                duration_ms[(tid, resource)] = max(
                    1,
                    int(round(runtime[key] * self.SCALE)),
                )

        # Safe upper bound:
        #
        # longest initial reservation
        # +
        # sum of worst placement for every remaining task.
        horizon = (
            max(
                max(
                    busy_offset_ms.values(),
                    default=0,
                ),
                max(
                    release_offset_ms.values(),
                    default=0,
                ),
            )
            +
            sum(
                max(
                    duration_ms[
                        (
                            t["task_id"],
                            r,
                        )
                    ]
                    for r in allowed[
                        t["task_id"]
                    ]
                )
                for t in normalized_tasks
            )
        )

        model = cp_model.CpModel()

        # ---------------------------------------------------------
        # Variables
        # ---------------------------------------------------------
        start_vars = {}
        end_vars = {}

        presence = {}
        optional_intervals = {}

        for task in normalized_tasks:
            tid = task["task_id"]

            start_vars[tid] = model.new_int_var(
                0,
                horizon,
                f"start_{tid}",
            )

            end_vars[tid] = model.new_int_var(
                0,
                horizon,
                f"end_{tid}",
            )

            model.add(
                start_vars[tid]
                >=
                release_offset_ms[
                    tid
                ]
            )

            task_presence = []

            for resource in allowed[tid]:
                dur = duration_ms[(tid, resource)]

                x = model.new_bool_var(
                    f"x_{tid}_{resource}"
                )

                presence[(tid, resource)] = x
                task_presence.append(x)

                # A newly scheduled task cannot start on a
                # resource until surviving pre-event work has
                # released that resource.
                model.add(
                    start_vars[tid]
                    >=
                    busy_offset_ms[
                        resource
                    ]
                ).only_enforce_if(
                    x
                )

                interval = model.new_optional_interval_var(
                    start_vars[tid],
                    dur,
                    end_vars[tid],
                    x,
                    f"interval_{tid}_{resource}",
                )

                optional_intervals[(tid, resource)] = interval

            # Exactly one resource selected.
            model.add_exactly_one(task_presence)

        # ---------------------------------------------------------
        # Resource capacity
        # ---------------------------------------------------------
        for resource in resources:
            intervals = []

            for task in normalized_tasks:
                tid = task["task_id"]

                if (tid, resource) in optional_intervals:
                    intervals.append(
                        optional_intervals[(tid, resource)]
                    )

            if intervals:
                model.add_no_overlap(intervals)

        # ---------------------------------------------------------
        # Workflow precedence
        # ---------------------------------------------------------
        by_model = defaultdict(dict)

        for task in normalized_tasks:
            by_model[task["model"]][task["stage"]] = task["task_id"]

        for model_name, stages in by_model.items():

            for s1, s2 in zip(
                self.STAGE_ORDER,
                self.STAGE_ORDER[1:],
            ):
                if s1 not in stages or s2 not in stages:
                    continue

                first = stages[s1]
                second = stages[s2]

                model.add(
                    start_vars[second] >= end_vars[first]
                )

        # ---------------------------------------------------------
        # Makespan objective
        # ---------------------------------------------------------
        makespan = model.new_int_var(
            0,
            horizon,
            "makespan",
        )

        model.add_max_equality(
            makespan,
            [
                end_vars[t["task_id"]]
                for t in normalized_tasks
            ],
        )

        model.minimize(makespan)

        # ---------------------------------------------------------
        # Solve
        # ---------------------------------------------------------
        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = self.time_limit_sec
        solver.parameters.num_search_workers = 8

        status = solver.solve(model)

        scheduling_overhead = time.perf_counter() - wall_start

        if status not in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ):
            raise RuntimeError(
                "CP-SAT could not find a feasible schedule. "
                f"Status={solver.status_name(status)}"
            )

        # ---------------------------------------------------------
        # Extract schedule
        # ---------------------------------------------------------
        assignments = []

        task_lookup = {
            t["task_id"]: t
            for t in normalized_tasks
        }

        for tid, task in task_lookup.items():

            selected_resource = None

            for resource in allowed[tid]:
                if solver.value(
                    presence[(tid, resource)]
                ):
                    selected_resource = resource
                    break

            if selected_resource is None:
                raise RuntimeError(
                    f"CP-SAT produced no placement for {tid}"
                )

            start_sec = (
                solver.value(start_vars[tid]) / self.SCALE
            )

            end_sec = (
                solver.value(end_vars[tid]) / self.SCALE
            )

            assignments.append(
                {
                    "task_id": tid,
                    "model": task["model"],
                    "stage": task["stage"],
                    "resource_id": selected_resource,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "runtime_sec": (
                        end_sec - start_sec
                    ),
                }
            )

        assignments.sort(
            key=lambda x: (
                x["start_sec"],
                x["resource_id"],
                x["task_id"],
            )
        )

        result = {
            "scheduler": "cp_sat",
            "success": True,
            "solver_status": solver.status_name(status),
            "makespan_sec": (
                solver.value(makespan) / self.SCALE
            ),
            "scheduling_overhead_sec": scheduling_overhead,
            "num_tasks": len(assignments),
            "assignments": assignments,
        }

        return result
