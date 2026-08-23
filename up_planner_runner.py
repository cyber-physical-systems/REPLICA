from __future__ import annotations

from typing import Dict, Tuple, Any

from unified_planning.shortcuts import OneshotPlanner, Problem
import unified_planning as up


def solve_problem(problem: Problem):
    try:
        up.shortcuts.get_environment().credits_stream = None
    except Exception:
        pass

    with OneshotPlanner(name="enhsp-opt") as planner:
        result = planner.solve(problem)
    return result


def extract_assignments(plan) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    if plan is None:
        return assignments

    for action_instance in plan.actions:
        name = action_instance.action.name
        if name.startswith("assign__") and "__to__" in name:
            _, rest = name.split("assign__", 1)
            stage, host = rest.split("__to__", 1)
            assignments[stage] = host

    return assignments


def plan_assignments(problem, engine_name: str = "tamer"):
    with OneshotPlanner(name=engine_name) as planner:
        result = planner.solve(problem)

    print(f"[up] engine: {engine_name}")
    print(f"[up] solve status: {result.status}")

    assignments = {}
    if result.plan is not None:
        print(f"[up] plan action count: {len(result.plan.actions)}")
        for action_instance in result.plan.actions:
            aname = action_instance.action.name
            print(f"[up] action: {aname}")

            if aname.startswith("assign__") and "__to__" in aname:
                stage, host = aname[len("assign__"):].split("__to__", 1)
                assignments[stage] = host

        print(f"[up] extracted assignments: {assignments}")
    else:
        print("[up] no plan returned by planner")

    return result, assignments