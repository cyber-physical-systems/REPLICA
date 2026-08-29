#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


PROJECT = Path("/workspace/sc26_rebuttal")

PROFILE_PATH = (
    PROJECT
    / "experiment2/generated/execution_profiles.json"
)


RECOVERY_STAGES = [
    "update",
    "evaluate",
    "package",
    "deploy",
    "validate",
    "reactivate",
]


def load_execution_profiles(
    path: Path = PROFILE_PATH,
):
    raw = json.loads(
        path.read_text()
    )

    profiles = {}

    for row in raw:

        key = (
            row["model"],
            row["stage"],
            row["resource_id"],
        )

        profiles[key] = {
            "runtime_sec":
                float(
                    row["runtime_sec"]
                ),

            "cpu_peak_cores":
                float(
                    row.get(
                        "cpu_peak_cores",
                        0.0,
                    )
                ),

            "ram_peak_mb":
                float(
                    row.get(
                        "ram_peak_mb",
                        0.0,
                    )
                ),

            "gpu_mean_pct":
                float(
                    row.get(
                        "gpu_mean_pct",
                        0.0,
                    )
                ),

            "gpu_peak_pct":
                float(
                    row.get(
                        "gpu_peak_pct",
                        0.0,
                    )
                ),

            "vram_peak_mb":
                float(
                    row.get(
                        "vram_peak_mb",
                        0.0,
                    )
                ),
        }

    return profiles


def recovery_runtime_table(
    model_name: str,
    profiles,
):
    """
    Return all measured recovery-stage runtimes for one model.
    """

    table = defaultdict(
        dict
    )

    for (
        model,
        stage,
        resource_id,
    ), values in profiles.items():

        if model != model_name:
            continue

        if stage not in RECOVERY_STAGES:
            continue

        table[stage][
            resource_id
        ] = values

    return dict(table)


def minimum_stage_runtimes(
    model_name: str,
    profiles,
) -> Dict[str, float]:
    """
    Minimum measured runtime for each recovery stage.

    Useful only as a lower bound / heuristic.
    The scheduler itself should still choose a specific resource.
    """

    table = recovery_runtime_table(
        model_name,
        profiles,
    )

    out = {
        "quarantine": 0.0,
    }

    for stage in RECOVERY_STAGES:

        candidates = table.get(
            stage,
            {},
        )

        if not candidates:
            raise RuntimeError(
                f"No empirical profile for "
                f"{model_name}:{stage}"
            )

        out[stage] = min(
            values["runtime_sec"]
            for values
            in candidates.values()
        )

    return out


def feasible_resources_by_stage(
    model_name: str,
    profiles,
):
    table = recovery_runtime_table(
        model_name,
        profiles,
    )

    return {
        stage: sorted(
            resource_map
        )
        for stage, resource_map
        in table.items()
    }


if __name__ == "__main__":

    profiles = (
        load_execution_profiles()
    )

    for model in [
        "lstm",
        "yolo11n",
    ]:

        print()
        print("=" * 100)
        print(
            f"EMPIRICAL RECOVERY PROFILE: "
            f"{model}"
        )
        print("=" * 100)

        table = recovery_runtime_table(
            model,
            profiles,
        )

        for stage in (
            RECOVERY_STAGES
        ):

            print()
            print(
                f"{stage.upper()}"
            )

            for rid, values in sorted(
                table.get(
                    stage,
                    {}
                ).items()
            ):

                print(
                    f"  {rid:10s} "
                    f"runtime="
                    f"{values['runtime_sec']:9.3f}s "
                    f"RAM="
                    f"{values['ram_peak_mb']:9.1f}MB "
                    f"VRAM="
                    f"{values['vram_peak_mb']:9.1f}MB"
                )

        print()
        print(
            "Minimum stage runtimes:"
        )

        for stage, runtime in (
            minimum_stage_runtimes(
                model,
                profiles,
            ).items()
        ):

            print(
                f"  {stage:12s} "
                f"{runtime:9.3f}s"
            )
