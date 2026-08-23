#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${1:-domain.pddl}"
PROBLEM="${2:-problem.pddl}"
PLAN="${3:-sas_plan}"

# Recommended simple/strong baseline for small classical problems:
./fast-downward.py "$DOMAIN" "$PROBLEM" --search "astar(lmcut())"

# Fast Downward usually writes its plan to sas_plan or sas_plan.N
if [[ -f "$PLAN" ]]; then
  echo "Plan file found: $PLAN"
else
  echo "Planner finished. Look for sas_plan or sas_plan.* in the current directory."
fi
