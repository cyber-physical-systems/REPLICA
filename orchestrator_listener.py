import time
from pathlib import Path
import subprocess

REQUEST_FILE = Path("/tmp/rais_recovery_request.json")

while True:
    if REQUEST_FILE.exists():
        print("[listener] recovery request detected")

        subprocess.run([
            "python", "pddl_orchestrator.py",
            "--planner-backend", "telemetry_opt",
            "--run-id", "run_H",
            "--config", "rais_pipeline_prune_config_attack.yaml"
        ])

        print("[listener] orchestrator run completed")

    time.sleep(5)