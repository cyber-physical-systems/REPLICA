#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime

import paho.mqtt.client as mqtt

MQTT_BROKER = "0.0.0.0"
MQTT_PORT = 1883
REQUEST_TOPIC = "rais/recovery/request"
STATUS_TOPIC = "rais/recovery/status"

RUN_ID = "run_H"
CONFIG = "rais_pipeline_prune_config_attack.yaml"
PLANNER_BACKEND = "telemetry_opt"

is_busy = False
lock = threading.Lock()

def publish_status(client: mqtt.Client, status: str, action: str, details: dict | None = None) -> None:
    payload = {
        "status": status,
        "run_id": RUN_ID,
        "action": action,
        "details": details or {},
        "timestamp": datetime.now().isoformat(),
    }
    client.publish(STATUS_TOPIC, json.dumps(payload), qos=1)

def run_orchestrator(client: mqtt.Client, request: dict) -> None:
    global is_busy
    try:
        event = request.get("event", "unknown")

        if event == "request_clean_model":
            action = "deploy_clean_model"
        elif event == "request_retraining":
            action = "retrain_pipeline"
        else:
            action = "unknown"

        publish_status(client, "accepted", action, {"event": event})

        cmd = [
            "python",
            "pddl_orchestrator.py",
            "--planner-backend",
            PLANNER_BACKEND,
            "--fresh-run",
            "--run-id",
            RUN_ID,
            "--config",
            CONFIG,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            publish_status(client, "completed", action, {"event": event})
        else:
            publish_status(
                client,
                "failed",
                action,
                {
                    "event": event,
                    "returncode": result.returncode,
                    "stderr_tail": result.stderr[-1000:],
                },
            )

    finally:
        with lock:
            is_busy = False

def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected with rc={rc}")
    client.subscribe(REQUEST_TOPIC, qos=1)

def on_message(client, userdata, msg):
    global is_busy

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print(f"[warn] bad JSON payload: {e}")
        return

    print(f"[mqtt] received request on {msg.topic}: {payload}")

    with lock:
        if is_busy:
            print("[listener] orchestrator busy; ignoring new request")
            publish_status(client, "ignored_busy", "none", payload)
            return
        is_busy = True

    thread = threading.Thread(target=run_orchestrator, args=(client, payload), daemon=True)
    thread.start()

def main():
    client = mqtt.Client(client_id="orchestrator_listener")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("127.0.0.1", MQTT_PORT, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()