import subprocess

DEVICES = {
    "ncps": {
        "host": "NCPS@100.114.8.110",
        "remote_dir": "/home/NCPS/sc26",
        "python": "/home/NCPS/miniconda3/bin/python3",
    },
    "gpu": {
        "host": "cpslab@100.77.214.32",
        "remote_dir": "/home/cpslab/Documents/rais_venv/Security-aware-Model-Compression/sc26",
        "python": "/home/cpslab/Documents/rais_venv/bin/python",
    },
    "jetson": {
        "host": "widney@100.97.5.96",
        "remote_dir": "/home/widney/Documents/rais_venv/Security-aware-Model-Compression/sc26",
        "python": "/home/widney/Documents/rais_venv/bin/python",
    },
    "pi5a": {
        "host": "widney@100.127.157.32",
        "remote_dir": "/home/widney/Documents/rais_venv/Security-aware-Model-Compression/sc26",
        "python": "/home/widney/rais_project/bin/python",
    },
    "pi5b": {
        "host": "widneypi2@100.118.46.99",
        "remote_dir": "/home/widneypi2/Documents/rais_venv/Security-aware-Model-Compression/sc26",
        "python": "/home/widneypi2/rais_project/bin/python",
    },
    "pi5d": {
        "host": "pi5d@100.73.148.79",
        "remote_dir": "/home/pi5d/Documents/rais_venv/Security-aware-Model-Compression/sc26",
        "python": "/home/pi5d/rais_project/bin/python3",
    },
}

LOCAL_PROJECT_DIR = "."  # current repo root


def run(cmd):
    print(f"\n[cmd] {cmd}")
    return subprocess.run(cmd, shell=True)


def sync_code(device_name, cfg):
    host = cfg["host"]
    remote_dir = cfg["remote_dir"]

    print(f"\n[SYNC] {device_name} → {host}")

    cmd = (
        f'rsync -av --delete '
        f'--exclude "shared_runs/" '
        f'--exclude "__pycache__/" '
        f'{LOCAL_PROJECT_DIR}/ {host}:{remote_dir}/'
    )
    run(cmd)


def verify_stage_suffix(device_name, cfg):
    host = cfg["host"]
    remote_dir = cfg["remote_dir"]
    py = cfg["python"]

    print(f"[VERIFY] {device_name}")

    cmd = (
        f'ssh {host} '
        f'"cd {remote_dir} && {py} rais_pipeline_with_yaml.py --help | grep stage-suffix"'
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if "stage-suffix" in result.stdout:
        print(f"[PASS] {device_name} supports --stage-suffix")
    else:
        print(f"[FAIL] {device_name} missing --stage-suffix")
        print(result.stdout)
        print(result.stderr)


def main():
    for name, cfg in DEVICES.items():
        sync_code(name, cfg)

    print("\n================ VERIFY ================")

    for name, cfg in DEVICES.items():
        verify_stage_suffix(name, cfg)


if __name__ == "__main__":
    main()