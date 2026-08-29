# REPLICA Experiment 1 — Frozen HVAC Inputs

This revision separates one-time workload preparation from timed model execution.

## Why

The first LSTM validation run showed that reading the ~58M-row 09A training
split and materializing sequences dominated the wall-clock runtime. That is not
the hardware training cost we want to compare across A100, RTX 5090, and RTX
4080.

The new workflow is:

```text
09A train/test splits
        |
        |  ONE TIME, not timed
        v
Frozen Experiment 1 workload inputs
        |
        +--> RF update
        +--> XGBoost update
        +--> LSTM update
```

Every hardware platform receives identical `.npz` files. The manifest stores
SHA-256 hashes so we can verify that the workloads are bit-for-bit identical.

## Install

Extract this package in:

```text
/workspace/sc26_rebuttal/
```

It creates/updates:

```text
experiment1_workloads/
```

## Step 1 — prepare frozen inputs ONCE

```bash
cd /workspace/sc26_rebuttal

python experiment1_workloads/prepare_hvac_workload_inputs.py
```

Default outputs:

```text
outputs/experiment1/workload_inputs/
  hvac_tabular_train_600000.npz
  hvac_tabular_eval_100000.npz
  hvac_lstm_train_600000.npz
  hvac_lstm_eval_100000.npz
  workload_input_manifest.json
```

This step may take time because it scans the large 09A split. Do it once.

## Step 2 — inspect files and manifest

```bash
ls -lh outputs/experiment1/workload_inputs
cat outputs/experiment1/workload_inputs/workload_input_manifest.json
```

## Step 3 — quick post-freeze LSTM validation

For a fast test, the default frozen file contains 600k sequences but use only 2
epochs:

```bash
python experiment1_profiler/profile_service_task.py \
  --model lstm \
  --task update \
  --host A100 \
  --gpu-index 0 \
  --artifact outputs/experiment1/candidates/lstm_frozen_test.pt \
  --metric-file outputs/experiment1/metrics/lstm_frozen_test.json \
  --output-dir outputs/experiment1 \
  --run-id lstm_update_a100_frozen_test \
  -- \
  python experiment1_workloads/hvac_lstm_update.py \
    --epochs 2 \
    --artifact outputs/experiment1/candidates/lstm_frozen_test.pt \
    --metric-file outputs/experiment1/metrics/lstm_frozen_test.json
```

This now times loading the fixed `.npz`, training, and saving the model. It does
NOT scan the giant 09A Parquet file or reconstruct sequences.

## Step 4 — full measured Experiment 1

After validation:

- one unrecorded warm-up;
- 5 recorded runs per model/resource;
- RF/XGB use the same frozen 600k tabular sample;
- LSTM uses the same frozen 600k sequence sample;
- evaluation uses the same frozen 100k evaluation samples.

Do not regenerate frozen inputs between A100/5090/4080 runs.
