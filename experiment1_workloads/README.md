# REPLICA Experiment 1 — HVAC Workloads

These scripts extract the three HVAC model workloads from the original 09A
notebook without rerunning the entire CPS analysis notebook.

Source artifacts:

```text
/workspace/sc26_rebuttal/outputs/09A_ai_normal_dynamics/
```

Original settings preserved from 09A:

- seed = 42
- RF subsample = 600,000
- XGBoost subsample = 600,000
- LSTM window = 8
- LSTM subsample = 600,000
- LSTM epochs = 25
- LSTM batch size = 512
- LSTM learning rate = 1e-3
- LSTM patience = 5

## Important Experiment 1 decision

`update` is currently defined as **retraining the model using the original 09A
training procedure**. This matches the professor's request to characterize
training/update time and is comparable across A100 / RTX5090 / RTX4080.

We can add a separate fine-tuning service mode later, but should not mix it with
the nominal training characterization.

## Files

- `hvac_common.py` — shared paths, scalers, sampling, metrics
- `hvac_lstm_update.py` — LSTM retraining
- `hvac_rf_update.py` — Random Forest retraining
- `hvac_xgb_update.py` — XGBoost retraining
- `hvac_evaluate.py` — model-specific evaluation

Copy the directory to:

```text
/workspace/sc26_rebuttal/experiment1_workloads/
```

## First: quick validation runs

Do NOT immediately launch the full 600k / 25-epoch jobs.

Validate each workload using smaller budgets first.

### Random Forest quick test

```bash
cd /workspace/sc26_rebuttal

mkdir -p outputs/experiment1/candidates outputs/experiment1/metrics

python experiment1_profiler/profile_service_task.py \
  --model random_forest \
  --task update \
  --host A100 \
  --gpu-index 0 \
  --artifact outputs/experiment1/candidates/rf_test.pkl \
  --metric-file outputs/experiment1/metrics/rf_test.json \
  --output-dir outputs/experiment1 \
  --run-id rf_update_a100_test \
  -- \
  python experiment1_workloads/hvac_rf_update.py \
    --samples 10000 \
    --artifact outputs/experiment1/candidates/rf_test.pkl \
    --metric-file outputs/experiment1/metrics/rf_test.json
```

### XGBoost quick test

```bash
python experiment1_profiler/profile_service_task.py \
  --model xgboost \
  --task update \
  --host A100 \
  --gpu-index 0 \
  --artifact outputs/experiment1/candidates/xgb_test.pkl \
  --metric-file outputs/experiment1/metrics/xgb_test.json \
  --output-dir outputs/experiment1 \
  --run-id xgb_update_a100_test \
  -- \
  python experiment1_workloads/hvac_xgb_update.py \
    --samples 10000 \
    --artifact outputs/experiment1/candidates/xgb_test.pkl \
    --metric-file outputs/experiment1/metrics/xgb_test.json
```

### LSTM quick test

```bash
python experiment1_profiler/profile_service_task.py \
  --model lstm \
  --task update \
  --host A100 \
  --gpu-index 0 \
  --artifact outputs/experiment1/candidates/lstm_test.pt \
  --metric-file outputs/experiment1/metrics/lstm_test.json \
  --output-dir outputs/experiment1 \
  --run-id lstm_update_a100_test \
  -- \
  python experiment1_workloads/hvac_lstm_update.py \
    --samples 10000 \
    --epochs 2 \
    --artifact outputs/experiment1/candidates/lstm_test.pt \
    --metric-file outputs/experiment1/metrics/lstm_test.json
```

## Then evaluate the candidates

Example:

```bash
python experiment1_profiler/profile_service_task.py \
  --model lstm \
  --task evaluate \
  --host A100 \
  --gpu-index 0 \
  --metric-file outputs/experiment1/metrics/lstm_eval_test.json \
  --output-dir outputs/experiment1 \
  --run-id lstm_eval_a100_test \
  -- \
  python experiment1_workloads/hvac_evaluate.py \
    --model-type lstm \
    --model-path outputs/experiment1/candidates/lstm_test.pt \
    --samples 10000 \
    --metric-file outputs/experiment1/metrics/lstm_eval_test.json
```

For profiler `--artifact` during evaluation, omit it; evaluation creates no new
model artifact.

## Full nominal runs

After all quick tests pass, use the original budgets:

- RF: 600000 samples
- XGBoost: 600000 samples
- LSTM: 600000 sampled sequences, up to 25 epochs

Run one warm-up first and do not include it in the measured repetitions.
Then collect 5 measured repetitions per compatible hardware configuration.
