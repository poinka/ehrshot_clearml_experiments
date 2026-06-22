# EHRSHOT multiseed stability experiments

These scripts run the three selected model families with several random seeds and then compare them on high-repeat diagnosis subgroups.

## Selected models

For `guo_readmission`:

- Tabular: `HistGradientBoosting`
- Code-only sequence: `compressed_dedup + RETAIN_lite`
- Numeric-aware sequence: `compressed_dedup + RETAIN_lite_numeric`

For `guo_icu`:

- Tabular: `RandomForest_balanced`
- Code-only sequence: `compressed_condition_era + LSTM_1L`
- Numeric-aware sequence: `compressed_hybrid + GRU_2L_numeric`

## Required inputs

Expected folders/files in the working directory:

```text
EHRSHOT_MEDS/                         # only needed if you rebuild cached data elsewhere
ehrshot_baseline_cache/               # tabular feature cache from classical notebook
ehrshot_sequence_datasets/            # sequence datasets from sequence notebook
ehrshot_copy_forwarding_audit/         # audit outputs for high-repeat subgroup
```

The scripts use cached features / sequence datasets. They do not rebuild EHRSHOT-MEDS from raw events.

## Run locally / inside ClearML agent

```bash
python train_tabular_multiseed.py \
  --seeds 42,43,44,45,46 \
  --enable-clearml \
  --clearml-task-name tabular_multiseed_stability

python train_sequence_multiseed.py \
  --seeds 42,43,44,45,46 \
  --enable-clearml \
  --clearml-task-name sequence_multiseed_stability

python train_numeric_sequence_multiseed.py \
  --seeds 42,43,44,45,46 \
  --enable-clearml \
  --clearml-task-name numeric_sequence_multiseed_stability
```

Then compare all outputs:

```bash
python compare_multiseed_high_repeat.py
```

## Run on ClearML agent

`--enable-clearml` alone only logs the local run.  
To actually run training on a ClearML agent, use `--execute-remotely` and provide a queue.

### Tabular baseline

Tabular models do not need GPU. Prefer a CPU queue if available:

```bash
python train_tabular_multiseed.py \
  --seeds 42,43,44,45,46 \
  --enable-clearml \
  --execute-remotely \
  --clearml-queue cpu \
  --clearml-task-name tabular_multiseed_stability
```

If there is no CPU queue, use the available GPU queue, but the job will still run sklearn on CPU:

```bash
python train_tabular_multiseed.py \
  --seeds 42,43,44,45,46 \
  --enable-clearml \
  --execute-remotely \
  --clearml-queue gpu_40 \
  --clearml-task-name tabular_multiseed_stability
```

### Code-only sequence baseline
```bash
python train_sequence_multiseed.py \
  --seeds 42,43,44,45,46 \
  --device auto \
  --enable-clearml \
  --execute-remotely \
  --clearml-queue gpu_40 \
  --clearml-task-name sequence_multiseed_stability
```

### Numeric-aware sequence baseline
```bash
python train_numeric_sequence_multiseed.py \
  --seeds 42,43,44,45,46 \
  --device auto \
  --enable-clearml \
  --execute-remotely \
  --clearml-queue gpu_40 \
  --clearml-task-name numeric_sequence_multiseed_stability
```

### Compare all model families
```bash
python compare_multiseed_high_repeat.py \
  --enable-clearml \
  --execute-remotely \
  --clearml-queue cpu \
  --clearml-task-name multiseed_high_repeat_comparison
```

If the comparison is run locally, omit --execute-remotely.


#### Device policy for sequence scripts:
- `--device auto` uses CUDA if available, otherwise CPU.
- MPS is not used by default because it was slow for this workload.
- To force local MPS explicitly, pass `--device mps`.

## Main outputs

Training scripts save:

- `*_multiseed_results.csv`
- `*_multiseed_heldout_predictions.csv`
- `*_multiseed_topk.csv`
- per-seed histories and predictions

Comparison script saves:

- `single_seed_subgroup_metrics.csv` — metrics per seed and subgroup
- `seed_metric_summary.csv` — mean/std/min/max metrics across seeds
- `prediction_stability_by_example.csv` — mean/std of risk per prediction example
- `prediction_stability_summary.csv` — aggregate prediction STD by model and subgroup
- `ensemble_mean_predictions.csv` — mean risk over seeds
- `ensemble_subgroup_performance.csv` — metrics for seed-ensemble predictions
- `ensemble_paired_bootstrap_delta.csv` — paired bootstrap deltas between model families
- `huly_paired_auprc_delta_table.csv` — compact Huly table for AUPRC deltas

## Interpretation

- Seed STD answers: how sensitive predictions are to random initialization / training stochasticity.
- Ensemble mean predictions answer: what quality we get after averaging risks over seeds.
- Paired bootstrap on ensemble predictions answers: whether differences between model families are stable on the same held-out examples.

For `guo_icu` high-repeat group, remember that there are only 5 positive events. Bootstrap will likely produce wide confidence intervals.
