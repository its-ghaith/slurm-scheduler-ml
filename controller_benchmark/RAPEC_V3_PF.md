# RAPEC-v3-PF

RAPEC-v3-PF is a profile-free ablation of RAPEC-v3. It inherits the original
prediction, utility, uncertainty, quality guard, learning-state, and stopping
logic. Only the task- and scenario-dependent minimum-epoch rule is replaced.

## Task-independent contract

The controller consumes only the current run:

- normalized validation quality `quality_score` in `[0, 1]`;
- per-epoch training energy in Wh;
- duration and optional hardware/model telemetry already supported by v3.

It never selects parameters from task type, dataset name, model name, or
scratch/pretrained metadata. Equivalent quality, energy, and telemetry traces
therefore produce equivalent decisions regardless of their labels.

## Profile-free warm-up

The structural history floor is derived from the v3 predictor itself:

```text
max(
  min_epochs_floor,
  min_fit_points + horizon_epochs + knn_neighbors,
  trend_window + longest_confirmation_horizon
)
```

With the energy-oriented packaged configuration this is epoch 24. The gate
compares best-so-far quality gain over only the latest 5 epochs with robust MAD
noise. The dimensionless progress-to-noise ratio must enter the lower 40% of
its current-run history once before the original v3 stopping logic is enabled.
A no-learning fallback releases the gate at 1.5 times the structural floor
(epoch 36 for a 100-epoch run). The shared RAPEC-v3 predictor, uncertainty,
utility, quality guard, patience, and stopping parameters remain unchanged.

## Files

- Controller: `controller_benchmark/controllers/rapec_v3_pf.py`
- Parameters: `controller_benchmark/config/rapec-v3-pf-controller.json`
- Nine-dataset manifest: `controller_benchmark/config/benchmark-v3-nine-dataset-v3-pf.json`
- Tests: `controller_benchmark/tests/test_rapec_v3_pf.py`
- Submit script: `controller_benchmark/scripts/submit-rapec-v3-pf-controller-benchmark-async.ps1`

## Validation workflow

Plan without submitting jobs:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v3-pf-controller-benchmark-async.ps1 `
  -RequireAllStages `
  -PlanOnly
```

Submit the nine candidate runs after reviewing the plan:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v3-pf-controller-benchmark-async.ps1 `
  -RequireAllStages
```

The Rancher orchestrator continues independently after submission.

## Current offline limitation

Replay on the nine stored Full100 traces confirms profile invariance but also
shows that the unchanged v3 stopping core can still treat temporary plateaus as
convergence. The current implementation is therefore a controlled scientific
ablation, not yet a quality-safe replacement for v3 in production. No Rancher
training was submitted as part of its implementation.
