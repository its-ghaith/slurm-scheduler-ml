# RAPEC-v8 Sobol-ALE Sensitivity Analysis: rapec-v8-focused-sobol-20260802-165052

- Mode: `surrogate`
- Full100 traces: `9`
- Task types: `image_classification, object_detection, semantic_segmentation`
- Energy scope: training/epoch GPU energy
- Existing source bundles and previous results were not modified.

## Parameter ranking

| Rank | Parameter | Maximum ST | Mean ST |
|---:|---|---:|---:|
| 1 | `min_epochs_floor` | 0.6276 | 0.4988 |
| 2 | `noise_multiplier` | 0.5917 | 0.2576 |
| 3 | `trend_window` | 0.4119 | 0.0957 |
| 4 | `utility_quantile` | 0.3974 | 0.3508 |
| 5 | `minimum_quality_floor` | 0.1447 | 0.1246 |

## Surrogate validation

Sobol results from surrogate mode must be interpreted together with these holdout metrics.

| Output | R2 | MAE | Status |
|---|---:|---:|---|
| `overall__mean_energy_saving_fraction` | 0.9749 | 0.019877 | reliable |
| `overall__mean_quality_regret_pp` | 0.9683 | 0.258754 | reliable |
| `overall__mean_stop_epoch` | 0.9749 | 1.994395 | reliable |
| `overall__bayesian_veto_rate` | 0.5083 | 0.003383 | low-surrogate-fidelity |
| `overall__controller_runtime_ms_per_epoch` | 0.9093 | 0.255294 | reliable |

## Interpretation boundary

Sobol indices quantify variance within the declared parameter ranges. They are not causal effects outside those ranges. Offline replay assumes the Full100 trajectory up to the simulated stop epoch. Fresh Rancher runs remain necessary for final confirmation.
