# RAPEC-v8 Sobol-ALE Sensitivity Analysis: codex-focused-smoke-20260802

- Mode: `surrogate`
- Full100 traces: `9`
- Task types: `image_classification, object_detection, semantic_segmentation`
- Energy scope: training/epoch GPU energy
- Existing source bundles and previous results were not modified.

## Parameter ranking

| Rank | Parameter | Maximum ST | Mean ST |
|---:|---|---:|---:|
| 1 | `min_epochs_floor` | 0.5869 | 0.1941 |
| 2 | `utility_quantile` | 0.4711 | 0.4023 |
| 3 | `minimum_quality_floor` | 0.2986 | 0.2515 |
| 4 | `trend_window` | 0.1574 | 0.0338 |
| 5 | `noise_multiplier` | 0.0102 | 0.0071 |

## Surrogate validation

Sobol results from surrogate mode must be interpreted together with these holdout metrics.

| Output | R2 | MAE | Status |
|---|---:|---:|---|
| `overall__mean_energy_saving_fraction` | 0.4853 | 0.117903 | low-surrogate-fidelity |
| `overall__mean_quality_regret_pp` | 0.5539 | 1.284295 | low-surrogate-fidelity |
| `overall__mean_stop_epoch` | 0.4843 | 11.830874 | low-surrogate-fidelity |
| `overall__bayesian_veto_rate` | -0.0183 | 0.003100 | low-surrogate-fidelity |
| `overall__controller_runtime_ms_per_epoch` | 0.6180 | 0.609549 | low-surrogate-fidelity |

## Interpretation boundary

Sobol indices quantify variance within the declared parameter ranges. They are not causal effects outside those ranges. Offline replay assumes the Full100 trajectory up to the simulated stop epoch. Fresh Rancher runs remain necessary for final confirmation.
