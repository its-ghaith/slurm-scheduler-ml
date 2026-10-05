# RAPEC-v8 Sobol-ALE Sensitivity Analysis: rapec-v8-sobol-20260802-151745

- Mode: `surrogate`
- Full100 traces: `9`
- Task types: `image_classification, object_detection, semantic_segmentation`
- Energy scope: training/epoch GPU energy
- Existing source bundles and previous results were not modified.

## Parameter ranking

| Rank | Parameter | Maximum ST | Mean ST |
|---:|---|---:|---:|
| 1 | `min_epochs_floor` | 0.6008 | 0.5429 |
| 2 | `noise_multiplier` | 0.3512 | 0.3271 |
| 3 | `utility_quantile` | 0.3255 | 0.2960 |
| 4 | `minimum_quality_floor` | 0.0885 | 0.0829 |
| 5 | `trend_window` | 0.0462 | 0.0392 |
| 6 | `bayesian_window` | 0.0259 | 0.0111 |
| 7 | `horizon_epochs` | 0.0247 | 0.0191 |
| 8 | `remaining_room_fraction` | 0.0132 | 0.0054 |
| 9 | `patience` | 0.0103 | 0.0051 |
| 10 | `bayesian_samples` | 0.0082 | 0.0020 |
| 11 | `bayesian_confirmation_patience` | 0.0051 | 0.0026 |
| 12 | `max_probability_gain_gt_threshold` | 0.0029 | 0.0013 |
| 13 | `bayesian_probability_limit` | 0.0025 | 0.0016 |
| 14 | `min_fit_points` | 0.0024 | 0.0018 |
| 15 | `min_meaningful_gain` | 0.0005 | 0.0004 |
| 16 | `knn_neighbors` | 0.0005 | 0.0003 |
| 17 | `late_stage_relaxation` | 0.0004 | 0.0003 |
| 18 | `bayesian_prior_strength` | 0.0004 | 0.0004 |
| 19 | `bootstrap_samples` | 0.0004 | 0.0002 |
| 20 | `recent_gain_fraction` | 0.0004 | 0.0002 |
| 21 | `bayesian_min_observations` | 0.0004 | 0.0003 |
| 22 | `max_uncertainty_to_gain_ratio` | 0.0003 | 0.0002 |
| 23 | `bayesian_prior_alpha` | 0.0003 | 0.0002 |

## Surrogate validation

Sobol results from surrogate mode must be interpreted together with these holdout metrics.

| Output | R2 | MAE | Status |
|---|---:|---:|---|
| `overall__mean_energy_saving_fraction` | 0.7168 | 0.054138 | low-surrogate-fidelity |
| `overall__mean_quality_regret_pp` | 0.6944 | 0.587944 | low-surrogate-fidelity |
| `overall__mean_stop_epoch` | 0.7166 | 5.438632 | low-surrogate-fidelity |
| `overall__bayesian_veto_rate` | 0.3731 | 0.019284 | low-surrogate-fidelity |
| `overall__controller_runtime_ms_per_epoch` | 0.5756 | 0.529065 | low-surrogate-fidelity |

## Interpretation boundary

Sobol indices quantify variance within the declared parameter ranges. They are not causal effects outside those ranges. Offline replay assumes the Full100 trajectory up to the simulated stop epoch. Fresh Rancher runs remain necessary for final confirmation.
