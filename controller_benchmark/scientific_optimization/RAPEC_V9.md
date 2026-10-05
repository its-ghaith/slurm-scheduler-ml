# RAPEC-v9: Risk-Constrained Bayesian Multi-Horizon Stopping

RAPEC-v9 is an online, task-independent early-stopping controller. It uses only
the current training run and never reads Full100 curves, dataset names, task
profiles, or historical model runs during a stop decision.

## Decision model

At every epoch, the controller predicts validation quality, training energy,
and training duration for horizons 1, 5, 10, 20, and the complete remaining
epoch budget. Bayesian model averaging combines linear, logarithmic,
saturating, exponential, change-point, and local-trend models.

The online feature vector contains validation quality, training loss, gradient
norm, learning rate, epoch duration, GPU utilization, GPU memory, GPU power,
epoch energy, parameter count, and FLOPs. Varying telemetry may contribute to
quality, energy, and duration prediction. Parameter count and FLOPs are static
within one run and are therefore retained as context rather than assigned a
spurious causal coefficient without cross-run history.

The quality-equivalence margin is derived online from detrended validation
noise. A rolling quantile of the robust noise estimates prevents the margin
from collapsing to zero merely because the learning-rate schedule becomes
small. The quantile is dimensionless and is selected under the outer quality
constraint; no fixed accuracy or percentage-point margin is encoded.

Prequential under-prediction residuals augment the posterior gain samples.
Their finite-sample conformal quantile remains available as a conservative
upper forecast bound. Training may stop only when the probability of a future
gain above the dynamic margin is at most `quality_risk_alpha` for every active
horizon.

A short recovery guard protects learning-rate restarts. If the telemetry model
is incomplete and receives less posterior support than its equal model prior,
an empirical record-waiting model learns the typical plateau length from the
current run. This prevents a short false plateau from stopping training without
introducing a fixed patience or a dataset-specific rule.

Energy is not traded against quality. Nested task-type holdout optimization
first rejects controllers that violate dynamic non-inferiority and then
maximizes the lower-quartile epoch-energy saving among feasible controllers.
The internal posterior risk and sequential-evidence settings may be optimized,
but they cannot relax the outer quality constraint.

Offline replay is used only for screening parameters. The reported scientific
result must come from the fresh paired seed-1 validation because replay assumes
that the Full100 trajectory is unchanged until the selected stop epoch.

## LC-PFN baseline

`controller_benchmark.controllers.lcpfn_quality_baseline:LcpfnQualityBaselineController`
provides the external LC-PFN comparison adapter. It deliberately predicts only
the quality curve; the same measured epoch-energy stream is used afterward to
report its energy outcome. RAPEC-v9 additionally models energy, duration, and
telemetry and applies a hard quality-risk constraint, so the comparison can
separate learning-curve prediction from the proposed decision mechanism.

LC-PFN is not installed into the existing Rancher training container at runtime:
its published package constrains PyTorch to an older version than the current
training stack. It must be evaluated in a pinned compatible image; silently
downgrading PyTorch inside a measured job would invalidate both reproducibility
and the energy comparison.

## Default benchmark

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v9-controller-benchmark-async.ps1 `
  -RequireAllStages
```

The v4 benchmark uses a separate baseline namespace and compares epoch energy
as the primary controller metric. Whole-job energy remains available as a
secondary operational metric. It uses training seed 1, while the immutable
optimization source uses seed 0. Previous benchmark versions and results are
not modified.

## Scientific optimization

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v9-scientific-optimization.ps1 `
  -StartInBackground
```

After completion, submit only a feasible selected controller:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v9-scientific-validation-async.ps1 `
  -OptimizationDir .\results\controller-v9-scientific-optimization\runs\<optimization-id> `
  -PlanOnly
```

Remove `-PlanOnly` after inspecting the nine planned candidate runs and nine
fresh Full100 baselines. The asynchronous Rancher orchestrator continues after
the local computer is turned off.

## Fair optimizer comparison

The separate optimizer study compares Sobol, qLogNEHVI, and constrained
RF-ParEGO with identical initial points, budgets, and optimizer seeds. It also
adds an Oracle upper bound, exact persistent replay caching, and skips repeated
Morris screening because all seven RAPEC-v9 parameters are mandatory.

See `controller_benchmark/scientific_optimization/OPTIMIZER_COMPARISON.md` for
the complete protocol and commands. Existing optimization and Rancher results
are never overwritten.
