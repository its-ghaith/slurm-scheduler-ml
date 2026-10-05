# Live-Shadow Controller Benchmark

## Purpose

This benchmark accelerates controller development by training each dataset only
once for 100 epochs while several stopping controllers observe the same live
epoch stream. It is an online experiment, not an offline replay: a controller
receives only the measurements available at the end of the current epoch.

The development campaign contains exactly nine jobs with `training_seed=0`:

| Task | Dataset case |
|---|---|
| Aerial vehicle detection | `carpk-aerial-vehicle` |
| Aerial vehicle detection | `vedai-aerial-vehicle` |
| Aerial vehicle detection | `visdrone-aerial-vehicle` |
| Image classification | `cifar10-classification` |
| Image classification | `cifar100-classification` |
| Image classification | `tiny-imagenet-classification` |
| Semantic segmentation | `oxford-pet-segmentation` |
| Semantic segmentation | `pascal-voc2012-segmentation` |
| Semantic segmentation | `uavid-segmentation` |

No stability-seed repetitions are included because this protocol is intended
for controller development. The final thesis evaluation must repeat selected
controllers as real stopping runs with multiple seeds.

## Shadow controllers

Every Full100 job evaluates these four policies after every completed epoch:

1. `standard-es-10`: task-independent patience-based early-stopping baseline.
2. `rapec-v3-pf`: profile-free RAPEC-v3 controller.
3. `rapec-v3-pf-energy5`: the energy-oriented RAPEC-v3-PF level-5 variant.
4. `rapec-v10`: dynamically calibrated, profile-free RAPEC-v10 controller.

The controller configuration is stored in
`controller_benchmark/config/live-shadow-controllers.json`.

## Online decision protocol

At epoch `t`, all controllers receive the same observation containing the
current quality score, cumulative epoch GPU energy, epoch duration, GPU
utilization and the available task-independent telemetry. Future epochs are
not visible.

When controller `c` emits its first stop signal, the suite freezes:

- virtual stop epoch `t_c`,
- quality and best-so-far quality at `t_c`,
- cumulative epoch GPU energy through `t_c`,
- cumulative controller compute time and controller overhead energy,
- stop reason and diagnostics.

The child controller is not evaluated again. The top-level shadow suite always
returns `stop=false`, so the shared model continues to epoch 100 and provides
the Full100 reference from the same physical training trajectory.

## Energy accounting

For controller `c`, the development benchmark reports:

\[
E_{shadow,c}=E_{epoch,1:t_c}+E_{controller,c}
\]

where:

- `E_epoch,1:t_c` is the measured cumulative GPU energy of completed training
  epochs through the virtual stop epoch.
- `E_controller,c` is the energy attributed to that controller's online
  calculations up to its first stop signal.

Controller energy is measured from Intel RAPL package counters when available.
If RAPL is unavailable, the implementation reports an explicitly labelled
process-CPU-time estimate using the configured watts-per-CPU-second factor.
Controller wall time is always reported separately.

Preprocessing and finalization energy are intentionally excluded from the
primary shadow comparison because the shared Full100 run executes these phases
only once and their counterfactual cost cannot be observed separately for each
virtual controller. Consequently, the metric scope is:

```text
epoch GPU energy + attributed controller CPU energy
```

It is not a complete lifecycle-energy estimate. CodeCarbon whole-job energy is
also not divided among shadow controllers because that would count the same
shared work multiple times.

## Scientific interpretation

The live-shadow protocol is appropriate for fast development because every
controller sees the same model trajectory, hardware state, data order and
random seed. This removes between-run noise and reduces nine datasets times
four policies to nine training jobs.

It does not prove that stopping had no causal effect on later training. A real
stopping run omits later epochs and may have different finalization overhead.
Therefore, live-shadow results are used to select promising controllers. Final
claims must be confirmed by independent physical runs that really stop at the
selected epoch.

## Commands

Validate the nine-job plan without submitting work:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-live-shadow-nine-dataset-async.ps1 -PlanOnly
```

Submit the real asynchronous Rancher campaign:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-live-shadow-nine-dataset-async.ps1
```

After submission, the script prints a `RunId`. The computer may then be turned
off because orchestration and training continue inside Rancher. Query status
with:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\get-controller-benchmark-status.ps1 -RunId '<RunId>'
```

## Outputs

The analyzer writes:

- `live-shadow-report.json`: complete campaign and controller aggregates.
- `benchmark-report.json`: compatibility copy for benchmark tooling.
- `paired-results.csv`: one row per dataset-controller pair.
- Prometheus metrics prefixed with `live_shadow_`.
- Grafana dashboard `Live Shadow Controllers - Nine Dataset Development`.

The controller-specific result files are persisted after every epoch as
`shadow_summary_job_<job-id>.json`, so completed virtual decisions survive an
interrupted Full100 job.
