from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ControllerContext:
    controller_id: str
    benchmark_version: str
    task_type: str
    quality_metric: str
    scenario: str
    max_epochs: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpochObservation:
    epoch: int
    max_epochs: int
    task_type: str
    quality_metric: str
    quality: float | None  # Task-independent validation quality Q_t in [0, 1].
    best_quality: float | None
    delta_quality: float | None
    epoch_energy_wh: float
    cumulative_energy_wh: float
    epoch_duration_seconds: float
    cumulative_duration_seconds: float
    gpu_utilization_pct: float | None
    history: Sequence[Mapping[str, Any]]
    raw_metrics: Mapping[str, Any]

    @property
    def checkpoint(self) -> int:
        """Task-neutral alias for ``epoch``.

        A checkpoint can be an epoch, an evaluation cycle, a boosting round,
        or another regularly evaluated unit of iterative training.
        """

        return self.epoch

    @property
    def max_checkpoints(self) -> int:
        """Task-neutral alias for ``max_epochs``."""

        return self.max_epochs

    @property
    def progress_fraction(self) -> float:
        """Completed fraction of the configured training budget."""

        return max(0.0, min(1.0, self.checkpoint / max(1, self.max_checkpoints)))


@dataclass(frozen=True)
class ControllerDecision:
    stop: bool
    reason: str = "continue"
    confidence: float | None = None
    predicted_energy_saving_fraction: float | None = None
    predicted_quality_regret: float | None = None
    diagnostics: Mapping[str, float | int | bool | None] = field(default_factory=dict)


class Controller(ABC):
    """Stable interface implemented by every benchmark controller."""

    def __init__(self, context: ControllerContext):
        self.context = context

    @abstractmethod
    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        """Return one side-effect-free continue/stop decision."""

    def close(self) -> None:
        """Optional lifecycle hook called after training."""
