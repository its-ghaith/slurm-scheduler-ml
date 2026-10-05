from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class QualityDefinition:
    """Documented mapping from a task metric to maximized quality in [0, 1]."""

    metric: str
    transform: str = "identity"
    scale: float = 1.0
    lower_bound: float | None = None
    upper_bound: float | None = None

    def normalize(self, raw_value: float) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{self.metric} must be finite, got {raw_value!r}.")
        if self.transform == "identity":
            quality = value
        elif self.transform == "complement":
            quality = 1.0 - value
        elif self.transform == "inverse_positive":
            if self.scale <= 0:
                raise ValueError("inverse_positive requires scale > 0.")
            quality = 1.0 / (1.0 + max(0.0, value) / self.scale)
        elif self.transform == "exp_negative":
            if self.scale <= 0:
                raise ValueError("exp_negative requires scale > 0.")
            quality = math.exp(-max(0.0, value) / self.scale)
        elif self.transform == "bounded_linear":
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("bounded_linear requires lower_bound and upper_bound.")
            width = self.upper_bound - self.lower_bound
            if width <= 0:
                raise ValueError("upper_bound must be greater than lower_bound.")
            quality = (value - self.lower_bound) / width
        else:
            raise ValueError(f"Unsupported quality transform: {self.transform!r}")
        return max(0.0, min(1.0, quality))

    def metadata(self) -> dict[str, Any]:
        return {
            "raw_quality_metric": self.metric,
            "quality_transform": self.transform,
            "quality_transform_scale": self.scale,
            "quality_transform_lower_bound": self.lower_bound,
            "quality_transform_upper_bound": self.upper_bound,
            "quality_direction": "maximize",
        }


def quality_definition(spec: Mapping[str, Any]) -> QualityDefinition:
    definition = spec.get("quality_definition") or {}
    if not isinstance(definition, Mapping):
        raise ValueError("quality_definition must be an object.")
    return QualityDefinition(
        metric=str(definition.get("metric", spec.get("primary_metric", "quality_score"))),
        transform=str(definition.get("transform", "identity")),
        scale=float(definition.get("scale", 1.0)),
        lower_bound=(
            float(definition["lower_bound"])
            if definition.get("lower_bound") is not None
            else None
        ),
        upper_bound=(
            float(definition["upper_bound"])
            if definition.get("upper_bound") is not None
            else None
        ),
    )

