from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ParameterCodec:
    base: Mapping[str, Any]
    specifications: Mapping[str, Mapping[str, Any]]
    active_names: tuple[str, ...]

    @classmethod
    def from_search_space(
        cls,
        base: Mapping[str, Any],
        search_space: Mapping[str, Any],
        active_names: Sequence[str] | None = None,
    ) -> "ParameterCodec":
        specifications = dict(search_space["parameters"])
        tunable = [name for name, spec in specifications.items() if spec["type"] != "fixed"]
        selected = tuple(active_names) if active_names is not None else tuple(tunable)
        unknown = sorted(set(selected) - set(tunable))
        if unknown:
            raise ValueError(f"Unknown or fixed active parameters: {unknown}")
        return cls(dict(base), specifications, selected)

    @property
    def dimension(self) -> int:
        return len(self.active_names)

    @property
    def tunable_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, spec in self.specifications.items() if spec["type"] != "fixed"
        )

    def decode(self, values: Sequence[float]) -> dict[str, Any]:
        if len(values) != self.dimension:
            raise ValueError(f"Expected {self.dimension} normalized values, got {len(values)}")
        parameters = dict(self.base)
        for name, spec in self.specifications.items():
            if spec["type"] == "fixed":
                parameters[name] = spec["value"]
        for name, raw in zip(self.active_names, values):
            parameters[name] = self._decode_value(self.specifications[name], float(raw))
        return parameters

    def encode(self, parameters: Mapping[str, Any]) -> list[float]:
        return [
            self._encode_value(self.specifications[name], parameters.get(name, self.base.get(name)))
            for name in self.active_names
        ]

    @staticmethod
    def _decode_value(spec: Mapping[str, Any], raw: float) -> Any:
        value = max(0.0, min(1.0, raw))
        kind = spec["type"]
        if kind == "int":
            low, high = int(spec["low"]), int(spec["high"])
            step = int(spec.get("step", 1))
            count = max(0, (high - low) // step)
            return low + min(count, int(round(value * count))) * step
        if kind == "float":
            low, high = float(spec["low"]), float(spec["high"])
            if spec.get("log"):
                return math.exp(math.log(low) + value * (math.log(high) - math.log(low)))
            return low + value * (high - low)
        if kind == "categorical":
            choices = list(spec["choices"])
            index = min(len(choices) - 1, int(value * len(choices)))
            return choices[index]
        raise ValueError(f"Unsupported parameter type: {kind}")

    @staticmethod
    def _encode_value(spec: Mapping[str, Any], raw: Any) -> float:
        kind = spec["type"]
        if kind == "int":
            low, high = int(spec["low"]), int(spec["high"])
            return max(0.0, min(1.0, (int(raw) - low) / max(1, high - low)))
        if kind == "float":
            low, high = float(spec["low"]), float(spec["high"])
            value = max(low, min(high, float(raw)))
            if spec.get("log"):
                return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
            return (value - low) / (high - low)
        if kind == "categorical":
            choices = list(spec["choices"])
            index = choices.index(raw)
            return (index + 0.5) / len(choices)
        raise ValueError(f"Unsupported parameter type: {kind}")
