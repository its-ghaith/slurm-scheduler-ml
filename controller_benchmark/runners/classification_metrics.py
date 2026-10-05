from __future__ import annotations


def primary_quality(primary_metric: str, accuracy: float, macro_f1: float) -> float:
    if primary_metric == "accuracy":
        return accuracy
    if primary_metric == "macro_f1":
        return macro_f1
    raise ValueError(f"Unsupported classification primary metric: {primary_metric}")
