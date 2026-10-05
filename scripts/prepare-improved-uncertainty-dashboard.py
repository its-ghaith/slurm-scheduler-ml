from __future__ import annotations

import argparse
import json
from pathlib import Path


def update_strings(value):
    if isinstance(value, dict):
        return {key: update_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [update_strings(item) for item in value]
    if isinstance(value, str):
        value = value.replace(
            'controller_mode="uncertainty_aware"',
            'controller_mode=~"uncertainty_aware|conformal_energy_aware|hybrid_pareto_energy_aware"',
        )
        value = value.replace(
            'controller_mode=~"uncertainty_aware|conformal_energy_aware"',
            'controller_mode=~"uncertainty_aware|conformal_energy_aware|hybrid_pareto_energy_aware"',
        )
        value = value.replace(
            "Uncertainty-Aware Controller", "Uncertainty-Aware Controller Comparison"
        )
        value = value.replace(
            "Old and Improved Uncertainty-Aware Controllers",
            "Uncertainty-Aware Controller Comparison",
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dashboard = json.loads(args.source.read_text(encoding="utf-8-sig"))
    dashboard = update_strings(dashboard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, separators=(",", ":")), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
