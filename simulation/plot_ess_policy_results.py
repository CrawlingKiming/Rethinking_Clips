"""Regenerate the ESS policy-validation figure from saved CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from ess_policy_optimization import (
    make_crossover_figure,
    make_figure,
    make_formula_oracle_figure,
)


def read_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, float | str]] = []
        for row in csv.DictReader(handle):
            converted: dict[str, float | str] = {"method": row.get("method", "")}
            for key, value in row.items():
                if key == "method":
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
        return rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    results = root / "simulation" / "results"
    diagnostic_bins = read_rows(results / "rlvr_diagnostic_bins.csv")
    make_figure(
        diagnostic_bins,
        read_rows(results / "rlvr_training_paths.csv"),
        0.1,
        root / "figures" / "ess_policy_validation",
        2048,
    )
    make_crossover_figure(
        read_rows(results / "rlvr_crossover_bins.csv"),
        root / "figures" / "ess_estimator_crossover",
    )
    make_formula_oracle_figure(
        read_rows(results / "rlvr_formula_oracle_components.csv"),
        read_rows(results / "rlvr_formula_oracle_thresholds.csv"),
        read_rows(results / "rlvr_formula_oracle_summary.csv"),
        read_rows(results / "rlvr_n512_optimization_paths.csv"),
        root / "figures" / "ess_formula_oracle",
    )
