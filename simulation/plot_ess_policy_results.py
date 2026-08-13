"""Regenerate the ESS policy-validation figure from saved CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from ess_policy_optimization import make_figure


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
    results = Path("simulation/results")
    make_figure(
        read_rows(results / "ess_coverage_results.csv"),
        Path("figures/ess_policy_validation"),
    )
