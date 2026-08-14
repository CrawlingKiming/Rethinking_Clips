"""Regenerate the ESS policy-validation figure from saved CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from ess_policy_optimization import bin_diagnostics, make_figure, write_csv


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
    diagnostic_bins = bin_diagnostics(
        read_rows(results / "rlvr_minibatch_diagnostics.csv")
    )
    write_csv(results / "rlvr_diagnostic_bins.csv", diagnostic_bins)
    make_figure(
        diagnostic_bins,
        read_rows(results / "rlvr_training_paths.csv"),
        0.1,
        Path("figures/ess_policy_validation"),
    )
