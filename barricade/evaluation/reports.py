"""Longitudinal scoreboards built from standalone evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def _protocol(report: dict) -> dict:
    settings = report["settings"]
    return {
        "opening_suite_sha256": report["opening_suite"]["sha256"],
        "simulations": settings["simulations"],
        "max_plies": settings["max_plies"],
        "confidence": settings["confidence"],
        "promotion_threshold": settings["promotion_threshold"],
    }


def build_scoreboard(reports: Sequence[tuple[str, dict]]) -> dict:
    if not reports:
        raise ValueError("at least one evaluation report is required")
    reference = _protocol(reports[0][1])
    warnings = []
    rows = []
    for source, report in reports:
        if report.get("format_version") != 1:
            raise ValueError(f"unsupported report format in {source}")
        protocol = _protocol(report)
        if protocol != reference:
            warnings.append(
                f"{source} uses a different protocol and is not directly comparable"
            )
        candidate = report["candidate"]
        matches = {}
        for match in report["matches"]:
            matches[match["opponent"]] = {
                "score": match["score"],
                "lower": match["interval"]["lower"],
                "upper": match["interval"]["upper"],
                "wins": match["wins"],
                "draws": match["draws"],
                "losses": match["losses"],
                "decision": match["decision"]["status"],
            }
        rows.append(
            {
                "source": source,
                "generation": candidate.get("generation"),
                "checkpoint": candidate["path"],
                "model_sha256": candidate["model_sha256"],
                "promotion": report["promotion"]["status"],
                "matches": matches,
            }
        )
    rows.sort(
        key=lambda row: (
            row["generation"] is None,
            row["generation"] if row["generation"] is not None else 0,
            row["checkpoint"],
        )
    )
    return {
        "format_version": 1,
        "comparable": not warnings,
        "protocol": reference,
        "warnings": warnings,
        "generations": rows,
    }


def load_reports(paths: Sequence[str | Path]) -> list[tuple[str, dict]]:
    results = []
    for path in paths:
        resolved = Path(path).resolve()
        data = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"evaluation report must be a JSON object: {resolved}")
        results.append((str(resolved), data))
    return results
