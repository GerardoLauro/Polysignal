"""Bundle manifest for the high-signal Telegram dataset.

The manifest is a small JSON file that ties together the outputs of the
pipeline (selected markets, plan, coverage report, raw Telegram run
directories). It exists so a downstream pipeline can locate every
artefact from one file instead of probing the filesystem.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polysignal.dataset.models import DatasetTargets


def build_manifest(
    *,
    selected_markets_path: Path,
    telegram_plan_parquet_path: Path,
    telegram_plan_json_path: Path,
    coverage_report_path: Path | None,
    raw_run_dirs: Sequence[Path] = (),
    normalized_messages_path: Path | None = None,
    segments_path: Path | None = None,
    channels_config_path: Path | None = None,
    targets: DatasetTargets,
    notes: Sequence[str] = (),
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Compose the manifest payload."""
    moment = (generated_at or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)

    def _opt(value: Path | None) -> str | None:
        return str(value) if value is not None else None

    return {
        "generated_at": moment.isoformat(),
        "dataset_name": "high_signal_telegram",
        "targets": {
            "selected_market_target": targets.selected_market_target,
            "raw_telegram_messages_target_min": targets.raw_telegram_messages_target_min,
            "raw_telegram_messages_target_max": targets.raw_telegram_messages_target_max,
            "candidate_segments_target_min": targets.candidate_segments_target_min,
            "candidate_segments_target_max": targets.candidate_segments_target_max,
            "linked_segments_target_min": targets.linked_segments_target_min,
            "linked_segments_target_max": targets.linked_segments_target_max,
            "minimum_viable_linked_segments": targets.minimum_viable_linked_segments,
        },
        "artefacts": {
            "selected_markets_parquet": str(selected_markets_path),
            "telegram_search_plan_parquet": str(telegram_plan_parquet_path),
            "telegram_search_plan_json": str(telegram_plan_json_path),
            "telegram_coverage_report_json": _opt(coverage_report_path),
            "normalized_messages_parquet": _opt(normalized_messages_path),
            "segments_parquet": _opt(segments_path),
            "channels_config_yaml": _opt(channels_config_path),
            "raw_run_dirs": [str(p) for p in raw_run_dirs],
        },
        "notes": list(notes),
    }


def write_manifest(payload: dict[str, Any], path: Path) -> Path:
    """Serialise the manifest payload to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = ["build_manifest", "write_manifest"]
