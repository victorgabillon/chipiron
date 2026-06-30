"""Tests for human-readable Morpion bootstrap training logs."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_LOGGING_PATH = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "bootstrap"
    / "training_logging.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "chipiron.environments.morpion.bootstrap.training_logging",
    _TRAINING_LOGGING_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(_TRAINING_LOGGING_PATH)
_training_logging = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _training_logging
_SPEC.loader.exec_module(_training_logging)

TrainingActiveModelCursorSummary = _training_logging.TrainingActiveModelCursorSummary
log_training_cycle_idle = _training_logging.log_training_cycle_idle
log_training_cycle_start = _training_logging.log_training_cycle_start
log_training_evaluator_start = _training_logging.log_training_evaluator_start
log_training_progress = _training_logging.log_training_progress

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


def _messages(caplog: LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def test_training_cycle_start_summary_includes_dataset_and_evaluators(
    caplog: LogCaptureFixture,
) -> None:
    """Cycle start logs include generation, dataset, chunk, and evaluator facts."""
    caplog.set_level(logging.INFO)

    log_training_cycle_start(
        generation=26,
        rows_path=Path("/work/rows/generation_000026.jsonl"),
        row_count=162_118,
        row_format="jsonl",
        chunk_size=8192,
        evaluator_names=("entity_token_transformer_small", "linear_10"),
        active_model_cursor_summary=TrainingActiveModelCursorSummary(
            active_model_generation=430,
            active_model_source_generation=430,
            active_model_source="external_seed",
            cursor_started_generation=24,
            cursor_completed_generation=None,
            local_lower_bound_generation=24,
        ),
        max_rows=None,
    )

    messages = _messages(caplog)
    assert "[training-cycle] generation=26 status=STARTED" in messages
    assert "rows=162118" in messages
    assert "chunk_size=8192" in messages
    assert "chunks=20" in messages
    assert "evaluators=2" in messages
    assert "source=external_seed" in messages
    assert "source_generation=430" in messages


def test_training_evaluator_start_summary_includes_index(
    caplog: LogCaptureFixture,
) -> None:
    """Evaluator start logs include the human-readable evaluator index."""
    caplog.set_level(logging.INFO)

    log_training_evaluator_start(
        generation=26,
        evaluator_name="entity_token_transformer_small",
        evaluator_index=1,
        evaluator_count=9,
        row_count=162_118,
        chunk_count=20,
    )

    messages = _messages(caplog)
    assert "[training-evaluator]" in messages
    assert "evaluator=entity_token_transformer_small" in messages
    assert "index=1/9" in messages
    assert "status=STARTED" in messages


def test_training_progress_summary_includes_rows_percent_and_chunk(
    caplog: LogCaptureFixture,
) -> None:
    """Progress logs include rows seen, percent complete, and chunk index."""
    caplog.set_level(logging.INFO)

    log_training_progress(
        generation=26,
        evaluator_name="entity_token_transformer_small",
        chunk_index=1,
        chunk_count=20,
        rows_seen=8192,
        row_count=162_118,
        elapsed_s=12.4,
    )

    messages = _messages(caplog)
    assert "[training-progress]" in messages
    assert "chunk=1/20" in messages
    assert "rows_seen=8192/162118" in messages
    assert "percent=5.1" in messages


def test_training_idle_summary_includes_reason_and_lower_bound(
    caplog: LogCaptureFixture,
) -> None:
    """Idle logs include no-claimable reason and active lower-bound fields."""
    caplog.set_level(logging.INFO)

    log_training_cycle_idle(
        reason="no_claimable_generation",
        pending_generations=(),
        claimable_generations=(),
        local_lower_bound=24,
        active_model_source_generation=430,
    )

    messages = _messages(caplog)
    assert "[training-cycle] status=IDLE" in messages
    assert "reason=no_claimable_generation" in messages
    assert "pending_generations=none" in messages
    assert "claimable_generations=none" in messages
    assert "local_lower_bound=24" in messages
    assert "active_model_source_generation=430" in messages
