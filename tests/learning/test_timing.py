"""Tests for common learning timing helpers."""

from __future__ import annotations

import pytest

from chipiron.learning.timing import PhaseDurations, format_phase_durations


def test_phase_durations_add_duration_accumulates_values() -> None:
    """Phase durations should accumulate repeated phase values."""
    durations = PhaseDurations()

    durations.add_duration("forward", 1.25)
    durations.add_duration("forward", 0.75)

    assert durations.get("forward") == 2.0


def test_phase_durations_rejects_negative_duration() -> None:
    """Negative phase durations should fail clearly."""
    durations = PhaseDurations()

    with pytest.raises(ValueError, match="non-negative"):
        durations.add_duration("forward", -0.1)


def test_phase_durations_get_returns_zero_for_missing_phase() -> None:
    """Missing phases should read as zero seconds."""
    assert PhaseDurations().get("missing") == 0.0


def test_phase_durations_as_dict_returns_copy() -> None:
    """The exported duration mapping should not mutate the source."""
    durations = PhaseDurations()
    durations.add_duration("forward", 1.0)

    exported = durations.as_dict()
    exported["forward"] = 3.0

    assert durations.get("forward") == 1.0


def test_phase_durations_prefixed_dict_adds_prefix_and_suffix() -> None:
    """Prefixed duration mappings should use stable timing-style keys."""
    durations = PhaseDurations()
    durations.add_duration("forward", 1.5)

    assert durations.prefixed_dict("train") == {"train_forward_s": 1.5}


def test_format_phase_durations_respects_min_seconds() -> None:
    """Formatted timings should filter small phases when requested."""
    formatted = format_phase_durations(
        {"forward": 1.2345, "loss": 0.001},
        min_seconds=0.01,
    )

    assert formatted == "forward=1.234s"
