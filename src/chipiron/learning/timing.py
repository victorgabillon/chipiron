"""Small wall-clock timing helpers for Chipiron learning workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


@dataclass(slots=True)
class PhaseDurations:
    """Accumulated wall-clock durations, in seconds, keyed by phase name."""

    _durations: dict[str, float] = field(default_factory=dict)

    def add_duration(self, phase: str, seconds: float) -> None:
        """Add a non-negative duration to one named phase."""
        if seconds < 0.0:
            raise ValueError("seconds must be non-negative.")  # noqa: TRY003
        self._durations[phase] = self.get(phase) + seconds

    @contextmanager
    def time_phase(self, phase: str) -> Iterator[None]:
        """Measure and accumulate wall-clock time spent in one named phase."""
        started_at = perf_counter()
        try:
            yield
        finally:
            self.add_duration(phase, perf_counter() - started_at)

    def get(self, phase: str) -> float:
        """Return the accumulated seconds for one phase, or 0.0."""
        return self._durations.get(phase, 0.0)

    def as_dict(self) -> dict[str, float]:
        """Return a copy of accumulated seconds by phase."""
        return dict(self._durations)

    def prefixed_dict(self, prefix: str) -> dict[str, float]:
        """Return durations as ``{prefix}_{phase}_s: seconds}``."""
        return {
            f"{prefix}_{phase}_s": seconds for phase, seconds in self._durations.items()
        }

    def total_seconds(self) -> float:
        """Return the sum of all accumulated phase durations."""
        return sum(self._durations.values())


def format_phase_durations(
    durations: Mapping[str, float],
    *,
    min_seconds: float = 0.0,
) -> str:
    """Format phase timings for compact logs."""
    return " ".join(
        f"{phase}={seconds:.3f}s"
        for phase, seconds in durations.items()
        if seconds >= min_seconds
    )
