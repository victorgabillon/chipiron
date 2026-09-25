"""Small, isolated inputs for environment-specific unit tests."""

from types import SimpleNamespace

import pytest
from valanga.evaluations import Certainty, Value

from chipiron.environments.morpion.bootstrap.profiling import growth_memory
from chipiron.environments.morpion.bootstrap.profiling.recursive import rendering


@pytest.fixture
def bounded_profile_heap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply a small ambient heap without changing node traversal or summaries.

    Event/anatomy tests assert runner graphs and emitted fields, not the unrelated
    Torch/Qt/pytest heap. Keep fresh containers and a real project object here;
    the unpatched reverse-referrer test still scans the real process heap.
    Replace only these modules' GC references, never the global GC implementation.
    """
    objects: list[object] = [
        [1, 2],
        {"value": 1},
        (1, 2),
        {1, 2},
        frozenset({1, 2}),
        Value(score=0.5, certainty=Certainty.ESTIMATE),
    ]
    collector = SimpleNamespace(get_objects=lambda: list(objects))
    monkeypatch.setattr(growth_memory, "gc", collector)
    monkeypatch.setattr(rendering, "gc", collector)
