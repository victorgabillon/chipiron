"""Opaque admitted work keeps its resource scope and never retries itself."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Never

import pytest

from chipiron.bootstrap import Pending, Resolved, run_attempt

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.fast


def unexpected(*_args: object) -> Never:
    """Reject computation or publication on an already resolved admission."""
    pytest.fail("Resolved admission invoked scientific work")


@pytest.mark.parametrize("result", [None, object()], ids=["no-work", "accepted"])
def test_resolved_admission_only_enters_and_releases(result: object) -> None:
    """Even a falsey resolved result bypasses all callbacks and releases admission."""
    events = []

    @contextmanager
    def admission() -> Iterator[Resolved[object]]:
        events.append("enter")
        try:
            yield Resolved(result)
        finally:
            events.append("release")

    assert (
        run_attempt(
            admission=admission(),
            compute=unexpected,
            publish=unexpected,
            accept=unexpected,
        )
        is result
    )
    assert events == ["enter", "release"]


@pytest.mark.parametrize(
    ("failed_stage", "failure_type"),
    [(None, RuntimeError)]
    + [
        (stage, error)
        for stage in ("compute", "publish", "accept")
        for error in (RuntimeError, KeyboardInterrupt)
    ],
)
def test_pending_scope_and_failure_order(
    failed_stage: str | None, failure_type: type[BaseException]
) -> None:
    """Publication and acceptance use live resources; failures unwind without retry."""
    attempt, product, reference, result = (object() for _ in range(4))
    expected = failure_type("attempt failed")
    events: list[str] = []
    live = False

    @contextmanager
    def admission() -> Iterator[Pending[object]]:
        nonlocal live
        live = True
        events.append("enter")
        try:
            yield Pending(attempt)
        finally:
            events.append("release")
            live = False

    def visit(stage: str) -> None:
        assert live
        events.append(stage)
        if stage == failed_stage:
            raise expected

    def compute(admitted: object) -> object:
        assert admitted is attempt
        visit("compute")
        return product

    def publish(admitted: object, computed: object) -> object:
        assert admitted is attempt and computed is product
        visit("publish")
        return reference

    def accept(published: object) -> object:
        assert published is reference
        visit("accept")
        return result

    if failed_stage is None:
        assert (
            run_attempt(
                admission=admission(), compute=compute, publish=publish, accept=accept
            )
            is result
        )
        assert events == ["enter", "compute", "publish", "accept", "release"]
    else:
        with pytest.raises(failure_type) as caught:
            run_attempt(
                admission=admission(), compute=compute, publish=publish, accept=accept
            )
        assert caught.value is expected
        sequence = ["enter", "compute", "publish", "accept"]
        assert events == [*sequence[: sequence.index(failed_stage) + 1], "release"]
    assert not live


@pytest.mark.parametrize("before_yield", [True, False])
def test_admission_failure_is_not_hidden(before_yield: bool) -> None:
    """An admission or release failure remains the adapter's original exception."""
    expected = RuntimeError("admission failed")

    @contextmanager
    def admission() -> Iterator[Resolved[object]]:
        if before_yield:
            raise expected
        yield Resolved(object())
        raise expected

    with pytest.raises(RuntimeError) as caught:
        run_attempt(
            admission=admission(),
            compute=unexpected,
            publish=unexpected,
            accept=unexpected,
        )
    assert caught.value is expected


def test_admission_cannot_hide_computation_failure() -> None:
    """A suppressing external context cannot report an implicit successful result."""

    @contextmanager
    def admission() -> Iterator[Pending[object]]:
        try:
            yield Pending(object())
        except RuntimeError:
            return

    def compute(_attempt: object) -> Never:
        message = "computation failed"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="must not suppress"):
        run_attempt(
            admission=admission(),
            compute=compute,
            publish=unexpected,
            accept=unexpected,
        )
