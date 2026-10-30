"""Coordinate admitted work without owning its locks, recipes or persistence."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager


@dataclass(frozen=True, slots=True)
class Resolved[Result]:
    """Return an adapter's replay or no-work result without computing again."""

    result: Result


@dataclass(frozen=True, slots=True)
class Pending[Attempt]:
    """Carry opaque admitted work while its adapter owns the surrounding scope."""

    attempt: Attempt


def run_attempt[Attempt, Product, Reference, Result](
    *,
    admission: AbstractContextManager[Resolved[Result] | Pending[Attempt]],
    compute: Callable[[Attempt], Product],
    publish: Callable[[Attempt, Product], Reference],
    accept: Callable[[Reference], Result],
) -> Result:
    """Compute, publish and accept inside admission, or return its resolved result.

    The adapter owns admission, cleanup and every scientific operation. Failures
    propagate without retries; admission contexts must not suppress them. The
    result tags are transient coordination values, never persisted schemas.
    """
    completed: list[Result] = []
    with admission as admitted:
        if isinstance(admitted, Resolved):
            completed.append(admitted.result)
        else:
            product = compute(admitted.attempt)
            reference = publish(admitted.attempt, product)
            completed.append(accept(reference))
    if not completed:
        message = "Admission must not suppress attempt failures."
        raise RuntimeError(message)
    return completed[0]
