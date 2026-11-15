"""Morpion split provenance over shared cached row-index mechanics."""

# Explicit aliases preserve the public exports for strict type checking.
# pylint: disable=useless-import-alias,unused-import
from chipiron.learning.supervised.index_schedule import (
    CachedIndexSchedule as CachedIndexSchedule,
)
from chipiron.learning.supervised.index_schedule import (
    cached_index_schedule as _cached_index_schedule,
)
from chipiron.learning.supervised.index_schedule import (
    index_batches as index_batches,
)
from chipiron.learning.supervised.index_schedule import (
    shuffled_epoch_train_indices as shuffled_epoch_train_indices,
)
from chipiron.learning.supervised.index_schedule import (
    split_indices_for_streaming_policy as split_indices_for_streaming_policy,
)

# pylint: enable=useless-import-alias,unused-import


def cached_index_schedule(
    *,
    row_count: int,
    validation_fraction: float,
    validation_indices: frozenset[int] | None = None,
) -> CachedIndexSchedule:
    """Retain Morpion's canonical-state label for caller-supplied membership."""
    return _cached_index_schedule(
        row_count=row_count,
        validation_fraction=validation_fraction,
        validation_indices=validation_indices,
        explicit_split_policy="canonical_state_sha256_v1",
    )
