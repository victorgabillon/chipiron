"""Tests for cached Morpion training index schedules."""

from __future__ import annotations

from chipiron.environments.morpion.players.evaluators.neural_networks.training.cached_index_schedule import (
    cached_index_schedule,
    index_batches,
    shuffled_epoch_train_indices,
)


def test_cached_index_schedule_matches_streaming_validation_policy() -> None:
    """Cached split selection should match the streaming modulo policy."""
    schedule = cached_index_schedule(row_count=10, validation_fraction=0.2)

    assert schedule.train_indices == (0, 1, 2, 3, 5, 6, 7, 8)
    assert schedule.validation_indices == (4, 9)
    assert schedule.split_policy == "index_modulo_5"


def test_cached_index_schedule_has_no_validation_when_fraction_is_zero() -> None:
    """A zero validation fraction should keep all rows in training."""
    schedule = cached_index_schedule(row_count=5, validation_fraction=0.0)

    assert schedule.train_indices == (0, 1, 2, 3, 4)
    assert schedule.validation_indices == ()
    assert schedule.split_policy == "none"


def test_shuffled_epoch_train_indices_is_deterministic_per_epoch() -> None:
    """Global cached shuffling should be deterministic and epoch-dependent."""
    indices = tuple(range(20))

    first = shuffled_epoch_train_indices(
        train_indices=indices,
        shuffle=True,
        validation_seed=17,
        epoch_index=0,
    )
    repeated = shuffled_epoch_train_indices(
        train_indices=indices,
        shuffle=True,
        validation_seed=17,
        epoch_index=0,
    )
    next_epoch = shuffled_epoch_train_indices(
        train_indices=indices,
        shuffle=True,
        validation_seed=17,
        epoch_index=1,
    )

    assert first == repeated
    assert first != indices
    assert next_epoch != first
    assert sorted(first) == list(indices)


def test_shuffled_epoch_train_indices_preserves_order_when_shuffle_is_false() -> None:
    """Disabling shuffle should preserve cached training-index order."""
    indices = tuple(range(20))

    assert (
        shuffled_epoch_train_indices(
            train_indices=indices,
            shuffle=False,
            validation_seed=17,
            epoch_index=1,
        )
        == indices
    )


def test_index_batches_yields_fixed_size_batches() -> None:
    """Index batching should preserve order and yield a final short batch."""
    assert tuple(index_batches((0, 1, 2, 3, 4), batch_size=2)) == (
        (0, 1),
        (2, 3),
        (4,),
    )
