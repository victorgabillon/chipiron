"""Pure schedule semantics and the shared learning dependency boundary."""

import ast
import inspect
import random
from typing import Literal

import pytest

from chipiron.learning.supervised import cached, index_schedule
from chipiron.learning.supervised.index_schedule import cached_index_schedule


@pytest.mark.parametrize(
    ("row_count", "fraction", "indices", "train", "validation", "label"),
    [
        (7, 0.25, None, (0, 1, 2, 4, 5, 6), (3,), "index_modulo_4"),
        (4, 0.75, None, (0, 2), (1, 3), "index_modulo_2"),
        (4, 0.0, None, (0, 1, 2, 3), (), "none"),
        (4, 0.75, frozenset({-1, 2, 4}), (0, 1, 3), (2,), "explicit_indices"),
        (0, 0.2, None, (), (), "index_modulo_5"),
        (-1, 0.2, frozenset(), (), (), "explicit_indices"),
    ],
)
def test_neutral_index_membership(
    row_count: int,
    fraction: float,
    indices: frozenset[int] | None,
    train: tuple[int, ...],
    validation: tuple[int, ...],
    label: str,
) -> None:
    """Labels describe the supplied mechanism without inventing game provenance."""
    schedule = cached_index_schedule(
        row_count=row_count,
        validation_fraction=fraction,
        validation_indices=indices,
    )
    assert (
        schedule.train_indices,
        schedule.validation_indices,
        schedule.split_policy,
    ) == (train, validation, label)
    splits: tuple[tuple[Literal["train", "validation"], tuple[int, ...]], ...] = (
        ("train", train),
        ("validation", validation),
    )
    for split, expected in splits:
        assert (
            index_schedule.split_indices_for_streaming_policy(
                row_count=row_count,
                validation_fraction=fraction,
                validation_indices=indices,
                split=split,
            )
            == expected
        )


def test_explicit_label_and_shuffle_do_not_change_membership_or_global_rng() -> None:
    """The caller owns provenance; shuffling consumes only a local RNG."""
    default = cached_index_schedule(
        row_count=4, validation_fraction=0.0, validation_indices=frozenset({3})
    )
    labeled = cached_index_schedule(
        row_count=4,
        validation_fraction=0.0,
        validation_indices=frozenset({3}),
        explicit_split_policy="integer_fixture_holdout",
    )
    assert labeled.train_indices == default.train_indices == (0, 1, 2)
    assert labeled.validation_indices == default.validation_indices == (3,)
    assert labeled.split_policy == "integer_fixture_holdout"
    before = random.getstate()
    expected = [0, 1, 2]
    random.Random(17).shuffle(expected)
    assert index_schedule.shuffled_epoch_train_indices(
        train_indices=default.train_indices,
        shuffle=True,
        validation_seed=16,
        epoch_index=1,
    ) == tuple(expected)
    assert random.getstate() == before


def test_cached_learning_has_no_game_dependency() -> None:
    """Inspect these two modules, not transitive imports by unrelated factories."""
    for module in (cached, index_schedule):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("chipiron.environments")
            elif isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith("chipiron.environments")
                    for alias in node.names
                )
            elif isinstance(node, ast.Name):
                assert not node.id.startswith("Morpion")
