"""Exact candidate records, converter bytes and legacy coordinate behavior."""

from dataclasses import replace
from random import Random
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from atomheart.games.morpion import initial_state
from atomheart.games.morpion.state import Variant

from chipiron.environments.morpion.bootstrap.runtime.checkpoint_codec import (
    ChipironMorpionStateCheckpointCodec,
    new_morpion_state_checkpoint_codec,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    feature_extractor as features,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    morpion_feature_subset_from_name,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState


def bounding_box_oracle(state: MorpionState) -> tuple[features.CandidateSegment, ...]:
    """Scan the same lattice box, retaining only segments consumed by features."""
    bbox = features._bbox(state.points)
    if bbox is None:
        return ()

    margin = 4
    min_x = bbox.min_x - margin
    max_x = bbox.max_x + margin
    min_y = bbox.min_y - margin
    max_y = bbox.max_y + margin
    candidate_segments: list[features.CandidateSegment] = []

    for dir_index, direction in enumerate(features.DIRECTIONS):
        for x0 in range(min_x, max_x + 1):
            for y0 in range(min_y, max_y + 1):
                points5 = features._line_points(x0=x0, y0=y0, direction=direction)
                missing_points = tuple(
                    point for point in points5 if point not in state.points
                )
                # Aggregates consume only 3/4-present segments. Fully occupied
                # segments contribute no missing cells to the frontier features.
                if len(missing_points) not in (1, 2):
                    continue
                unit_segments4 = features._unit_segments_on_line(points5)
                overlap_blocked = any(
                    segment in state.used_unit_segments for segment in unit_segments4
                )
                parallel_compatible = features._is_parallel_compatible(
                    state=state,
                    dir_index=dir_index,
                    points5=points5,
                )
                candidate_segments.append(
                    features.CandidateSegment(
                        dir_index=dir_index,
                        start_x=x0,
                        start_y=y0,
                        points5=points5,
                        unit_segments4=unit_segments4,
                        num_present=5 - len(missing_points),
                        num_missing=len(missing_points),
                        missing_points=missing_points,
                        overlap_blocked=overlap_blocked,
                        parallel_compatible=parallel_compatible,
                        legal=(
                            len(missing_points) == 1
                            and not overlap_blocked
                            and parallel_compatible
                        ),
                    )
                )

    return tuple(candidate_segments)


def assert_exact(state: MorpionState) -> None:
    """Compare every ordered record, then full bytes for each supported width."""
    expected = bounding_box_oracle(state)
    actual = features._candidate_segments(state)
    assert actual == expected
    assert [(s.dir_index, s.start_x, s.start_y) for s in actual] == sorted(
        (s.dir_index, s.start_x, s.start_y) for s in actual
    )
    rng = torch.get_rng_state().clone()
    for width in (41, 20, 10, 5):
        name = "handcrafted_41" if width == 41 else f"handcrafted_{width}_core"
        converter = MorpionFeatureTensorConverter(
            dynamics=MorpionDynamics(),
            feature_subset=morpion_feature_subset_from_name(name),
        )
        actual_tensor = converter.state_to_tensor(state)
        with patch.object(features, "_candidate_segments", bounding_box_oracle):
            expected_tensor = converter.state_to_tensor(state)
        assert actual_tensor.dtype == expected_tensor.dtype == torch.float32
        assert actual_tensor.shape == expected_tensor.shape == (width,)
        assert actual_tensor.numpy().tobytes() == expected_tensor.numpy().tobytes()
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.fast
@pytest.mark.parametrize("variant", list(Variant))
@pytest.mark.parametrize("seed", [0, 1, 1729])
def test_complete_legal_histories(variant: Variant, seed: int) -> None:
    """Both rules, actual overlap/parallel usage, and terminal states stay exact."""
    dynamics = MorpionDynamics()
    atom = initial_state(variant)
    random = Random(seed)
    for _ in range(160):
        state = MorpionState.from_atomheart_state(atom)
        assert_exact(state)
        actions = dynamics.inner.all_legal_actions(atom)
        if not actions:
            return
        atom = dynamics.inner.step(atom, random.choice(actions)).next_state
    pytest.fail("Fixture must include its terminal position")


@pytest.mark.fast
@pytest.mark.parametrize("variant", list(Variant))
def test_geometry_and_order(variant: Variant) -> None:
    """Empty, sparse, negative, translated and 3/4/5-occupied lines stay exact."""
    populations = [
        frozenset(),
        frozenset({(100, -200)}),
        frozenset({(-20, -20), (20, 20), (20, -20), (-20, 20)}),
        *(
            frozenset((100 + i * dx, -200 + i * dy) for i in range(count))
            for dx, dy in features.DIRECTIONS
            for count in (3, 4, 5)
        ),
    ]
    for points in populations:
        state = MorpionState(
            points=points,
            used_unit_segments=frozenset(),
            dir_usage_entries=(),
            moves=17,
            variant=variant,
        )
        assert_exact(state)
        assert_exact(
            replace(
                state,
                used_unit_segments=frozenset({((100, -200), (101, -200))}),
                dir_usage_entries=((((100, -200), 0), 3),),
                is_terminal=True,
            )
        )
        assert features._candidate_segments(state) == features._candidate_segments(
            replace(state, points=frozenset(reversed(sorted(points))))
        )


class IntegerSubclass(int):
    """An integer-like direct input outside the codec's plain-int fast path."""


@pytest.mark.fast
@pytest.mark.parametrize("variant", list(Variant))
def test_codec_reconstructs_integer_lattice_and_refuses_noninteger_codes(
    variant: Variant,
) -> None:
    """Scientific state refs replay integer move codes, not arbitrary coordinates."""
    codec = ChipironMorpionStateCheckpointCodec(
        inner=new_morpion_state_checkpoint_codec(profile_checkpoint=False),
        dynamics=MorpionDynamics(),
    )
    state = MorpionState.from_atomheart_state(initial_state(variant))
    payload = cast("tuple[int, tuple[int, ...]]", codec.dump_state_ref(state))
    restored = codec.load_state_ref(payload)
    assert type(restored.points) is frozenset
    assert all(type(c) is int for point in restored.points for c in point)
    for invalid_move in (True, 1.0, "1"):
        with pytest.raises(ValueError):
            codec.load_state_ref((payload[0], (invalid_move,)))


@pytest.mark.fast
@pytest.mark.parametrize(
    "points",
    [
        frozenset({(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)}),
        # Integral extrema make the old scan accept fractional interior points.
        frozenset({(-2, -2), (6, 2), (0.5, 0.5), (1.5, 0.5), (2.5, 0.5)}),
        frozenset({(-2, -2), (6, 2), (0.0, 0.0), (1.0, 0.0), (2.0, 0.0)}),
        frozenset({(False, False), (True, False), (2, 0)}),
        frozenset((IntegerSubclass(i), IntegerSubclass(0)) for i in range(4)),
        frozenset((np.int64(i), np.int64(0)) for i in range(4)),
        frozenset({("x", "y")}),
        frozenset({(0,)}),
        frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0)}),
        [(0, 0), (0, 0), (0, 0)],
        None,
    ],
)
def test_legacy_coordinate_domain(points: Any) -> None:
    """Preserve the old result or exception type, without sanitizing coordinates."""
    state = MorpionState(
        points=points, used_unit_segments=frozenset(), dir_usage_entries=()
    )
    try:
        expected = bounding_box_oracle(state)
    except (TypeError, ValueError, IndexError) as error:
        with pytest.raises(type(error)):
            features._candidate_segments(state)
    else:
        assert features._candidate_segments(state) == expected


@pytest.mark.fast
def test_fractional_extrema_remain_refused_by_converter() -> None:
    """The faster scan must not make a formerly refused converter input succeed."""
    state = MorpionState(
        points=cast("Any", frozenset({(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)})),
        used_unit_segments=frozenset(),
        dir_usage_entries=(),
    )
    converter = MorpionFeatureTensorConverter(dynamics=MorpionDynamics())
    for implementation in (bounding_box_oracle, features._candidate_segments):
        with (
            patch.object(features, "_candidate_segments", implementation),
            pytest.raises(TypeError),
        ):
            converter.state_to_tensor(state)


@pytest.mark.fast
def test_malformed_usage_remains_visible() -> None:
    """Enumeration cannot suppress the same downstream predicate error."""
    state = MorpionState(
        points=frozenset((i, 0) for i in range(4)),
        used_unit_segments=cast("Any", None),
        dir_usage_entries=(),
    )
    for implementation in (bounding_box_oracle, features._candidate_segments):
        with pytest.raises(TypeError):
            implementation(state)
