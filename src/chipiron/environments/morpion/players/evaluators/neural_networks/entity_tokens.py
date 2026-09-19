"""Entity-token tensor conversion for Morpion neural evaluators."""
# pyright: reportMissingImports=false

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor

from chipiron.environments.morpion.action_geometry import (
    morpion_action_new_point,
    morpion_action_segments,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.latent_window_features import (
    LatentWindowMoveFeatureCounts,
    MorpionLatentWindowInstrumentation,
    build_morpion_latent_window_feature_context,
    compute_promoted_latent_window_count_reference,
    compute_promoted_latent_window_counts,
    corrected_latent_window_move_feature_counts,
)
from chipiron.environments.morpion.types import (
    MorpionAction,
    MorpionDynamics,
    MorpionState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from atomheart.games.morpion.state import Point, Segment

MORPION_ENTITY_TOKEN_MODEL_KIND: Final[str] = "entity_token_transformer_value_net"
MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION: Final[str] = "morpion_entity_tokens_v1"
MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_INPUT_REPRESENTATION: Final[str] = (
    "morpion_entity_tokens_global_extent_v2"
)
MORPION_ENTITY_TOKEN_PROSPECTIVE_EDGES_INPUT_REPRESENTATION: Final[str] = (
    "morpion_entity_tokens_prospective_edges_v2"
)
MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_INPUT_REPRESENTATION: Final[str] = (
    "morpion_entity_tokens_promoted_windows_v2"
)
MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_INPUT_REPRESENTATION: Final[str] = (
    "morpion_entity_tokens_promoted_and_blocked_windows_v3"
)
MORPION_ENTITY_TOKEN_CORRECTED_BLOCKING_INPUT_REPRESENTATION: Final[str] = (
    "morpion_entity_tokens_promoted_and_blocked_windows_corrected_v4"
)
type MorpionGlobalGeometryFeatures = Literal["none", "normalization_extent"]
type MorpionEdgeTokenMode = Literal["drawn_only", "drawn_and_prospective"]
type MorpionLatentWindowMoveFeatures = Literal[
    "none",
    "promoted_only",
    "promoted_and_blocked",
    "promoted_and_blocked_corrected",
]


def is_morpion_entity_token_model_kind(model_kind: str) -> bool:
    """Return whether a model kind consumes Morpion entity-token tensors."""
    return model_kind == MORPION_ENTITY_TOKEN_MODEL_KIND


class MorpionEntityTokenType(IntEnum):
    """Entity types used by the first clean Morpion token representation."""

    GLOBAL = 0
    DOT = 1
    EDGE = 2
    MOVE = 3


MORPION_ENTITY_TOKEN_DIRECTIONS: Final[tuple[str, ...]] = (
    "horizontal",
    "vertical",
    "diag_up",
    "diag_down",
)

MORPION_ENTITY_TOKEN_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "type_global",
    "type_dot",
    "type_edge",
    "type_move",
    "x_rel",
    "y_rel",
    "dir_horizontal",
    "dir_vertical",
    "dir_diag_up",
    "dir_diag_down",
    "occupied",
    "candidate",
    "drawn",
    "degree_horizontal",
    "degree_vertical",
    "degree_diag_up",
    "degree_diag_down",
    "is_new_dot_for_move",
    "missing_index_in_5_window",
    "path_position",
    "num_points",
    "moves",
    "legal_action_count",
    "used_unit_segment_count",
    "validity",
)
MORPION_ENTITY_TOKEN_FEATURE_DIM: Final[int] = len(MORPION_ENTITY_TOKEN_FEATURE_NAMES)
MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_NAMES: Final[tuple[str, ...]] = (
    *MORPION_ENTITY_TOKEN_FEATURE_NAMES,
    "normalization_extent",
)
MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_DIM: Final[int] = len(
    MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_NAMES
)
MORPION_LATENT_WINDOW_MOVE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "promoted_latent_window_count",
    "blocked_by_consumed_segment_count",
    "blocked_by_same_direction_touching_count",
)
MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_FEATURE_NAMES: Final[tuple[str, ...]] = (
    *MORPION_ENTITY_TOKEN_FEATURE_NAMES,
    MORPION_LATENT_WINDOW_MOVE_FEATURE_NAMES[0],
)
MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_FEATURE_NAMES: Final[
    tuple[str, ...]
] = (
    *MORPION_ENTITY_TOKEN_FEATURE_NAMES,
    *MORPION_LATENT_WINDOW_MOVE_FEATURE_NAMES,
)

_VALIDITY_INDEX: Final[int] = MORPION_ENTITY_TOKEN_FEATURE_DIM - 1


@dataclass(frozen=True, slots=True)
class TokenFeatureDescriptor:
    """Canonical semantics and expected range for one emitted feature."""

    index: int
    name: str
    semantic_type: str
    token_types: tuple[str, ...]
    expected_minimum: float | None
    expected_maximum: float | None
    padding_value: float | None
    description: str


_ALL_TOKEN_TYPES: Final[tuple[str, ...]] = ("global", "dot", "edge", "move")


def _feature_descriptor(
    name: str,
    semantic_type: str,
    token_types: tuple[str, ...],
    expected_minimum: float | None,
    expected_maximum: float | None,
    description: str,
) -> TokenFeatureDescriptor:
    """Build a descriptor whose index is derived from the emitted ordering."""
    return TokenFeatureDescriptor(
        index=MORPION_ENTITY_TOKEN_FEATURE_NAMES.index(name),
        name=name,
        semantic_type=semantic_type,
        token_types=token_types,
        expected_minimum=expected_minimum,
        expected_maximum=expected_maximum,
        padding_value=0.0,
        description=description,
    )


MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS: Final[tuple[TokenFeatureDescriptor, ...]] = (
    *(
        _feature_descriptor(
            f"type_{token_type}",
            "categorical_one_hot",
            _ALL_TOKEN_TYPES,
            0.0,
            1.0,
            f"One-hot indicator that this is a {token_type} token.",
        )
        for token_type in _ALL_TOKEN_TYPES
    ),
    _feature_descriptor(
        "x_rel",
        "continuous_coordinate",
        ("dot", "edge", "move"),
        -1.0,
        1.0,
        "State-local x coordinate relative to the occupied-board center and scale.",
    ),
    _feature_descriptor(
        "y_rel",
        "continuous_coordinate",
        ("dot", "edge", "move"),
        -1.0,
        1.0,
        "State-local y coordinate relative to the occupied-board center and scale.",
    ),
    *(
        _feature_descriptor(
            f"dir_{direction}",
            "categorical_one_hot",
            ("edge", "move"),
            0.0,
            1.0,
            f"Edge or legal-move orientation indicator: {direction}.",
        )
        for direction in MORPION_ENTITY_TOKEN_DIRECTIONS
    ),
    _feature_descriptor(
        "occupied",
        "binary",
        ("dot",),
        0.0,
        1.0,
        "Whether a dot position is already occupied.",
    ),
    _feature_descriptor(
        "candidate",
        "binary",
        ("dot", "move"),
        0.0,
        1.0,
        "Dot is a legal new-dot candidate; MOVE uses one for every legal action.",
    ),
    _feature_descriptor(
        "drawn",
        "binary",
        ("edge",),
        0.0,
        1.0,
        "Whether the represented unit edge is already drawn.",
    ),
    *(
        _feature_descriptor(
            f"degree_{direction}",
            "nonnegative_integer",
            ("dot",),
            0.0,
            None,
            f"Directional usage count at the dot for {direction}.",
        )
        for direction in MORPION_ENTITY_TOKEN_DIRECTIONS
    ),
    _feature_descriptor(
        "is_new_dot_for_move",
        "binary",
        ("move",),
        0.0,
        1.0,
        "Marks the move token coordinate as its prospective new dot.",
    ),
    _feature_descriptor(
        "missing_index_in_5_window",
        "ordinal_integer",
        ("move",),
        0.0,
        4.0,
        "Slot of the missing dot in the legal five-dot action window.",
    ),
    _feature_descriptor(
        "path_position",
        "continuous_fraction",
        ("move",),
        0.0,
        1.0,
        "Missing-window slot divided by four; deterministically redundant with it.",
    ),
    _feature_descriptor(
        "num_points",
        "nonnegative_integer",
        ("global",),
        0.0,
        None,
        "Number of occupied points in the state.",
    ),
    _feature_descriptor(
        "moves",
        "nonnegative_integer",
        ("global",),
        0.0,
        None,
        "Number of moves already played.",
    ),
    _feature_descriptor(
        "legal_action_count",
        "nonnegative_integer",
        ("global",),
        0.0,
        None,
        "Number of currently legal actions.",
    ),
    _feature_descriptor(
        "used_unit_segment_count",
        "nonnegative_integer",
        ("global",),
        0.0,
        None,
        "Number of already used unit segments.",
    ),
    _feature_descriptor(
        "validity",
        "binary_validity",
        _ALL_TOKEN_TYPES,
        0.0,
        1.0,
        "One for emitted semantic tokens and zero for batch padding.",
    ),
)
MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_DESCRIPTORS: Final[
    tuple[TokenFeatureDescriptor, ...]
] = (
    *MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS,
    TokenFeatureDescriptor(
        index=MORPION_ENTITY_TOKEN_FEATURE_DIM,
        name="normalization_extent",
        semantic_type="positive_continuous_scale",
        token_types=("global",),
        expected_minimum=1.0,
        expected_maximum=None,
        padding_value=0.0,
        description=(
            "Exact effective denominator used for x_rel/y_rel: "
            "max(board x-span, board y-span, 1.0)."
        ),
    ),
)
_LATENT_WINDOW_FEATURE_DESCRIPTIONS: Final[dict[str, str]] = {
    "promoted_latent_window_count": (
        "Unique three-of-five windows containing this MOVE token's new dot "
        "that remain segment- and same-direction-compatible after the move's "
        "geometry is accounted for."
    ),
    "blocked_by_consumed_segment_count": (
        "Unique potential promoted windows whose required unit segments "
        "overlap segments consumed by this MOVE."
    ),
    "blocked_by_same_direction_touching_count": (
        "Unique potential promoted windows made incompatible by this MOVE's "
        "same-direction point usage; zero when no such restriction applies."
    ),
}


def _latent_window_descriptor(
    name: str,
    *,
    index: int,
) -> TokenFeatureDescriptor:
    return TokenFeatureDescriptor(
        index=index,
        name=name,
        semantic_type="nonnegative_integer_count",
        token_types=("move",),
        expected_minimum=0.0,
        expected_maximum=20.0,
        padding_value=0.0,
        description=_LATENT_WINDOW_FEATURE_DESCRIPTIONS[name],
    )


MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_FEATURE_DESCRIPTORS: Final[
    tuple[TokenFeatureDescriptor, ...]
] = (
    *MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS,
    _latent_window_descriptor(
        MORPION_LATENT_WINDOW_MOVE_FEATURE_NAMES[0],
        index=MORPION_ENTITY_TOKEN_FEATURE_DIM,
    ),
)
MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_FEATURE_DESCRIPTORS: Final[
    tuple[TokenFeatureDescriptor, ...]
] = (
    *MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS,
    *(
        _latent_window_descriptor(
            name,
            index=MORPION_ENTITY_TOKEN_FEATURE_DIM + offset,
        )
        for offset, name in enumerate(MORPION_LATENT_WINDOW_MOVE_FEATURE_NAMES)
    ),
)

assert len(MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS) == MORPION_ENTITY_TOKEN_FEATURE_DIM
assert tuple(
    descriptor.index for descriptor in MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS
) == tuple(range(MORPION_ENTITY_TOKEN_FEATURE_DIM))
assert (
    tuple(descriptor.name for descriptor in MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS)
    == MORPION_ENTITY_TOKEN_FEATURE_NAMES
)
assert tuple(
    descriptor.index
    for descriptor in MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_DESCRIPTORS
) == tuple(range(MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_DIM))
assert (
    tuple(
        descriptor.name
        for descriptor in MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_DESCRIPTORS
    )
    == MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_NAMES
)


def validate_morpion_global_geometry_features(
    mode: str,
) -> MorpionGlobalGeometryFeatures:
    """Return one supported global-geometry mode."""
    if mode not in {"none", "normalization_extent"}:
        raise ValueError(  # noqa: TRY003
            f"Unsupported Morpion global geometry features: {mode!r}."
        )
    return mode  # type: ignore[return-value]


def validate_morpion_edge_token_mode(mode: str) -> MorpionEdgeTokenMode:
    """Return one supported edge-token representation mode."""
    if mode not in {"drawn_only", "drawn_and_prospective"}:
        raise ValueError(f"Unsupported Morpion edge token mode: {mode!r}.")  # noqa: TRY003
    return mode  # type: ignore[return-value]


def validate_morpion_latent_window_move_features(
    mode: str,
) -> MorpionLatentWindowMoveFeatures:
    """Return one supported legal-move latent-window feature mode."""
    if mode not in {
        "none",
        "promoted_only",
        "promoted_and_blocked",
        "promoted_and_blocked_corrected",
    }:
        raise ValueError(  # noqa: TRY003
            f"Unsupported Morpion latent-window move features: {mode!r}."
        )
    return mode  # type: ignore[return-value]


def morpion_entity_token_feature_names(
    mode: MorpionGlobalGeometryFeatures,
    latent_window_move_features: MorpionLatentWindowMoveFeatures = "none",
) -> tuple[str, ...]:
    """Return the stable feature names for one representation mode."""
    validate_morpion_latent_window_move_features(latent_window_move_features)
    if mode != "none" and latent_window_move_features != "none":
        raise ValueError(  # noqa: TRY003
            "Latent-window MOVE features cannot currently be combined with "
            "global geometry features."
        )
    if latent_window_move_features == "promoted_only":
        return MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_FEATURE_NAMES
    if latent_window_move_features in {
        "promoted_and_blocked",
        "promoted_and_blocked_corrected",
    }:
        return MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_FEATURE_NAMES
    if mode == "normalization_extent":
        return MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_NAMES
    return MORPION_ENTITY_TOKEN_FEATURE_NAMES


def morpion_entity_token_feature_descriptors(
    mode: MorpionGlobalGeometryFeatures,
    latent_window_move_features: MorpionLatentWindowMoveFeatures = "none",
) -> tuple[TokenFeatureDescriptor, ...]:
    """Return canonical descriptors for one representation mode."""
    _ = morpion_entity_token_feature_names(mode, latent_window_move_features)
    if latent_window_move_features == "promoted_only":
        return MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_FEATURE_DESCRIPTORS
    if latent_window_move_features in {
        "promoted_and_blocked",
        "promoted_and_blocked_corrected",
    }:
        return MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_FEATURE_DESCRIPTORS
    if mode == "normalization_extent":
        return MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_DESCRIPTORS
    return MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS


def morpion_entity_token_input_representation(
    mode: MorpionGlobalGeometryFeatures,
    edge_token_mode: MorpionEdgeTokenMode = "drawn_only",
    latent_window_move_features: MorpionLatentWindowMoveFeatures = "none",
) -> str:
    """Return the schema identifier for one representation mode."""
    validate_morpion_edge_token_mode(edge_token_mode)
    validate_morpion_latent_window_move_features(latent_window_move_features)
    _ = morpion_entity_token_feature_names(mode, latent_window_move_features)
    if latent_window_move_features != "none":
        if edge_token_mode != "drawn_only":
            raise ValueError(  # noqa: TRY003
                "Latent-window MOVE features require drawn-only edge tokens."
            )
        if latent_window_move_features == "promoted_only":
            return MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_INPUT_REPRESENTATION
        if latent_window_move_features == "promoted_and_blocked_corrected":
            return MORPION_ENTITY_TOKEN_CORRECTED_BLOCKING_INPUT_REPRESENTATION
        return MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_INPUT_REPRESENTATION
    if edge_token_mode == "drawn_and_prospective":
        if mode != "none":
            raise ValueError(  # noqa: TRY003
                "Prospective edge tokens cannot currently be combined with "
                "global geometry features."
            )
        return MORPION_ENTITY_TOKEN_PROSPECTIVE_EDGES_INPUT_REPRESENTATION
    if mode == "normalization_extent":
        return MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_INPUT_REPRESENTATION
    return MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION


class InvalidMorpionEntityTokenConverterError(ValueError):
    """Raised when entity-token conversion is configured invalidly."""

    @classmethod
    def invalid_max_tokens(cls) -> InvalidMorpionEntityTokenConverterError:
        """Return the invalid max-token-count error."""
        return cls("Morpion entity-token conversion requires max_tokens >= 1.")

    @classmethod
    def prospective_tokens_exceed_capacity(
        cls, *, required: int, maximum: int
    ) -> InvalidMorpionEntityTokenConverterError:
        """Return the forbidden prospective-token truncation error."""
        return cls(
            "Prospective-edge entity-token conversion requires "
            f"{required} tokens but max_tokens is {maximum}; silent truncation "
            "is forbidden."
        )


@dataclass(frozen=True, slots=True)
class MorpionEntityTokenLayout:
    """One entity tensor and stable indices for every surviving entity."""

    tensor: Tensor
    dot_index_by_point: Mapping[Point, int]
    edge_index_by_segment: Mapping[Segment, int]
    move_index_by_action: Mapping[MorpionAction, int]


@dataclass(frozen=True, slots=True)
class _MorpionEntityTokenRecord:
    """A token row and its optional game-entity identity."""

    row: list[float]
    point: Point | None = None
    segment: Segment | None = None
    action: MorpionAction | None = None


@dataclass(frozen=True, slots=True)
class _CoordinateNormalizer:
    """Normalize lattice coordinates into a state-local bounded scale."""

    center_x: float
    center_y: float
    scale: float

    def point(self, point: Point) -> tuple[float, float]:
        """Return normalized coordinates for one lattice point."""
        return (
            (float(point[0]) - self.center_x) / self.scale,
            (float(point[1]) - self.center_y) / self.scale,
        )

    def segment_center(self, segment: Segment) -> tuple[float, float]:
        """Return normalized coordinates for the midpoint of one segment."""
        (x0, y0), (x1, y1) = segment
        return (
            ((float(x0) + float(x1)) * 0.5 - self.center_x) / self.scale,
            ((float(y0) + float(y1)) * 0.5 - self.center_y) / self.scale,
        )


@dataclass(frozen=True, slots=True)
class MorpionEntityTokenConverter:
    """Convert Morpion states into variable-length entity-token tensors."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)
    max_tokens: int = 1536
    global_geometry_features: MorpionGlobalGeometryFeatures = "none"
    edge_token_mode: MorpionEdgeTokenMode = "drawn_only"
    latent_window_move_features: MorpionLatentWindowMoveFeatures = "none"
    latent_window_instrumentation: MorpionLatentWindowInstrumentation | None = field(
        default=None, compare=False, repr=False
    )
    use_scalar_latent_window_reference: bool = field(
        default=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate representation configuration without changing legacy defaults."""
        validate_morpion_global_geometry_features(self.global_geometry_features)
        validate_morpion_edge_token_mode(self.edge_token_mode)
        validate_morpion_latent_window_move_features(self.latent_window_move_features)
        _ = morpion_entity_token_input_representation(
            self.global_geometry_features,
            self.edge_token_mode,
            self.latent_window_move_features,
        )

    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered entity-token feature names."""
        return morpion_entity_token_feature_names(
            self.global_geometry_features,
            self.latent_window_move_features,
        )

    @property
    def input_dim(self) -> int:
        """Return the entity-token feature width."""
        return len(self.feature_names())

    def state_to_tensor(self, state: MorpionState) -> Tensor:
        """Return a ``T x F`` float32 tensor of real entity tokens for ``state``."""
        return self.state_to_layout(state).tensor

    def state_to_model_input_tensors(
        self,
        state: MorpionState,
    ) -> tuple[Tensor, ...]:
        """Return the ordinary model's single positional input tensor."""
        return (self.state_to_tensor(state),)

    def state_to_layout(self, state: MorpionState) -> MorpionEntityTokenLayout:
        """Return tokens and stable tensor indices for every surviving entity."""
        if self.max_tokens < 1:
            raise InvalidMorpionEntityTokenConverterError.invalid_max_tokens()

        normalizer = _normalizer_for_state(state)
        feature_names = self.feature_names()
        actions = self.dynamics.all_legal_actions(state)
        ordered_actions = tuple(sorted(actions, key=_action_sort_key))
        precomputed_latent_window_counts: (
            Mapping[MorpionAction, LatentWindowMoveFeatureCounts] | None
        ) = None
        if (
            self.latent_window_move_features
            in {"promoted_only", "promoted_and_blocked"}
            and not self.use_scalar_latent_window_reference
        ):
            latent_context = build_morpion_latent_window_feature_context(
                state,
                ordered_actions,
                instrumentation=self.latent_window_instrumentation,
            )
            precomputed_latent_window_counts = compute_promoted_latent_window_counts(
                context=latent_context,
                legal_actions=ordered_actions,
                state=state,
                instrumentation=self.latent_window_instrumentation,
            )
        candidate_points = _candidate_points_from_actions(actions)
        records: list[_MorpionEntityTokenRecord] = [
            _MorpionEntityTokenRecord(
                row=_global_token(
                    state=state,
                    legal_action_count=len(actions),
                    normalizer=normalizer,
                    feature_names=feature_names,
                    global_geometry_features=self.global_geometry_features,
                )
            )
        ]
        records.extend(
            _MorpionEntityTokenRecord(
                row=_dot_token(
                    point=point,
                    state=state,
                    candidate=point in candidate_points,
                    normalizer=normalizer,
                    feature_names=feature_names,
                ),
                point=point,
            )
            for point in sorted(state.points)
        )
        records.extend(
            _MorpionEntityTokenRecord(
                row=_dot_token(
                    point=point,
                    state=state,
                    candidate=True,
                    normalizer=normalizer,
                    feature_names=feature_names,
                ),
                point=point,
            )
            for point in sorted(candidate_points - state.points)
        )
        edge_records: list[_MorpionEntityTokenRecord] = []
        for raw_segment in sorted(state.used_unit_segments, key=_segment_sort_key):
            segment = canonical_segment(raw_segment)
            edge_records.append(
                _MorpionEntityTokenRecord(
                    row=_edge_token(
                        segment=segment,
                        drawn=True,
                        normalizer=normalizer,
                        feature_names=feature_names,
                    ),
                    segment=segment,
                )
            )
        if self.edge_token_mode == "drawn_and_prospective":
            drawn_segments = frozenset(
                canonical_segment(segment) for segment in state.used_unit_segments
            )
            prospective_segments = sorted(
                {
                    canonical_segment(segment)
                    for action in actions
                    for segment in morpion_action_segments(action)
                    if canonical_segment(segment) not in drawn_segments
                },
                key=_segment_sort_key,
            )
            edge_records.extend(
                _MorpionEntityTokenRecord(
                    row=_edge_token(
                        segment=segment,
                        drawn=False,
                        normalizer=normalizer,
                        feature_names=feature_names,
                    ),
                    segment=segment,
                )
                for segment in prospective_segments
            )
        records.extend(edge_records)
        records.extend(
            _MorpionEntityTokenRecord(
                row=_move_token(
                    action=action,
                    state=state,
                    normalizer=normalizer,
                    feature_names=feature_names,
                    latent_window_move_features=self.latent_window_move_features,
                    precomputed_latent_window_counts=precomputed_latent_window_counts,
                    latent_window_instrumentation=self.latent_window_instrumentation,
                    use_scalar_latent_window_reference=(
                        self.use_scalar_latent_window_reference
                    ),
                ),
                action=action,
            )
            for action in ordered_actions
        )

        if (
            self.edge_token_mode == "drawn_and_prospective"
            and len(records) > self.max_tokens
        ):
            raise InvalidMorpionEntityTokenConverterError.prospective_tokens_exceed_capacity(
                required=len(records), maximum=self.max_tokens
            )
        # Legacy mode retains its exact deterministic tail-truncation behavior.
        surviving_records = records[: self.max_tokens]
        dot_index_by_point: dict[Point, int] = {}
        edge_index_by_segment: dict[Segment, int] = {}
        move_index_by_action: dict[MorpionAction, int] = {}
        for index, record in enumerate(surviving_records):
            if record.point is not None:
                dot_index_by_point[record.point] = index
            if record.segment is not None:
                edge_index_by_segment[record.segment] = index
            if record.action is not None:
                move_index_by_action[record.action] = index

        return MorpionEntityTokenLayout(
            tensor=torch.tensor(
                [record.row for record in surviving_records], dtype=torch.float32
            ),
            dot_index_by_point=MappingProxyType(dot_index_by_point),
            edge_index_by_segment=MappingProxyType(edge_index_by_segment),
            move_index_by_action=MappingProxyType(move_index_by_action),
        )


def _blank_token(
    token_type: MorpionEntityTokenType,
    *,
    feature_names: tuple[str, ...],
) -> list[float]:
    """Return one zero-filled token row with type and validity set."""
    row = [0.0] * len(feature_names)
    row[token_type.value] = 1.0
    row[_VALIDITY_INDEX] = 1.0
    return row


def _global_token(
    *,
    state: MorpionState,
    legal_action_count: int,
    normalizer: _CoordinateNormalizer,
    feature_names: tuple[str, ...],
    global_geometry_features: MorpionGlobalGeometryFeatures,
) -> list[float]:
    """Return one global-context token."""
    row = _blank_token(MorpionEntityTokenType.GLOBAL, feature_names=feature_names)
    row[_feature_index("num_points", feature_names)] = float(len(state.points))
    row[_feature_index("moves", feature_names)] = float(state.moves)
    row[_feature_index("legal_action_count", feature_names)] = float(legal_action_count)
    row[_feature_index("used_unit_segment_count", feature_names)] = float(
        len(state.used_unit_segments)
    )
    if global_geometry_features == "normalization_extent":
        row[_feature_index("normalization_extent", feature_names)] = normalizer.scale
    return row


def _dot_token(
    *,
    point: Point,
    state: MorpionState,
    candidate: bool,
    normalizer: _CoordinateNormalizer,
    feature_names: tuple[str, ...],
) -> list[float]:
    """Return one occupied or candidate dot token."""
    row = _blank_token(MorpionEntityTokenType.DOT, feature_names=feature_names)
    x_rel, y_rel = normalizer.point(point)
    row[_feature_index("x_rel", feature_names)] = x_rel
    row[_feature_index("y_rel", feature_names)] = y_rel
    row[_feature_index("occupied", feature_names)] = (
        1.0 if point in state.points else 0.0
    )
    row[_feature_index("candidate", feature_names)] = 1.0 if candidate else 0.0
    dir_usage = state.dir_usage
    for dir_index, name in enumerate(MORPION_ENTITY_TOKEN_DIRECTIONS):
        row[_feature_index(f"degree_{name}", feature_names)] = float(
            dir_usage.get((point, dir_index), 0)
        )
    return row


def _edge_token(
    *,
    segment: Segment,
    drawn: bool,
    normalizer: _CoordinateNormalizer,
    feature_names: tuple[str, ...],
) -> list[float]:
    """Return one drawn or prospective unit-edge token."""
    row = _blank_token(MorpionEntityTokenType.EDGE, feature_names=feature_names)
    x_rel, y_rel = normalizer.segment_center(segment)
    row[_feature_index("x_rel", feature_names)] = x_rel
    row[_feature_index("y_rel", feature_names)] = y_rel
    row[_feature_index("drawn", feature_names)] = 1.0 if drawn else 0.0
    dir_index = _direction_index_for_segment(segment)
    if dir_index is not None:
        row[
            _feature_index(
                f"dir_{MORPION_ENTITY_TOKEN_DIRECTIONS[dir_index]}",
                feature_names,
            )
        ] = 1.0
    return row


def _move_token(
    *,
    action: MorpionAction,
    state: MorpionState,
    normalizer: _CoordinateNormalizer,
    feature_names: tuple[str, ...],
    latent_window_move_features: MorpionLatentWindowMoveFeatures,
    precomputed_latent_window_counts: Mapping[
        MorpionAction, LatentWindowMoveFeatureCounts
    ]
    | None,
    latent_window_instrumentation: MorpionLatentWindowInstrumentation | None,
    use_scalar_latent_window_reference: bool,
) -> list[float]:
    """Return one legal-action token."""
    # TODO: enrich MOVE tokens with explicit 5-dot/4-edge path features or
    # relation-aware encoding.
    dir_index, _x0, _y0, missing_index = action
    point = _missing_point_from_action(action)
    row = _blank_token(MorpionEntityTokenType.MOVE, feature_names=feature_names)
    x_rel, y_rel = normalizer.point(point)
    row[_feature_index("x_rel", feature_names)] = x_rel
    row[_feature_index("y_rel", feature_names)] = y_rel
    row[
        _feature_index(
            f"dir_{MORPION_ENTITY_TOKEN_DIRECTIONS[dir_index]}",
            feature_names,
        )
    ] = 1.0
    row[_feature_index("candidate", feature_names)] = 1.0
    row[_feature_index("is_new_dot_for_move", feature_names)] = 1.0
    row[_feature_index("missing_index_in_5_window", feature_names)] = float(
        missing_index
    )
    row[_feature_index("path_position", feature_names)] = float(missing_index) / 4.0
    if latent_window_move_features != "none":
        counts = (
            corrected_latent_window_move_feature_counts(state, action)
            if latent_window_move_features == "promoted_and_blocked_corrected"
            else compute_promoted_latent_window_count_reference(
                state,
                action,
                instrumentation=latent_window_instrumentation,
            )
            if use_scalar_latent_window_reference
            else _required_batched_latent_window_counts(
                precomputed_latent_window_counts, action
            )
        )
        values = {
            "promoted_latent_window_count": (counts.promoted_latent_window_count),
            "blocked_by_consumed_segment_count": (
                counts.blocked_by_consumed_segment_count
            ),
            "blocked_by_same_direction_touching_count": (
                counts.blocked_by_same_direction_touching_count
            ),
        }
        selected_names = (
            MORPION_LATENT_WINDOW_MOVE_FEATURE_NAMES[:1]
            if latent_window_move_features == "promoted_only"
            else MORPION_LATENT_WINDOW_MOVE_FEATURE_NAMES
        )
        for name in selected_names:
            value = float(values[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid latent-window feature {name}: {value}.")  # noqa: TRY003
            row[_feature_index(name, feature_names)] = value
    return row


def _required_batched_latent_window_counts(
    counts_by_action: Mapping[MorpionAction, LatentWindowMoveFeatureCounts] | None,
    action: MorpionAction,
) -> LatentWindowMoveFeatureCounts:
    if counts_by_action is None or action not in counts_by_action:
        raise ValueError(f"Missing batched latent-window result for {action!r}.")  # noqa: TRY003
    return counts_by_action[action]


def _candidate_points_from_actions(
    actions: tuple[MorpionAction, ...],
) -> frozenset[Point]:
    """Return the set of new-dot positions represented by legal actions."""
    return frozenset(_missing_point_from_action(action) for action in actions)


def _action_sort_key(action: MorpionAction) -> tuple[int, int, int, int]:
    """Return a stable key for raw Morpion actions."""
    dir_index, x0, y0, missing_index = action
    return (int(dir_index), int(x0), int(y0), int(missing_index))


def _segment_sort_key(
    segment: Segment,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return a stable key for unit segments independent of endpoint order."""
    point_a, point_b = segment
    ordered = tuple(sorted((point_a, point_b)))
    return (
        (int(ordered[0][0]), int(ordered[0][1])),
        (int(ordered[1][0]), int(ordered[1][1])),
    )


def canonical_segment(segment: Segment) -> Segment:
    """Return a unit segment with its endpoints in canonical point order."""
    point_a, point_b = segment
    ordered_a, ordered_b = sorted((point_a, point_b))
    return ordered_a, ordered_b


def _missing_point_from_action(action: MorpionAction) -> Point:
    """Return the absent point represented by one raw Morpion action."""
    return morpion_action_new_point(action)


def _direction_index_for_segment(segment: Segment) -> int | None:
    """Return the Morpion direction index for one unit segment, if recognized."""
    (x0, y0), (x1, y1) = segment
    delta = (abs(x1 - x0), y1 - y0)
    if delta == (1, 0):
        return 0
    if delta in ((0, 1), (0, -1)):
        return 1
    if delta == (1, 1):
        return 2
    if delta == (1, -1):
        return 3
    return None


def _normalizer_for_state(state: MorpionState) -> _CoordinateNormalizer:
    """Build a coordinate normalizer covering occupied points and used segments."""
    points = set(state.points)
    for segment in state.used_unit_segments:
        points.update(segment)
    if not points:
        return _CoordinateNormalizer(center_x=0.0, center_y=0.0, scale=1.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    center_x = (float(min(xs)) + float(max(xs))) * 0.5
    center_y = (float(min(ys)) + float(max(ys))) * 0.5
    scale = max(float(max(xs) - min(xs)), float(max(ys) - min(ys)), 1.0)
    return _CoordinateNormalizer(center_x=center_x, center_y=center_y, scale=scale)


def _feature_index(name: str, feature_names: tuple[str, ...]) -> int:
    """Return the index of one entity-token feature name."""
    return feature_names.index(name)


__all__ = [
    "MORPION_ENTITY_TOKEN_CORRECTED_BLOCKING_INPUT_REPRESENTATION",
    "MORPION_ENTITY_TOKEN_DIRECTIONS",
    "MORPION_ENTITY_TOKEN_FEATURE_DESCRIPTORS",
    "MORPION_ENTITY_TOKEN_FEATURE_DIM",
    "MORPION_ENTITY_TOKEN_FEATURE_NAMES",
    "MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_DESCRIPTORS",
    "MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_DIM",
    "MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_FEATURE_NAMES",
    "MORPION_ENTITY_TOKEN_GLOBAL_EXTENT_INPUT_REPRESENTATION",
    "MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION",
    "MORPION_ENTITY_TOKEN_MODEL_KIND",
    "MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_FEATURE_DESCRIPTORS",
    "MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_FEATURE_NAMES",
    "MORPION_ENTITY_TOKEN_PROMOTED_AND_BLOCKED_WINDOWS_INPUT_REPRESENTATION",
    "MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_FEATURE_DESCRIPTORS",
    "MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_FEATURE_NAMES",
    "MORPION_ENTITY_TOKEN_PROMOTED_WINDOWS_INPUT_REPRESENTATION",
    "MORPION_ENTITY_TOKEN_PROSPECTIVE_EDGES_INPUT_REPRESENTATION",
    "MORPION_LATENT_WINDOW_MOVE_FEATURE_NAMES",
    "MorpionEdgeTokenMode",
    "MorpionEntityTokenConverter",
    "MorpionEntityTokenLayout",
    "MorpionEntityTokenType",
    "MorpionGlobalGeometryFeatures",
    "MorpionLatentWindowMoveFeatures",
    "TokenFeatureDescriptor",
    "canonical_segment",
    "is_morpion_entity_token_model_kind",
    "morpion_entity_token_feature_descriptors",
    "morpion_entity_token_feature_names",
    "morpion_entity_token_input_representation",
    "validate_morpion_edge_token_mode",
    "validate_morpion_global_geometry_features",
    "validate_morpion_latent_window_move_features",
]
