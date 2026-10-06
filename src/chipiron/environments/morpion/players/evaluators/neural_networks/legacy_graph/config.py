"""Explicit historical architecture; never aliases modern entity-token features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from coral.neural_networks.models.entity_token_transformer_value_net import (
    EntityTokenTransformerValueNet,
    EntityTokenTransformerValueNetArgs,
)


@dataclass(frozen=True, slots=True)
class LegacyGraphConfig:
    """Preserve all nine historical graph fields without reinterpretation."""

    representation: str = "graph_tokens_v1"
    graph_max_tokens: int = 1536
    graph_input_feature_dim: int = 26
    graph_d_model: int = 64
    graph_n_head: int = 4
    graph_n_layer: int = 2
    graph_dim_feedforward: int = 256
    graph_dropout_ratio: float = 0.0
    graph_pooling: Literal["value_token", "masked_mean"] = "value_token"
    graph_output_tanh: bool = False

    def __post_init__(self) -> None:
        """Reject schema changes and validate the original Coral architecture."""
        if (
            self.representation != "graph_tokens_v1"
            or self.graph_input_feature_dim != 26
        ):
            message = "Legacy graph_tokens_v1 requires exactly 26 historical features."
            raise ValueError(message)
        if self.graph_max_tokens < 2:
            message = (
                "Legacy graph_tokens_v1 requires at least VALUE and GLOBAL tokens."
            )
            raise ValueError(message)
        self.coral_args()

    def coral_args(self) -> EntityTokenTransformerValueNetArgs:
        """Make historical implicit defaults explicit, including BOTH value tokens."""
        return EntityTokenTransformerValueNetArgs(
            input_feature_dim=self.graph_input_feature_dim,
            d_model=self.graph_d_model,
            n_head=self.graph_n_head,
            n_layer=self.graph_n_layer,
            dim_feedforward=self.graph_dim_feedforward,
            dropout_ratio=self.graph_dropout_ratio,
            pooling=self.graph_pooling,
            output_tanh=self.graph_output_tanh,
            use_validity_feature=True,
            validity_feature_index=-1,
            use_value_token=True,
        )

    def build_model(self) -> EntityTokenTransformerValueNet:
        """Use the same generic Coral network and parameter names as historical code."""
        return EntityTokenTransformerValueNet(self.coral_args())

    def to_dict(self) -> dict[str, object]:
        """Serialize the explicit legacy opt-in and unchanged historical fields."""
        return asdict(self)


def legacy_graph_config_from_dict(data: object) -> LegacyGraphConfig | None:
    """Parse a strict opt-in block; reject unknown fields or loose JSON coercions."""
    if data is None:
        return None
    if not isinstance(data, dict) or data.get("representation") != "graph_tokens_v1":
        message = "An explicit graph_tokens_v1 compatibility block is required."
        raise ValueError(message)
    defaults = LegacyGraphConfig().to_dict()
    if set(data) != set(defaults):
        message = (
            "Legacy graph configuration must contain exactly the documented fields."
        )
        raise ValueError(message)
    for key, value in data.items():
        expected = type(defaults[key])
        if expected is float:
            valid = type(value) in (int, float)
        else:
            # Exact types reject bool-as-int.
            valid = type(value) is expected  # pylint: disable=unidiomatic-typecheck
        if not valid:
            message = f"Invalid legacy graph field type: {key}."
            raise ValueError(message)
    return LegacyGraphConfig(**data)
