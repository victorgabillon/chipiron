"""Small model arguments for converter and bundle compatibility tests."""

from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MORPION_ENTITY_TOKEN_MODEL_KIND,
    MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
    MorpionRegressorArgs,
)


def tiny_entity_model_args(*, relational: bool) -> MorpionRegressorArgs:
    """Construct minimal inference fixtures without running any training."""
    return MorpionRegressorArgs(
        model_kind=MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND
        if relational
        else MORPION_ENTITY_TOKEN_MODEL_KIND,
        entity_max_tokens=128,
        entity_d_model=4,
        entity_n_head=1,
        entity_n_layer=0,
        entity_dim_feedforward=8,
        entity_dropout_ratio=0.0,
        entity_relation_schema=MORPION_ENTITY_RELATION_SCHEMA if relational else None,
        entity_relation_type_count=MORPION_ENTITY_RELATION_TYPE_COUNT
        if relational
        else None,
    )
