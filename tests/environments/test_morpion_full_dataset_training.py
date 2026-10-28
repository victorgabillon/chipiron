"""All nine existing candidates consume an unbounded Dataset beyond 512 rows."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap.cycle_training import (
    morpion_training_args_from_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    canonical_morpion_evaluator_specs,
    entity_token_transformer_small_morpion_evaluator_spec,
)
from chipiron.environments.morpion.learning import (
    iter_morpion_supervised_row_chunks_from_path,
    load_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    train_morpion_regressor_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
    MorpionTrainingContinuation,
)
from tests.environments.test_morpion_continual_training import rows_file

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.exhaustive
def test_all_nine_candidates_train_beyond_512_rows(tmp_path: Path) -> None:
    """Exercise actual cached streaming updates with bounded decoding chunks."""
    source = load_morpion_supervised_rows(rows_file(tmp_path / "source")).rows
    dataset = tmp_path / "full.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=(replace(source[i % len(source)], node_id=str(i)) for i in range(513)),
        metadata={"purpose": "unbounded_training_regression"},
        path=dataset,
    )
    seen = 0
    for chunk in iter_morpion_supervised_row_chunks_from_path(
        dataset, chunk_size=31, max_rows=None
    ):
        assert 0 < len(chunk) <= 31
        seen += len(chunk)
    assert seen == 513

    specs = canonical_morpion_evaluator_specs()
    transformer = entity_token_transformer_small_morpion_evaluator_spec()
    specs[transformer.name] = transformer
    assert len(specs) == 9
    for name, spec in specs.items():
        training = morpion_training_args_from_evaluator_spec(
            spec=replace(spec, num_epochs=1),
            dataset_file=dataset,
            output_dir=tmp_path / name,
            shuffle=True,
            validation_fraction=0.2,
            validation_seed=0,
            device="cpu",
            continuation=MorpionTrainingContinuation(name, 42),
        )
        _, metrics = train_morpion_regressor_streaming(
            MorpionStreamingTrainingArgs(training, row_chunk_size=31, max_rows=None)
        )
        assert metrics["num_samples"] == 513, name
        assert metrics["num_train_samples"] > 0, name
        assert metrics["num_validation_samples"] > 0, name
