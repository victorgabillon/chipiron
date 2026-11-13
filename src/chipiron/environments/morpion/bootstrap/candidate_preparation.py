"""Operational opt-in around the existing serial candidate preparation boundary."""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        MorpionStreamingTrainingArgs,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _PreparedInputs,
    )

    from .candidate_job import CandidateJob

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparationRecipe:
    """Parent-independent common native preparation options."""

    fraction: float
    seed: int
    max_rows: int | None
    row_chunk_size: int

    def identity(self) -> dict[str, object]:
        """Reuse the native representation/split identity, binding the chunk recipe too."""
        from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
            _Recipe,
        )

        return {
            **_Recipe(self.fraction, self.seed, self.max_rows).identity(),
            "row_chunk_size": self.row_chunk_size,
        }


def preparation_recipe(
    streams: Sequence[MorpionStreamingTrainingArgs], *, serial: bool
) -> PreparationRecipe | None:
    """Resolve the same eligibility without loading parents or initializing a model."""
    raw = os.environ.get("CHIPIRON_SHARED_PREPARATION")
    if raw is None or raw == "0":
        return None
    if raw != "1":
        message = "CHIPIRON_SHARED_PREPARATION must be 0 or 1"
        raise ValueError(message)
    from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
        is_relational_entity_token_model_kind,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.flat_tensor_cache import (
        is_flat_morpion_training_model_kind,
    )

    reason = None
    if not serial or not streams:
        reason = "requires a nonempty serial candidate cohort"
    else:
        first = streams[0]
        base = first.training_args
        kinds = [stream.training_args.model_kind for stream in streams]
        if not (
            any(is_flat_morpion_training_model_kind(kind) for kind in kinds)
            and any(is_relational_entity_token_model_kind(kind) for kind in kinds)
            and all(
                is_flat_morpion_training_model_kind(kind)
                or is_relational_entity_token_model_kind(kind)
                for kind in kinds
            )
        ):
            reason = "requires the flat plus relational-v1 family"
        elif Path(base.dataset_file).suffix != ".jsonl":
            reason = "requires JSONL rows"
        else:
            for stream in streams:
                args = stream.training_args
                if not args.stable_validation_split or (
                    Path(args.dataset_file).resolve(),
                    args.validation_fraction,
                    args.validation_seed,
                    stream.max_rows,
                    stream.row_chunk_size,
                ) != (
                    Path(base.dataset_file).resolve(),
                    base.validation_fraction,
                    base.validation_seed,
                    first.max_rows,
                    first.row_chunk_size,
                ):
                    reason = "requires one common source and stable split recipe"
                    break
                if is_relational_entity_token_model_kind(args.model_kind) and (
                    args.entity_max_tokens != 1536
                    or args.legacy_graph_tokens is not None
                ):
                    reason = "requires the existing relational1536 representation"
                    break
    if reason is not None:
        LOGGER.info("shared_preparation fallback=%s", reason)
        return None
    first = streams[0]
    base = first.training_args
    return PreparationRecipe(
        float(base.validation_fraction),
        base.validation_seed,
        first.max_rows,
        first.row_chunk_size,
    )


@contextmanager
def prepare_candidate_inputs(
    jobs: Sequence[CandidateJob],
    *,
    serial: bool,
    borrowed: _PreparedInputs | None = None,
) -> Iterator[_PreparedInputs | None]:
    """Acquire native ownership once, or validate an already leased Training input."""
    recipe = preparation_recipe([job.training for job in jobs], serial=serial)
    if borrowed is not None:
        from .generation_publication_io import require

        require(recipe is not None, "Borrowed preparation no longer eligible")
        assert recipe is not None
        require(
            borrowed.recipe.identity()
            == {k: v for k, v in recipe.identity().items() if k != "row_chunk_size"},
            "Borrowed preparation recipe differs",
        )
        for job in jobs:
            borrowed.prepare_split(job.training.training_args, job.training.max_rows)
        yield borrowed
        return
    if recipe is None:
        yield None
        return
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        prepare_shared_inputs,
    )

    with prepare_shared_inputs(
        Path(jobs[0].training.training_args.dataset_file),
        fraction=recipe.fraction,
        seed=recipe.seed,
        max_rows=recipe.max_rows,
        row_chunk_size=recipe.row_chunk_size,
    ) as prepared:
        if prepared is not None:
            prepared.prepare_split(jobs[0].training.training_args, recipe.max_rows)
        yield prepared
