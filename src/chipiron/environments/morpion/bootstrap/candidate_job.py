"""One candidate recipe, shared by the inline and spawned CPU executors.

No selection, acceptance, status publication or new optimizer loop belongs here.
D2's admitted-attempt lifecycle owns the executor's lifetime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.args import (
        MorpionStreamingTrainingArgs,  # noqa: TC004 - deferred annotation on the job recipe
    )


@dataclass(frozen=True)
class CandidateJob:
    """Pickleable recipe with the original seed, parent, input and output identity."""

    name: str
    training: MorpionStreamingTrainingArgs
    cpu_threads: int | None = None


def run_candidate(
    job: CandidateJob,
    *,
    trainer: Callable[
        [MorpionStreamingTrainingArgs],
        tuple[MorpionRegressor, dict[str, float | str | None]],
    ]
    | None = None,
) -> tuple[MorpionRegressor, dict[str, float | str | None]]:
    """Invoke the existing public streaming service without changing its recipe."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        train_morpion_regressor_streaming,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.metadata import (
        update_saved_manifest_metadata,
    )

    from .candidate_threading import KEY, candidate_threads, validate_thread_provenance
    from .generation_publication_io import read_object

    args = job.training.training_args
    continuation = args.continuation
    if continuation is not None and continuation.parent_bundle is not None:
        parent = read_object(Path(continuation.parent_bundle) / "morpion_manifest.json")
        validate_thread_provenance(parent["metadata"], job.cpu_threads)
    with candidate_threads(
        job.cpu_threads, name=job.name, device=args.device
    ) as evidence:
        result = (train_morpion_regressor_streaming if trainer is None else trainer)(
            job.training
        )
    if evidence:
        manifest = read_object(Path(args.output_dir) / "morpion_manifest.json")
        update_saved_manifest_metadata(
            args.output_dir, {**manifest["metadata"], KEY: evidence}
        )
    return result


def prepare_candidate_cache(job: CandidateJob) -> None:
    """Prepare an attempt-owned cache before any child can read it; retain no tensors."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
        is_relational_entity_token_model_kind,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
        is_morpion_entity_token_model_kind,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.entity_token_cache import (
        load_or_materialize_entity_token_cache,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.flat_tensor_cache import (
        is_flat_morpion_training_model_kind,
        load_or_materialize_flat_tensor_cache,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.relational_entity_token_cache import (
        load_or_materialize_relational_entity_token_cache,
    )

    args = job.training.training_args
    if is_flat_morpion_training_model_kind(args.model_kind):
        load_or_materialize_flat_tensor_cache(
            rows_path=args.dataset_file,
            row_chunk_size=job.training.row_chunk_size,
            max_rows=job.training.max_rows,
        )
    elif is_relational_entity_token_model_kind(args.model_kind):
        load_or_materialize_relational_entity_token_cache(
            rows_path=args.dataset_file,
            row_chunk_size=job.training.row_chunk_size,
            max_rows=job.training.max_rows,
            entity_max_tokens=args.entity_max_tokens,
        )
    elif is_morpion_entity_token_model_kind(args.model_kind):
        load_or_materialize_entity_token_cache(
            rows_path=args.dataset_file,
            row_chunk_size=job.training.row_chunk_size,
            max_rows=job.training.max_rows,
            entity_max_tokens=args.entity_max_tokens,
            legacy_graph_tokens=args.legacy_graph_tokens,
        )
