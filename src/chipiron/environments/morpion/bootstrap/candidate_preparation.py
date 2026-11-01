"""Operational opt-in around the existing serial candidate preparation boundary."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _PreparedInputs,
    )

    from .candidate_job import CandidateJob

LOGGER = logging.getLogger(__name__)


@contextmanager
def prepare_candidate_inputs(
    jobs: Sequence[CandidateJob], *, serial: bool
) -> Iterator[_PreparedInputs | None]:
    """Share only the qualified family; retain every other existing valid path."""
    raw = os.environ.get("CHIPIRON_SHARED_PREPARATION")
    if raw is None or raw == "0":
        yield None
        return
    if raw != "1":
        message = "CHIPIRON_SHARED_PREPARATION must be 0 or 1"
        raise ValueError(message)
    from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
        is_relational_entity_token_model_kind,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.flat_tensor_cache import (
        is_flat_morpion_training_model_kind,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        prepare_shared_inputs,
    )

    reason = None
    if not serial or not jobs:
        reason = "requires a nonempty serial candidate cohort"
    else:
        first = jobs[0].training
        base = first.training_args
        kinds = [job.training.training_args.model_kind for job in jobs]
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
            for job in jobs:
                stream = job.training
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
        yield None
        return
    first = jobs[0].training
    base = first.training_args
    with prepare_shared_inputs(
        Path(base.dataset_file),
        fraction=base.validation_fraction,
        seed=base.validation_seed,
        max_rows=first.max_rows,
        row_chunk_size=first.row_chunk_size,
    ) as prepared:
        if prepared is not None:
            # Includes split handling in coordinator preparation, before the
            # first model/optimizer exists. Consumers seed the same exact seam.
            prepared.prepare_split(base, first.max_rows)
        yield prepared
