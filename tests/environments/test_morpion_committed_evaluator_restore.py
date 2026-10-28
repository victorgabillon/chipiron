"""Restore actual sealed neural models without a scientific model transition."""

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from anemone.checkpoints import checkpoint_payload_to_jsonable, serialize_value
from anemone.checkpoints.rng_state import rollout_random_generator

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    MorpionPipelineActiveModel,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.config import (
    MorpionBootstrapRolloutConfig,
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import (
    DerivationError,
    atomic_json,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GenerationPublicationError,
    committed_evaluator,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from tests.environments.morpion_durable_v2_helpers import (
    scientific_args,
    scientific_producer,
)
from tests.environments.test_morpion_anemone_runner import _make_model_bundle
from tests.environments.test_morpion_checkpoint_parity import _selection_signature


def nodes(instance: Any) -> Any:
    """Iterate resident nodes for small-fixture comparison."""
    descendants = instance._runtime.tree.descendants
    for depth in descendants.range():
        yield from descendants[depth].values()


def fingerprint(instance: Any) -> str:
    """Bounded stream of all current values, DAG links, PV and direct versions."""
    h = hashlib.sha256()
    for n in nodes(instance):
        e = n.tree_evaluation
        payload = dict(
            id=n.id,
            depth=n.tree_depth,
            parents=sorted(p.id for p in n.parent_nodes),
            children=sorted((str(b), c.id) for b, c in n.iter_child_links()),
            values=[
                None
                if v is None
                else checkpoint_payload_to_jsonable(serialize_value(v))
                for v in (e.direct_value, e.tree_value, e.effective_value)
            ],
            source=e.effective_value_source.value,
            pv=[str(b) for b in e.best_branch_sequence],
            version=e.direct_evaluation_version,
        )
        h.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def link_iteration_digest(instance: Any) -> str:
    """Retain ordered link identity separately; graph equality does not hide it."""
    h = hashlib.sha256()
    for n in nodes(instance):
        h.update(
            repr((n.id, [(str(b), c.id) for b, c in n.iter_child_links()])).encode()
        )
    return h.hexdigest()


def committed_neural(
    root: Path, checkpoint_format: str, rollout: bool = False
) -> tuple:
    """Fresh real model, runtime, checkpoint and closed scientific genesis."""
    args = replace(scientific_args(root), runtime_checkpoint_format=checkpoint_format)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    paths.ensure_directories()
    config = bootstrap_config_from_args(args)
    atomic_json(paths.bootstrap_config_path, bootstrap_config_to_dict(config))
    bundle = _make_model_bundle(root / "model")
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=1,
            evaluator_name="neural",
            model_bundle_path="model",
            updated_at_utc="now",
        ),
        paths.pipeline_active_model_path,
    )
    runner_args = AnemoneMorpionSearchRunnerArgs(
        search_args=default_search_args(
            rollout=MorpionBootstrapRolloutConfig(enabled=rollout, max_extra_steps=1)
        ),
        random_seed=4,
        runtime_checkpoint_format=checkpoint_format,
    )
    current = AnemoneMorpionSearchRunner(runner_args)
    current.load_or_create(None, bundle)
    current.grow(3)
    checkpoint = root / (
        "source-checkpoint" if checkpoint_format == "sharded" else "source.json.zst"
    )
    current.save_checkpoint(checkpoint)
    source = current.export_closed_scientific_snapshot(
        root / "source-science", generation=1, producer=scientific_producer(args)
    )
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        last_save_unix_s=1e9,
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
        latest_tree_snapshot_path=paths.relative_to_work_dir(source),
        metadata={"bootstrap_config_hash": bootstrap_config_sha256(config)},
    )
    state = initialize_scientific_v2_publication(
        paths,
        state,
        producer=scientific_producer(args),
        closed_scientific_snapshot=True,
    )
    return paths, state, current, runner_args


@pytest.mark.fast
@pytest.mark.parametrize("checkpoint_format", ["json-zst", "sharded"])
@pytest.mark.parametrize("rollout", [False, True])
def test_committed_neural_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_format: str,
    rollout: bool,
) -> None:
    """No attachment/version change; exact Values and next decisions survive both loaders."""
    paths, state, current, args = committed_neural(tmp_path, checkpoint_format, rollout)
    identity, pinned = committed_evaluator(tmp_path, state)
    assert identity["path"] == "model" and pinned != tmp_path / "model"
    # Mutable model discovery is neither a restore dependency nor an activation.
    paths.pipeline_active_model_path.write_text("corrupted mutable projection")
    (tmp_path / "model/param.pt").write_bytes(b"unrelated new weights")
    restored = AnemoneMorpionSearchRunner(replace(args, random_seed=999))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("A committed restore must not call set_evaluator or refresh")

    monkeypatch.setattr(restored, "_set_runtime_evaluator_from_bundle", forbidden)
    restored.load_or_create(tmp_path / state.latest_runtime_checkpoint_path, None)
    assert restored._runtime.evaluator_version == current._runtime.evaluator_version
    assert restored._current_evaluator_bundle_path == pinned
    from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
        load_closed_scientific_snapshot,
    )

    restored_source = restored.export_closed_scientific_snapshot(
        tmp_path / "restored-science",
        generation=1,
        producer=scientific_producer(
            replace(
                scientific_args(tmp_path), runtime_checkpoint_format=checkpoint_format
            )
        ),
    )
    assert load_closed_scientific_snapshot(
        restored_source
    ) == load_closed_scientific_snapshot(tmp_path / state.latest_tree_snapshot_path)
    assert fingerprint(restored) == fingerprint(current)
    assert restored._random_generator.getstate() == current._random_generator.getstate()
    # Keep ordered-link differences observable instead of normalizing them away.
    (tmp_path / "ordered-links.txt").write_text(
        str(link_iteration_digest(restored) == link_iteration_digest(current))
    )
    for _ in range(2):
        current.grow(1)
        restored.grow(1)
        assert fingerprint(restored) == fingerprint(current)
        assert _selection_signature(restored) == _selection_signature(current)
        assert (
            restored._random_generator.getstate()
            == current._random_generator.getstate()
        )
        if rollout:
            assert (
                rollout_random_generator(restored._runtime).getstate()
                == rollout_random_generator(current._runtime).getstate()
            )


@pytest.mark.fast
@pytest.mark.parametrize(
    "fault", ["missing", "corrupt", "substitute", "config", "refresh"]
)
def test_committed_dependency_failure(tmp_path: Path, fault: str) -> None:
    """Never fall back to default weights or silently activate a different model."""
    paths, state, _, args = committed_neural(tmp_path, "sharded")
    _, pinned = committed_evaluator(tmp_path, state)
    kwargs = {}
    model = None
    if fault == "missing":
        (pinned / "param.pt").unlink()
    elif fault == "corrupt":
        (pinned / "param.pt").write_bytes(b"corrupt")
    elif fault == "substitute":
        model = _make_model_bundle(tmp_path / "different-model")
    elif fault == "config":
        from chipiron.environments.morpion.bootstrap.run_state import (
            save_bootstrap_run_state,
        )

        state = replace(
            state, metadata={**state.metadata, "bootstrap_config_hash": "wrong"}
        )
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    else:
        kwargs["reevaluate_tree"] = True
    with pytest.raises((GenerationPublicationError, DerivationError)):
        AnemoneMorpionSearchRunner(args).load_or_create(
            tmp_path / state.latest_runtime_checkpoint_path, model, **kwargs
        )


@pytest.mark.fast
def test_growth_ignores_mutable_active_projection(tmp_path: Path) -> None:
    """A saved head remains active even if discovery advertises unrelated Training."""
    from tests.environments.morpion_durable_closed_helpers import (
        ClosedPublicationRunner,
        run_closed_publication,
        setup_closed_baseline,
    )

    args, paths, initial = setup_closed_baseline(tmp_path)
    model, _ = committed_evaluator(tmp_path, initial)
    paths.pipeline_active_model_path.write_text("invalid mutable model projection")
    result = run_closed_publication(args, ClosedPublicationRunner())
    assert committed_evaluator(tmp_path, result)[0] == model


@pytest.mark.parametrize("checkpoint_format", ["json-zst", "sharded"])
def test_both_loaders_use_default_bounded_resolver(
    tmp_path: Path, checkpoint_format: str
) -> None:
    """The real two restore paths inherit Anemone's default; no caller cache needed."""
    from chipiron.environments.morpion.bootstrap import (
        MorpionBootstrapEffectiveRuntimeConfig,
    )

    current = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            random_seed=4, runtime_checkpoint_format=checkpoint_format
        )
    )
    config = MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    current.load_or_create(None, None, config)
    while current.current_tree_size() < 300:
        current.grow(1)
    checkpoint = tmp_path / (
        "checkpoint" if checkpoint_format == "sharded" else "checkpoint.json.zst"
    )
    current.save_checkpoint(checkpoint)
    restored = AnemoneMorpionSearchRunner(current._args)
    restored.load_or_create(checkpoint, None, config)
    resolver = next(nodes(restored)).tree_node.state_handle.resolver
    for node in nodes(restored):
        assert node.state is not None
        assert len(resolver._resolved_states) <= 256
    assert len(resolver._resolved_states) == 256
    assert fingerprint(restored) == fingerprint(current)
