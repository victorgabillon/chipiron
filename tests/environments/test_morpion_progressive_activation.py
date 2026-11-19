"""Actual progressive Growth, cursor, mixed publication and activation recovery."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import run_pipeline_training_stage
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
    read_object,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    MODEL_ONLY_POLICY,
    POLICY,
    _runner,
    activate_model,
    activation_completion,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_activation_completion import cold_state
from tests.environments.test_morpion_prediction_origin_writes import grow, tracked
from tests.environments.test_morpion_rolling_reevaluation import (
    accepted,
    inventory,
    propose,
)
from tests.environments.test_morpion_unique_node_budget import payload


def setup(root: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    """Select the fresh public mode, with no source/authentication substitution."""
    local_leaderboard(monkeypatch, root)
    return setup_activation(
        root / "work", count=8, model_only_activation=True, **kwargs
    )


def test_switch_preserves_native_state_and_truthful_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare to a cold native set_evaluator reference, never a full refresh."""
    args, paths, _ = setup(tmp_path, monkeypatch)
    assert propose(args, 1).patch_written
    grow(args, paths)
    before = load_bootstrap_run_state(paths.run_state_path)
    origins = tracked(paths)
    cursor = accepted(paths)
    request = prepare_model_activation_request(paths, training_generation=1)
    assert request["schema_version"] == 3 and request["policy"] == MODEL_ONLY_POLICY
    reference, effective = _runner(paths, before)
    assert isinstance(reference, AnemoneMorpionSearchRunner)
    _, model = committed_evaluator(paths.work_dir, before)
    reference.load_or_create(
        paths.work_dir / before.latest_runtime_checkpoint_path, model, effective
    )
    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        load_morpion_evaluator_from_model_bundle,
    )

    with torch.random.fork_rng(devices=[]):
        evaluator = load_morpion_evaluator_from_model_bundle(
            paths.work_dir / request["candidate"]["path"]
        )
    before_payload = payload(reference)
    reference._require_runtime().set_evaluator(evaluator)
    expected = payload(reference)
    reference.discard_uncommitted_runtime()
    # Compare everything except the explicitly advanced native evaluator version.
    assert isinstance(expected, dict) and isinstance(before_payload, dict)
    changed = {key for key in expected if expected[key] != before_payload[key]}
    assert changed == {"evaluator_version"}
    with monkeypatch.context() as guard:
        guard.setattr(
            AnemoneMorpionSearchRunner,
            "refresh_model_estimates",
            lambda *a: pytest.fail("full refresh in model-only mode"),
        )
        rng = torch.get_rng_state().clone()
        after = activate_model(paths, request)
        assert torch.equal(rng, torch.get_rng_state())
    assert cold_state(paths) == expected
    assert tracked(paths).current_origins == origins.current_origins
    assert tracked(paths).creation_generations == origins.creation_generations
    assert accepted(paths) == cursor
    completion = activation_completion(
        paths, after, publication_records(after.metadata)[-1]
    )
    assert (
        completion["mode"] == "owned_cold_model_only_v1" and "refresh" not in completion
    )
    authority = paths.run_state_path.read_bytes()
    assert activate_model(paths, request) == after
    assert paths.run_state_path.read_bytes() == authority
    grown = grow(args, paths)
    current = tracked(paths)
    new_ids = set(current.creation_generations) - set(origins.creation_generations)
    assert new_ids
    assert all(
        current.current_origins[key].evaluator_sha256 == request["candidate"]["sha256"]
        for key in new_ids
        if key in current.current_origins
    )
    assert grown.generation > after.generation


def test_real_abc_mixed_dataset_training_and_finite_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted A/B/C models survive mixed sources with finite traversal."""
    args, paths, _ = setup(tmp_path, monkeypatch)
    original = inventory(paths, 1)
    a = committed_evaluator(
        paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
    )[0]
    assert a is not None
    models = [a["sha256"]]
    costs = []
    requests = []
    for index in range(2):
        started = time.perf_counter()
        assert propose(args, 1).patch_written
        packet_wall = time.perf_counter() - started
        patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
        assert patch.tree_generation == 1 and patch.rows[0].node_id == original[index]
        state = grow(args, paths)
        cursor = accepted(paths)
        assert cursor["next_node_cursor"] == original[index + 1]
        generation = 1 if index == 0 else state.generation
        if index:
            run_dataset(args, generation=generation)
            result = run_pipeline_training_stage(args, generation=generation)
            assert result.training_status == "done"
        request = prepare_model_activation_request(
            paths, training_generation=generation
        )
        requests.append(request)
        switch_started = time.perf_counter()
        activated = activate_model(paths, request)
        activation_wall = time.perf_counter() - switch_started
        assert accepted(paths) == cursor
        models.append(request["candidate"]["sha256"])
        state = grow(args, paths)
        origins = tracked(paths)
        assert {models[0], models[-1]} <= {
            o.evaluator_sha256 for o in origins.current_origins.values()
        }
        head = publication_records(state.metadata)[-1]
        header = read_object(paths.work_dir / head["tree_snapshot_path"])
        descriptor = read_object(paths.work_dir / head["descriptor_path"])
        assert (
            header["prediction_lineage"]
            == descriptor["prediction_lineage"]
            == head["prediction_lineage"]
        )
        with ShardedTrainingGenerationView(
            paths.work_dir / head["tree_snapshot_path"], allow_closed_snapshot=True
        ) as view:
            assert view.mixed_source is not None
            assert view.mixed_source.current == origins.current_origins
            view.validate_source_identity()
        costs.append({
            "round": index,
            "wall_s": time.perf_counter() - started,
            "packet_wall_s": packet_wall,
            "activation_wall_s": activation_wall,
            "accepted_source": cursor["tree_generation"],
            "next_node_cursor": cursor["next_node_cursor"],
            "direct_model_counts": {
                sha: sum(
                    o.evaluator_sha256 == sha for o in origins.current_origins.values()
                )
                for sha in models
            },
            "generation": state.generation,
            "activation_generation": activated.generation,
            "nodes": len(origins.creation_generations),
        })
    assert len(set(models)) == 3
    assert propose(args, 1000).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    assert tuple(row.node_id for row in patch.rows) == original[2:]
    grow(args, paths)
    assert accepted(paths)["completed_full_pass_count"] == 1
    assert propose(args, 1000).patch_written
    next_generation = load_reevaluation_patch(
        paths.pipeline_reevaluation_patch_path
    ).tree_generation
    assert next_generation is not None and next_generation > 1
    grow(args, paths)
    assert accepted(paths)["completed_full_pass_count"] == 2
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        artifact_digest,
    )
    from tests.environments.test_morpion_rolling_recovery_report import report_module

    state = load_bootstrap_run_state(paths.run_state_path)
    pins = {
        (paths.work_dir / record["descriptor_path"]).parent / pin["path"]: pin["sha256"]
        for record in publication_records(state.metadata)
        for pin in read_object(paths.work_dir / record["descriptor_path"])["model_pins"]
    }
    before_retirement = cold_state(paths)
    plan = report_module.analyze(paths, retirement_plan=True)
    assert plan["validated"] and 3 in plan["conditional_candidates"], plan
    outcome = report_module.retire(paths, generations=(3,))
    assert outcome["removed_files"]
    assert cold_state(paths) == before_retirement
    assert all(artifact_digest(path) == sha for path, sha in pins.items())
    assert tracked(paths).current_origins
    authority = paths.run_state_path.read_bytes()
    activate_model(paths, requests[0])
    assert paths.run_state_path.read_bytes() == authority
    costs.append({
        "retired_generation": 3,
        "retained_pins": len(pins),
        "pin_models": sorted(set(pins.values())),
        "workspace_bytes": sum(
            p.stat().st_size for p in paths.work_dir.rglob("*") if p.is_file()
        ),
    })
    (tmp_path / "observations.json").write_text(
        json.dumps({"models": models, "rounds": costs}, indent=2)
    )


@pytest.mark.parametrize("damage", ["legacy", "policy", "candidate", "numeric"])
def test_ambiguous_or_changed_requests_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Old schemas and wrong model/numeric bindings cannot select the new mode."""
    _, paths, _ = setup(tmp_path, monkeypatch)
    request = prepare_model_activation_request(paths, training_generation=1)
    if damage == "legacy":
        request["schema_version"] = 1
    elif damage == "policy":
        request["policy"] = dict(POLICY)
    elif damage == "candidate":
        request["candidate"]["path"] = "seed"
    else:
        request["policy"]["numeric"] = "unknown"
    before = paths.run_state_path.read_bytes()
    with pytest.raises(ValueError):
        activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == before


def crash(root: Path, phase: str) -> None:
    """Kill only this test process at an actual durable rename boundary."""
    paths = MorpionBootstrapPaths.from_work_dir(root)
    request = read_object(root / "activation-request.json")
    from chipiron.environments.morpion.bootstrap import (
        generation_publication as publication,
    )

    original_publish = publication.publish_directory_if_absent

    def publish(*args: Any, **kwargs: Any) -> Any:
        if phase == "before_capsule":
            os._exit(91)
        result = original_publish(*args, **kwargs)
        if phase == "after_capsule":
            os._exit(91)
        return result

    publication.publish_directory_if_absent = publish
    original = os.replace
    commits = 0

    def replace(src: Any, dst: Any, **kwargs: Any) -> None:
        nonlocal commits
        target = Path(dst)
        state = target == paths.run_state_path
        capsule = target.parent.name == "objects"
        if (state and phase == "before_commit") or (
            capsule and phase == "before_capsule"
        ):
            os._exit(91)
        original(src, dst, **kwargs)
        if state:
            commits += 1
        if (
            (state and phase == "after_commit")
            or (capsule and phase == "after_capsule")
            or (commits == 1 and phase == "after_drain")
        ):
            os._exit(91)

    os.replace = replace
    activate_model(paths, request)
    message = "crash boundary not reached"
    raise AssertionError(message)


@pytest.mark.parametrize(
    "phase",
    ["before_capsule", "after_capsule", "before_commit", "after_commit", "after_drain"],
)
def test_process_death_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Only complete old/drained/new heads survive; retry never re-applies."""
    args, paths, before = setup(tmp_path, monkeypatch)
    request = prepare_model_activation_request(paths, training_generation=1)
    if phase == "after_drain":
        assert propose(args, 1).patch_written
    (paths.work_dir / "activation-request.json").write_text(json.dumps(request))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from tests.environments.test_morpion_progressive_activation import crash; import sys; crash(Path(sys.argv[1]),sys.argv[2])",
            str(paths.work_dir),
            phase,
        ],
        timeout=60,
        check=False,
    )
    assert result.returncode == 91
    survived = load_bootstrap_run_state(paths.run_state_path)
    assert survived.generation == before.generation + (
        phase in {"after_commit", "after_drain"}
    )
    if phase == "after_drain":
        assert len(receipt_ledger(survived)) == 1
        with pytest.raises(ValueError, match="Stale"):
            activate_model(paths, request)
        request = prepare_model_activation_request(paths, training_generation=1)
    after = activate_model(paths, request)
    assert tracked(paths).current_origins
    authority = paths.run_state_path.read_bytes()
    assert activate_model(paths, request) == after
    assert paths.run_state_path.read_bytes() == authority
