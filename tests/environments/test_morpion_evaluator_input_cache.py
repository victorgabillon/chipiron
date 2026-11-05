"""Exact input-only reuse, storage ownership, and consumer lifetime controls."""

import copy
import dataclasses
import hashlib
import json
import signal
import sqlite3
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from valanga.evaluations import Value

    from chipiron.environments.morpion.types import MorpionState
from unittest.mock import patch

import pytest
import torch
from atomheart.games.morpion import initial_state
from atomheart.games.morpion.state import Variant

from chipiron.environments.morpion.players.evaluators import (
    _evaluator_input_storage as storage,
)
from chipiron.environments.morpion.players.evaluators import (
    evaluator_input_cache as cache,
)
from chipiron.environments.morpion.players.evaluators._evaluator_input_key import (
    state_payload,
)
from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionOverEventDetector,
    MorpionStateEvaluator,
)
from chipiron.environments.morpion.players.evaluators.neural_evaluator import (
    MorpionRegressorMasterEvaluator,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressorArgs,
    build_morpion_regressor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.torch_runtime import torch_execution_profile


@pytest.fixture
def evaluator() -> MorpionRegressorMasterEvaluator:
    """Build a real scalar MLP41 without changing the calling RNG."""
    over = MorpionOverEventDetector()
    with torch.random.fork_rng(devices=[]):
        model = build_morpion_regressor(
            MorpionRegressorArgs(model_kind="mlp", hidden_sizes=(4,))
        ).eval()
    return MorpionRegressorMasterEvaluator(
        evaluator=MorpionStateEvaluator(),
        over=over,
        over_detector=over,
        input_converter=MorpionFeatureTensorConverter(MorpionDynamics()),
        regressor=model,
    )


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Opt in under the real strict profile without changing threads."""
    monkeypatch.setenv(cache.ENVIRONMENT, "1")
    monkeypatch.setenv(cache.ENVIRONMENT + "_DIR", str(tmp_path / "cache"))
    # Raw-storage unit tests do not change execution threads. Actual installed
    # consumer qualification separately runs strict_math with real 1/1 threads.
    with torch_execution_profile("strict_math"):
        yield tmp_path / "cache"


def states() -> list[MorpionState]:
    """Four actual legal states with distinct complete histories."""
    dynamics = MorpionDynamics()
    state = initial_state()
    result = []
    for _ in range(4):
        result.append(dynamics.wrap_atomheart_state(state))
        state = dynamics.inner.step(
            state, dynamics.inner.all_legal_actions(state)[0]
        ).next_state
    return result


def consume(
    root: Path, evaluator: MorpionRegressorMasterEvaluator, values: list[MorpionState]
) -> tuple[Any, Any, list[bytes], list[Value]]:
    """Exercise raw conversion and every scalar forward before detachment."""
    with (
        cache.activation_inputs(root.parent) as observed,
        cache.evaluator_inputs(evaluator, len(values)) as wrapped,
    ):
        inputs = [
            wrapped.input_converter.state_to_model_input_tensors(s)[0].numpy().tobytes()
            for s in values
        ]
        scores = [wrapped.evaluate(s) for s in values]
    return wrapped, observed, inputs, scores


def test_default_off_does_not_inspect_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evaluator: MorpionRegressorMasterEvaluator,
) -> None:
    """Default off does not inspect storage."""
    monkeypatch.delenv(cache.ENVIRONMENT, raising=False)
    monkeypatch.setenv(cache.ENVIRONMENT + "_DIR", "/unusable")
    with patch.object(storage, "ownership", side_effect=AssertionError):
        with (
            cache.activation_inputs(tmp_path) as observed,
            cache.evaluator_inputs(evaluator, 1) as value,
        ):
            assert value is evaluator
        assert observed is None


def test_exact_cold_warm_new_model_and_post_close(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator
) -> None:
    """Exact cold warm new model and post close."""
    expected = [
        evaluator.input_converter.state_to_model_input_tensors(s)[0].numpy().tobytes()
        for s in states()
    ]
    cold, cold_obs, cold_raw, cold_scores = consume(enabled, evaluator, states())
    warm, warm_obs, warm_raw, warm_scores = consume(enabled, evaluator, states())
    assert cold_obs["misses"] == 8 and warm_obs["hits"] == 8
    assert cold_raw == warm_raw == expected and cold_scores == warm_scores
    assert cold.input_converter.store is warm.input_converter.store is None
    with torch.no_grad():
        evaluator.regressor.net[-1].bias.add_(4)
    _, obs, raw, scores = consume(enabled, evaluator, states())
    assert obs["hits"] == 8 and raw == expected and scores != warm_scores
    # Retained evaluators must not access pruned/closed storage after context.
    with patch.object(storage.InputStore, "lookup", side_effect=AssertionError):
        assert [warm.evaluate(s) for s in states()] == scores
        assert [
            warm.input_converter.state_to_model_input_tensors(s)[0].numpy().tobytes()
            for s in states()
        ] == expected


def test_complete_state_key_exact_fields_and_aliases() -> None:
    """Complete state key exact fields and aliases."""
    s = states()[-1]
    key = state_payload(s)
    assert key == state_payload(copy.deepcopy(s))
    assert key == state_payload(
        dataclasses.replace(s, points=frozenset(reversed(sorted(s.points))))
    )
    variants = [
        dataclasses.replace(s, played_moves=tuple(reversed(s.played_moves))),
        dataclasses.replace(s, dir_usage_entries=tuple(reversed(s.dir_usage_entries))),
        dataclasses.replace(s, moves=s.moves + 1),
        dataclasses.replace(s, is_terminal=not s.is_terminal),
        dataclasses.replace(s, variant=Variant.DISJOINT_5D),
        dataclasses.replace(s, points=frozenset((x + 1, y) for x, y in s.points)),
        dataclasses.replace(s, used_unit_segments=frozenset()),
    ]
    assert all(state_payload(v) != key for v in variants)
    with pytest.raises(TypeError):
        state_payload(dataclasses.replace(s, moves=True))


@pytest.mark.parametrize(
    "damage", ["missing", "incomplete", "checksum", "identity", "binding"]
)
def test_unusable_prior_rebuilds_before_use(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator, damage: str
) -> None:
    """Unusable prior rebuilds before use."""
    _, _, expected, scores = consume(enabled, evaluator, states())
    entry = enabled / "entry"
    if damage == "missing":
        (entry / "complete.json").unlink()
    elif damage == "incomplete":
        (entry / "inputs.sqlite").unlink()
    elif damage == "checksum":
        with (entry / "inputs.sqlite").open("ab") as f:
            f.write(b"bad")
    elif damage == "identity":
        m = json.loads((entry / "complete.json").read_text())
        m["identity"]["torch"] = "different"
        (entry / "complete.json").write_text(json.dumps(m))
    else:
        con = sqlite3.connect(entry / "inputs.sqlite")
        con.execute("UPDATE inputs SET binding='bad'")
        con.commit()
        con.close()
        m = json.loads((entry / "complete.json").read_text())
        m["sha256"] = storage.file_digest(entry / "inputs.sqlite")
        (entry / "complete.json").write_text(json.dumps(m))
    _, observed, raw, actual = consume(enabled, evaluator, states())
    assert observed["prior"] == "incompatible-or-corrupt"
    assert observed["hits"] == 0 and raw == expected and actual == scores


@pytest.mark.parametrize(
    "child",
    [
        "OWNER",
        ".lock",
        "entry",
        "entry/complete.json",
        "entry/inputs.sqlite",
        "building",
        "building/inputs.sqlite",
    ],
)
def test_every_symlink_child_refused_without_cleanup(
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
    tmp_path: Path,
    child: str,
) -> None:
    """Every symlink child refused without cleanup."""
    consume(enabled, evaluator, states())
    target = tmp_path / "foreign"
    target.write_text("untouched")
    path = enabled / child
    if path.is_dir():
        import shutil

        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.parent.mkdir(exist_ok=True)
    path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        consume(enabled, evaluator, states())
    assert target.read_text() == "untouched" and path.is_symlink()


def test_symlink_parent_and_foreign_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evaluator: MorpionRegressorMasterEvaluator,
    enabled: Path,
) -> None:
    """Symlink parent and foreign root."""
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "sentinel").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(foreign, target_is_directory=True)
    for root in [foreign, link / "nested"]:
        monkeypatch.setenv(cache.ENVIRONMENT + "_DIR", str(root))
        with pytest.raises((ValueError, OSError)):
            consume(root, evaluator, states())
    assert (foreign / "sentinel").read_text() == "keep" and not (
        foreign / "nested"
    ).exists()


def test_contention_keeps_active_reader_and_prior(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator
) -> None:
    """Contention keeps active reader and prior."""
    consume(enabled, evaluator, states())
    before = (enabled / "entry/complete.json").read_bytes()
    with storage.ownership(enabled) as owned:
        assert owned
        _, observed, _, _ = consume(enabled, evaluator, states())
        assert observed["outcome"] == "bypass-contention"
        assert (enabled / "entry/complete.json").read_bytes() == before
        assert not (enabled / "building").exists()


def test_early_budget_and_unsupported_fallback(
    monkeypatch: pytest.MonkeyPatch,
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
) -> None:
    """Early budget and unsupported fallback."""
    monkeypatch.setenv(cache.ENVIRONMENT + "_MAX_ROWS", "1")
    wrapped, obs, _, _ = consume(enabled, evaluator, states())
    assert (
        wrapped is evaluator
        and obs["outcome"].startswith("bypass")
        and not enabled.exists()
    )
    monkeypatch.delenv(cache.ENVIRONMENT + "_MAX_ROWS")
    with patch.object(storage, "resources", return_value=False):
        wrapped, obs, _, _ = consume(enabled, evaluator, states())
        assert wrapped is evaluator and obs["outcome"] == "bypass-resource-admission"
    linear = dataclasses.replace(evaluator, regressor=torch.nn.Linear(41, 1).eval())
    wrapped, obs, _, _ = consume(enabled, linear, states())
    assert wrapped is linear and not enabled.exists()


def test_late_integrity_failure_detaches_before_release(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator
) -> None:
    """Late integrity failure detaches before release."""
    consume(enabled, evaluator, states())
    with (
        pytest.raises(ValueError, match=r"completion|changed"),
        cache.activation_inputs(enabled.parent),
        cache.evaluator_inputs(evaluator, 100) as wrapped,
    ):
        wrapped.evaluate(states()[0])
        (enabled / "entry/complete.json").write_text("{}")
    assert wrapped.input_converter.store is None
    assert wrapped.evaluate(states()[0]) == evaluator.evaluate(states()[0])
    with storage.ownership(enabled) as owned:
        assert owned


@pytest.mark.parametrize("where", ["building", "replacement-gap"])
def test_sigkill_releases_lease_and_rebuilds(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator, where: str
) -> None:
    """Sigkill releases lease and rebuilds."""
    consume(enabled, evaluator, states())
    code = (
        """
import sys,os,signal,shutil
sys.path[:0]=PATHS
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from valanga.evaluations import Value
    from chipiron.environments.morpion.types import MorpionState
from chipiron.environments.morpion.players.evaluators import _evaluator_input_storage as storage
root=Path(ROOT)
with storage.ownership(root):
 (root/'building').mkdir()
 (root/'building/inputs.sqlite').write_bytes(b'incomplete')
 if WHERE=='replacement-gap':shutil.rmtree(root/'entry')
 print('ready',flush=True)
 signal.pause()
"""
        .replace("PATHS", repr(sys.path))
        .replace("ROOT", repr(str(enabled)))
        .replace("WHERE", repr(where))
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
    )
    try:
        import select

        assert select.select([child.stdout], [], [], 20)[0]
        assert child.stdout.readline().strip() == "ready"
        _, obs, _, _ = consume(enabled, evaluator, states())
        assert obs["outcome"] == "bypass-contention"
        child.kill()
        assert child.wait(timeout=10) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    _, obs, _, _ = consume(enabled, evaluator, states())
    assert obs["outcome"] == "completed" and not (enabled / "building").exists()


def test_same_runtime_growth_export_and_late_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
) -> None:
    """Real runner survives detachment; late cache failures discard private values."""
    from chipiron.environments.morpion.bootstrap.runtime import runner as rr
    from tests.environments.test_morpion_value_certainty import make_runner, values

    def hashes(path: Path) -> dict[str, bytes]:
        if path.is_file():
            return {"checkpoint": path.read_bytes()}
        return {
            str(p.relative_to(path)): p.read_bytes()
            for p in path.rglob("*")
            if p.is_file()
        }

    monkeypatch.setattr(
        rr, "load_morpion_evaluator_from_model_bundle", lambda _: evaluator
    )
    outcomes = []
    for mode in ["0", "1"]:
        monkeypatch.setenv(cache.ENVIRONMENT, mode)
        runner = make_runner(monkeypatch)
        runner.apply_effective_runtime_config(
            dataclasses.replace(
                runner._last_applied_runtime_config, tree_branch_limit=100000
            )
        )
        before_rng = torch.get_rng_state().clone()
        with cache.activation_inputs(tmp_path):
            report = runner.refresh_model_estimates(tmp_path / "unused-model-path")
        assert torch.equal(before_rng, torch.get_rng_state())
        selected = (
            runner._runtime.tree_manager.node_evaluator.master_state_value_evaluator
        )
        if mode == "1":
            assert selected.input_converter.store is None
        # Same runtime, no cold reload; exporter and Growth outlive cache storage.
        with patch.object(storage.InputStore, "lookup", side_effect=AssertionError):
            selected.evaluate(states()[0])
            runner.export_closed_scientific_snapshot(
                tmp_path / f"science-{mode}",
                generation=1,
                producer={
                    "revision": "private-input-lifetime-fixture",
                    "objective": "maximum_total_moves",
                    "evaluator": hashlib.sha256(
                        b"".join(
                            p.detach().numpy().tobytes()
                            for p in evaluator.regressor.parameters()
                        )
                    ).hexdigest(),
                    "config_sha256": hashlib.sha256(
                        repr(runner._args).encode()
                    ).hexdigest(),
                },
            )
            step = runner._runtime.step()
            assert step.nodes_added > 0
            runner.save_checkpoint(tmp_path / f"checkpoint-{mode}.json")
        outcomes.append((
            report,
            tuple(
                getattr(step, name)
                for name in (
                    "selected_node_id",
                    "selected_depth",
                    "nodes_before",
                    "nodes_after",
                    "nodes_added",
                )
            ),
            values(runner),
            hashes(tmp_path / f"checkpoint-{mode}.json"),
        ))
        runner.discard_uncommitted_runtime()
    assert outcomes[0] == outcomes[1]
    runner = make_runner(monkeypatch)
    original_finish = storage.InputStore.finish

    def damaged_finish(store: storage.InputStore) -> None:
        (store.entry / "complete.json").write_text("{}")
        original_finish(store)

    with (
        patch.object(storage.InputStore, "finish", damaged_finish),
        pytest.raises(ValueError),
        cache.activation_inputs(tmp_path),
    ):
        runner.refresh_model_estimates(tmp_path / "unused-model-path")
    assert runner._runtime is None


def test_payload_binding_corruption_mid_read(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator
) -> None:
    """An altered raw row cannot be used even after valid whole-file admission."""
    consume(enabled, evaluator, states())
    with (
        pytest.raises(ValueError, match="binding"),
        cache.activation_inputs(enabled.parent),
        cache.evaluator_inputs(evaluator, 100) as wrapped,
    ):
        con = sqlite3.connect(enabled / "entry/inputs.sqlite")
        con.execute("UPDATE inputs SET binding='bad'")
        con.commit()
        con.close()
        wrapped.evaluate(states()[0])
    assert wrapped.input_converter.store is None


def test_hard_sqlite_page_limit_does_not_publish(tmp_path: Path) -> None:
    """Full staging aborts without row truncation or a completed entry."""
    root = tmp_path / "cache"
    with storage.ownership(root):
        store = storage.InputStore(root, {"fixture": "bounded-raw"}, 16384, 100, {})
        try:
            with pytest.raises(sqlite3.DatabaseError):
                for i in range(100):
                    payload = str(i).encode() + b"a" * 8000
                    store.lookup(payload)
                    store.append(payload, bytes(41 * 4))
            assert (root / "building/inputs.sqlite").stat().st_size <= 16384
            assert not (root / "entry").exists()
        finally:
            store.close()


def test_subset_and_unsupported_state_use_original(
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
) -> None:
    """Unsupported recipes/types retain exact uncached conversion without storage."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
        MorpionFeatureSubset,
    )

    subset = MorpionFeatureSubset(
        name="j314-five", feature_names=evaluator.input_converter.feature_names()[:5]
    )
    converter = dataclasses.replace(evaluator.input_converter, feature_subset=subset)
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1)
        ).eval()
    small = dataclasses.replace(evaluator, input_converter=converter, regressor=model)
    wrapped, observed, raw, _ = consume(enabled, small, states())
    assert (
        wrapped is small
        and observed["outcome"].startswith("bypass")
        and not enabled.exists()
    )
    assert raw == [
        converter.state_to_model_input_tensors(s)[0].numpy().tobytes() for s in states()
    ]
    state = dataclasses.replace(states()[0], moves=True)
    _, observed, raw, _ = consume(enabled, evaluator, [state])
    assert (
        observed["bypassed_states"] == 2 and observed["hits"] == observed["misses"] == 0
    )
    assert raw == [
        evaluator.input_converter
        .state_to_model_input_tensors(state)[0]
        .numpy()
        .tobytes()
    ]
