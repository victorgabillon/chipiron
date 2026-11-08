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
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    MorpionFeatureSubset,
    morpion_feature_subset_from_name,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
    MorpionRegressorArgs,
    build_morpion_regressor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.torch_runtime import torch_execution_profile


def make_evaluator(
    subset_name: str, kind: str = "mlp"
) -> MorpionRegressorMasterEvaluator:
    """Build a real canonical scalar MLP without changing the calling RNG."""
    over = MorpionOverEventDetector()
    with torch.random.fork_rng(devices=[]):
        model = build_morpion_regressor(
            MorpionRegressorArgs(
                model_kind=kind,
                hidden_sizes=(4,) if kind == "mlp" else None,
                feature_subset_name=subset_name,
            )
        ).eval()
    return MorpionRegressorMasterEvaluator(
        evaluator=MorpionStateEvaluator(),
        over=over,
        over_detector=over,
        input_converter=MorpionFeatureTensorConverter(
            MorpionDynamics(), morpion_feature_subset_from_name(subset_name)
        ),
        regressor=model,
    )


def feature_converter(
    evaluator: MorpionRegressorMasterEvaluator,
) -> MorpionFeatureTensorConverter:
    """Narrow the fixture's converter without weakening the production protocol."""
    converter = evaluator.input_converter
    assert isinstance(converter, MorpionFeatureTensorConverter)
    return converter


@pytest.fixture(params=["handcrafted_41", "handcrafted_20_core"])
def evaluator(request: pytest.FixtureRequest) -> MorpionRegressorMasterEvaluator:
    """Run existing ownership, failure and lifetime controls for both recipes."""
    return make_evaluator(str(request.param))


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Opt in under the real strict profile without changing threads."""
    monkeypatch.setenv(cache.ENVIRONMENT, "1")
    monkeypatch.setenv(cache.ENVIRONMENT + "_DIR", str(tmp_path / "cache"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_execution_profile("strict_math"):
            yield tmp_path / "cache"
    finally:
        torch.set_num_threads(previous_threads)


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
    monkeypatch.setenv(cache.ENVIRONMENT + "_MAX_ROWS", "invalid-off-must-not-consume")
    monkeypatch.setenv(cache.ENVIRONMENT + "_MAX_MB", "invalid-off-must-not-consume")
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
    assert isinstance(evaluator.regressor, MorpionRegressor)
    assert isinstance(evaluator.regressor.net, torch.nn.Sequential)
    output = evaluator.regressor.net[-1]
    assert isinstance(output, torch.nn.Linear) and output.bias is not None
    with torch.no_grad():
        output.bias.add_(4)
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
    linear = dataclasses.replace(
        evaluator,
        regressor=torch.nn.Linear(feature_converter(evaluator).input_dim, 1).eval(),
    )
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

        assert child.stdout is not None
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
    assert isinstance(evaluator.regressor, MorpionRegressor)
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
            hashes(tmp_path / f"science-{mode}"),
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


def test_hard_sqlite_page_limit_does_not_publish(
    tmp_path: Path, evaluator: MorpionRegressorMasterEvaluator
) -> None:
    """Full staging aborts without row truncation or a completed entry."""
    root = tmp_path / "cache"
    with storage.ownership(root):
        store = storage.InputStore(root, cache._identity(), 16384, 100, {})
        try:
            with pytest.raises(sqlite3.DatabaseError):
                for i in range(100):
                    payload = str(i).encode() + b"a" * 8000
                    store.lookup(payload)
                    store.append(
                        payload, bytes(41 * 4)
                    )
            assert (root / "building/inputs.sqlite").stat().st_size <= 16384
            assert not (root / "entry").exists()
        finally:
            store.close()


def test_subset_and_unsupported_state_use_original(
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
) -> None:
    """Unsupported recipes/types retain exact uncached conversion without storage."""
    subset = MorpionFeatureSubset(
        name="j314-five", feature_names=feature_converter(evaluator).feature_names()[:5]
    )
    converter = dataclasses.replace(feature_converter(evaluator), feature_subset=subset)
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


def test_all_flat_consumers_share_canonical_rows(enabled: Path) -> None:
    """Eight consumers switch weights/subsets through ONE exact canonical entry."""
    previous = None
    values = states()
    calls: list[torch.Size] = []
    original_forward = MorpionRegressor.forward

    def forward(model: MorpionRegressor, *inputs: Any) -> Any:
        calls.append(inputs[0].shape)
        return original_forward(model, *inputs)

    for name in (
        "handcrafted_41",
        "handcrafted_5_core",
        "handcrafted_20_core",
        "handcrafted_10_core",
    ):
        for kind in ("mlp", "linear"):
            evaluator = make_evaluator(name, kind)
            expected = [evaluator.evaluate(s) for s in values]
            expected_raw = [
                feature_converter(evaluator).state_to_tensor(s).numpy().tobytes()
                for s in values
            ]
            assert isinstance(evaluator.regressor, MorpionRegressor)
            calls.clear()

            with patch.object(MorpionRegressor, "forward", forward):
                wrapped, observed, raw, scores = consume(enabled, evaluator, values)
            assert len(calls) == len(values)
            assert raw == expected_raw and scores == expected
            assert observed["stored_representation"] == "handcrafted_41"
            assert observed["consumer_representation"] == name
            assert observed["consumer_model_kind"] == kind
            assert observed["consumer_feature_indices"] == list(
                cache.feature_indices_for_subset(
                    feature_converter(evaluator).feature_names()
                )
            )
            assert observed["hits"] == (8 if previous else 0)
            manifest = json.loads((enabled / "entry/complete.json").read_text())
            identity = manifest["identity"]
            assert (
                identity["shape"] == [41]
                and identity["recipe"] == "canonical-flat-41-v1"
            )
            if previous is not None:
                assert identity == previous and observed["prior"] == "verified"
            previous = identity
            assert wrapped.input_converter.store is None
            net = evaluator.regressor.net
            output = net[-1] if isinstance(net, torch.nn.Sequential) else net
            assert isinstance(output, torch.nn.Linear) and output.bias is not None
            with torch.no_grad():
                output.bias.add_(7)
            changed = [evaluator.evaluate(s) for s in values]
            _, observed, _, actual = consume(enabled, evaluator, values)
            assert observed["hits"] == 8 and changed == actual and changed != scores
            with sqlite3.connect(enabled / "entry/inputs.sqlite") as connection:
                assert connection.execute(
                    "SELECT DISTINCT length(raw) FROM inputs"
                ).fetchall() == [(164,)]
            assert {child.name for child in enabled.iterdir()} == {
                "OWNER",
                ".lock",
                "entry",
            }


@pytest.mark.parametrize("damage", ["shape", "order", "name", "dtype", "legacy"])
def test_storage_recipe_refuses_wrong_shape_or_order(
    evaluator: MorpionRegressorMasterEvaluator, damage: str
) -> None:
    """Storage validates canonical representation metadata before opening rows."""
    identity = cache._identity()
    if damage == "shape":
        identity["shape"] = [20]
    elif damage == "order":
        identity["features"] = list(
            reversed(feature_converter(evaluator).feature_names())
        )
    elif damage == "name":
        identity["representation"] = "custom"
    elif damage == "dtype":
        identity["dtype"] = "float64"
    else:
        del identity["representation"]
    with pytest.raises(ValueError, match="representation"):
        storage._input_width(identity)


def test_wrong_width_row_with_valid_digest_is_refused(
    evaluator: MorpionRegressorMasterEvaluator,
) -> None:
    """A valid checksum cannot authorize another representation's raw width."""
    width = 41
    recipe = storage.encoded(cache._identity())
    payload = state_payload(states()[0])
    raw = bytes((41 if width == 20 else 20) * 4)
    row = (
        hashlib.sha256(payload).hexdigest(),
        payload,
        raw,
        storage.binding(recipe, payload, raw),
    )
    with pytest.raises(ValueError, match="binding"):
        storage.valid_row(row, recipe, width)


@pytest.mark.parametrize("damage", ["format", "legacy"])
def test_old_recipe_is_rebuilt_before_consumption(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator, damage: str
) -> None:
    """Owned old storage remains disposable without changing ownership format."""
    _, _, expected, scores = consume(enabled, evaluator, states())
    path = enabled / "entry/complete.json"
    manifest = json.loads(path.read_text())
    if damage == "format":
        manifest["identity"]["format"] = "foreign-input-format"
    else:
        del manifest["identity"]["representation"]
    path.write_text(json.dumps(manifest))
    _, observed, actual, actual_scores = consume(enabled, evaluator, states())
    assert observed["prior"] == "incompatible-or-corrupt"
    assert observed["hits"] == 0 and actual == expected and actual_scores == scores


def test_custom_subset_subclass_and_wrong_network_width_bypass(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator
) -> None:
    """Exact types, canonical names and network width remain admission fences."""

    class DerivedConverter(MorpionFeatureTensorConverter):
        """A deliberately unsupported converter subclass."""

    converter = feature_converter(evaluator)
    custom = dataclasses.replace(
        converter,
        feature_subset=MorpionFeatureSubset("custom", converter.feature_names()),
    )
    derived = DerivedConverter(converter.dynamics, converter.feature_subset)
    other = make_evaluator(
        "handcrafted_41" if converter.input_dim == 20 else "handcrafted_20_core"
    )
    with torch.random.fork_rng(devices=[]):
        custom_model = build_morpion_regressor(
            MorpionRegressorArgs(
                model_kind="mlp",
                hidden_sizes=(4,),
                feature_subset_name="custom",
                feature_names=converter.feature_names(),
            )
        ).eval()
    for value in [
        dataclasses.replace(evaluator, input_converter=custom),
        dataclasses.replace(evaluator, input_converter=derived),
        dataclasses.replace(evaluator, regressor=other.regressor),
        dataclasses.replace(evaluator, regressor=custom_model),
    ]:
        with (
            cache.activation_inputs(enabled.parent) as observed,
            cache.evaluator_inputs(value, 4) as wrapped,
        ):
            assert wrapped is value
        assert observed is not None and observed["outcome"].startswith("bypass")
    assert not enabled.exists()


def test_public_caps_and_coexistence_unchanged(
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent row settings still bypass above20k; byte/reserve bounds are unchanged."""
    with (
        patch.object(storage, "ownership", side_effect=AssertionError),
        cache.activation_inputs(enabled.parent) as observed,
        cache.evaluator_inputs(evaluator, 20001) as wrapped,
    ):
        assert wrapped is evaluator
    assert observed is not None
    assert observed["outcome"] == "bypass-unsupported-recipe-or-node-budget"
    assert observed["max_rows"] == 20000
    assert observed["max_entry_bytes"] == 256 * 2**20
    assert observed["max_coexistence_bytes"] == 2 * 256 * 2**20 + 3 * 65536
    assert not enabled.exists()
    for suffix, setting in [("_MAX_ROWS", "35001"), ("_MAX_MB", "257")]:
        with monkeypatch.context() as changed:
            changed.setenv(cache.ENVIRONMENT + suffix, setting)
            with (
                pytest.raises(ValueError, match="resource limit"),
                cache.activation_inputs(enabled.parent),
                cache.evaluator_inputs(evaluator, 1),
            ):
                pass


@pytest.mark.fast
@pytest.mark.parametrize(
    ("setting", "node_count", "mlp20_limit", "mlp41_limit"),
    [
        (None, 20000, 20000, 20000),
        (None, 20001, 20000, 20000),
        ("20000", 20000, 20000, 20000),
        ("20000", 20001, 20000, 20000),
        ("20001", 20001, 20001, 20000),
        ("35000", 20000, 35000, 20000),
        ("35000", 20001, 35000, 20000),
        ("35000", 35000, 35000, 20000),
        ("35000", 35001, 35000, 20000),
        ("3", 3, 3, 3),
        ("3", 4, 3, 3),
    ],
)
def test_per_recipe_row_admission(
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
    monkeypatch: pytest.MonkeyPatch,
    setting: str | None,
    node_count: int,
    mlp20_limit: int,
    mlp41_limit: int,
) -> None:
    """Declared-count boundaries with one real input, not a 35k-tree qualification."""
    if setting is None:
        monkeypatch.delenv(cache.ENVIRONMENT + "_MAX_ROWS", raising=False)
    else:
        monkeypatch.setenv(cache.ENVIRONMENT + "_MAX_ROWS", setting)
    rows = mlp20_limit if feature_converter(evaluator).input_dim == 20 else mlp41_limit
    admitted = node_count <= rows
    with (
        cache.activation_inputs(enabled.parent) as observed,
        cache.evaluator_inputs(evaluator, node_count) as wrapped,
    ):
        assert (wrapped is not evaluator) == admitted
        if admitted:
            raw = wrapped.input_converter.state_to_model_input_tensors(states()[0])
            expected = evaluator.input_converter.state_to_model_input_tensors(
                states()[0]
            )
            assert len(raw) == len(expected) == 1
            assert torch.equal(raw[0], expected[0])
    assert observed is not None
    assert observed["max_rows"] == rows
    assert observed["requested_max_rows"] == int(setting or "20000")
    assert observed["max_entry_bytes"] == 256 * 2**20
    assert observed["max_coexistence_bytes"] == 537067520
    assert observed["outcome"] == (
        "completed" if admitted else "bypass-unsupported-recipe-or-node-budget"
    )
    assert enabled.exists() == admitted


@pytest.mark.parametrize(
    "setting", ["", "0", "-1", "35001", "35_000", "35000.0", "nan", "True"]
)
def test_invalid_row_setting_refuses_before_storage(
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    """Reject invalid global requests instead of silently clamping either recipe."""
    monkeypatch.setenv(cache.ENVIRONMENT + "_MAX_ROWS", setting)
    with (
        patch.object(storage, "ownership", side_effect=AssertionError),
        pytest.raises(ValueError, match="resource limit"),
        cache.activation_inputs(enabled.parent),
        cache.evaluator_inputs(evaluator, 1),
    ):
        pass
    assert not enabled.exists()


def test_reduced_limit_counts_requests_not_unique_inputs(
    enabled: Path,
    evaluator: MorpionRegressorMasterEvaluator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated inputs exhaust the same effective request limit before publication."""
    monkeypatch.setenv(cache.ENVIRONMENT + "_MAX_ROWS", "2")
    with (
        pytest.raises(ValueError, match="request budget"),
        cache.activation_inputs(enabled.parent),
        cache.evaluator_inputs(evaluator, 1) as wrapped,
    ):
        for _ in range(3):
            wrapped.input_converter.state_to_model_input_tensors(states()[0])
    assert wrapped.input_converter.store is None
    assert not (enabled / "entry").exists()


@pytest.mark.parametrize("damage", ["dtype", "device", "training"])
def test_unqualified_model_runtime_bypasses_without_storage(
    enabled: Path, evaluator: MorpionRegressorMasterEvaluator, damage: str
) -> None:
    """Unsupported tensor/runtime modes bypass without initializing a GPU."""
    assert isinstance(evaluator.regressor, MorpionRegressor)
    if damage == "dtype":
        evaluator.regressor.double()
    elif damage == "device":
        evaluator.regressor.to("meta")
    else:
        evaluator.regressor.train()
    with (
        cache.activation_inputs(enabled.parent) as observed,
        cache.evaluator_inputs(evaluator, 4) as wrapped,
    ):
        assert wrapped is evaluator
    assert observed is not None and observed["outcome"].startswith("bypass")
    assert not enabled.exists() and not torch.cuda.is_initialized()


def test_missing_strict_profile_bypasses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evaluator: MorpionRegressorMasterEvaluator,
) -> None:
    """Enabling storage alone cannot establish or guess a scientific profile."""
    monkeypatch.setenv(cache.ENVIRONMENT, "1")
    monkeypatch.setenv(cache.ENVIRONMENT + "_DIR", str(tmp_path / "cache"))
    assert not cache.torch_execution_profile_metadata()
    with (
        cache.activation_inputs(tmp_path) as observed,
        cache.evaluator_inputs(evaluator, 4) as wrapped,
    ):
        assert wrapped is evaluator
    assert observed is not None and observed["outcome"].startswith("bypass")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("width", [5, 10, 20, 41])
def test_shared_storage_keeps_consumer_node_ceiling(
    enabled: Path, monkeypatch: pytest.MonkeyPatch, kind: str, width: int
) -> None:
    """Canonical storage width does not authorize another consumer's capacity."""
    name = "handcrafted_41" if width == 41 else f"handcrafted_{width}_core"
    evaluator = make_evaluator(name, kind)
    monkeypatch.setenv(cache.ENVIRONMENT + "_MAX_ROWS", "35000")
    with (
        cache.activation_inputs(enabled.parent) as observed,
        cache.evaluator_inputs(evaluator, 20001) as wrapped,
    ):
        assert (wrapped is not evaluator) == (kind == "mlp" and width == 20)
    assert observed is not None
    assert observed["max_rows"] == (35000 if kind == "mlp" and width == 20 else 20000)


@pytest.mark.parametrize("old_width", [20, 41])
def test_legacy_flat_entries_are_rebuilt(enabled: Path, old_width: int) -> None:
    """Even checksummed old20/41 rows must not become canonical entries by size."""
    evaluator = make_evaluator("handcrafted_20_core")
    _, _, expected, scores = consume(enabled, evaluator, states())
    path = enabled / "entry/complete.json"
    manifest = json.loads(path.read_text())
    identity = manifest["identity"]
    del identity["recipe"]
    old_name = "handcrafted_41" if old_width == 41 else "handcrafted_20_core"
    identity.update(
        representation=old_name,
        shape=[old_width],
        features=list(morpion_feature_subset_from_name(old_name).feature_names),
    )
    recipe = storage.encoded(identity)
    database = enabled / "entry/inputs.sqlite"
    with sqlite3.connect(database) as connection:
        for key, payload, raw in connection.execute(
            "SELECT key,payload,raw FROM inputs"
        ).fetchall():
            raw = raw[: old_width * 4]
            connection.execute(
                "UPDATE inputs SET raw=?,binding=? WHERE key=?",
                (raw, storage.binding(recipe, payload, raw), key),
            )
    manifest["sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    _, observed, actual, actual_scores = consume(enabled, evaluator, states())
    assert observed["prior"] == "incompatible-or-corrupt" and observed["hits"] == 0
    assert actual == expected and actual_scores == scores


@pytest.mark.parametrize(
    "damage",
    ["extra_relu", "wrong_hidden", "wrong_output", "normalized", "linear_args"],
)
def test_exact_network_and_args_gate(enabled: Path, damage: str) -> None:
    """Canonical names do not admit altered structures or normalized models."""
    evaluator = make_evaluator("handcrafted_20_core")
    model = evaluator.regressor
    assert isinstance(model, MorpionRegressor) and isinstance(
        model.net, torch.nn.Sequential
    )
    if damage == "extra_relu":
        model.net.append(torch.nn.ReLU())
    elif damage == "wrong_hidden":
        model.args = dataclasses.replace(model.args, hidden_sizes=(8,))
    elif damage == "wrong_output":
        model.net[-1] = torch.nn.Linear(4, 2).eval()
    elif damage == "normalized":
        model.args = dataclasses.replace(model.args, target_transform_enabled=True)
    else:
        model.args = dataclasses.replace(
            model.args, model_kind="linear", hidden_sizes=None
        )
    with (
        cache.activation_inputs(enabled.parent) as observed,
        cache.evaluator_inputs(evaluator, 1) as wrapped,
    ):
        assert wrapped is evaluator
    assert observed is not None and observed["outcome"].startswith("bypass")
    assert not enabled.exists()


def test_projected_tensors_do_not_alias_cached_bytes(enabled: Path) -> None:
    """Mutating a consumer tensor cannot poison a subsequent cache hit."""
    evaluator = make_evaluator("handcrafted_5_core", "linear")
    consume(enabled, evaluator, states())
    with (
        cache.activation_inputs(enabled.parent),
        cache.evaluator_inputs(evaluator, 4) as wrapped,
    ):
        value = states()[0]
        first = wrapped.input_converter.state_to_model_input_tensors(value)[0]
        first.add_(123)
        second = wrapped.input_converter.state_to_model_input_tensors(value)[0]
        assert torch.equal(second, feature_converter(evaluator).state_to_tensor(value))
        assert not second.requires_grad and first.data_ptr() != second.data_ptr()
