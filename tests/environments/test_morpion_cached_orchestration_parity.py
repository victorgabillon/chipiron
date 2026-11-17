"""Exact CPU parity against the frozen pre-consolidation flat loop.

The oracle predates the implementation edit; it never calls the packed loop.
A logical clock compares timing scope/order, not unpredictable wall time.
"""

import copy
import itertools
import logging
import random
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    MORPION_CANONICAL_FEATURE_NAMES,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
    MorpionRegressorArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionTrainingArgs,
    entity_token_cache_streaming,
    flat_cache_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.flat_tensor_cache import (
    FlatTensorCache,
    FlatTensorCacheManifest,
    FlatTensorCachePaths,
    flat_cache_batch,
)
from chipiron.learning import timing
from chipiron.learning.supervised import cached, regression
from chipiron.learning.torch_runtime import torch_execution_profile
from tests.environments import morpion_flat_cache_baseline

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from chipiron.learning.supervised import TensorSupervisedBatch

FEATURE_INDICES = (0, 3, 7)
FEATURES = tuple(MORPION_CANONICAL_FEATURE_NAMES[i] for i in FEATURE_INDICES)


def _cache(tmp_path: Path) -> FlatTensorCache:
    """Read a tiny private mmap cache; neither loop may change its bytes/tensors."""
    width = len(MORPION_CANONICAL_FEATURE_NAMES)
    inputs = torch.arange(13 * width, dtype=torch.float32).reshape(13, width) / 100
    targets = torch.linspace(-1, 1, 13).reshape(13, 1)
    path = tmp_path / "cache.pt"
    torch.save({"input": inputs, "target": targets}, path)
    path.chmod(0o444)
    payload = torch.load(path, mmap=True, weights_only=True)
    return FlatTensorCache(
        paths=FlatTensorCachePaths(path, tmp_path / "unused.manifest.json"),
        manifest=FlatTensorCacheManifest(
            format="morpion_flat_features_v1",
            source_rows_path="synthetic",
            source_rows_size=0,
            source_rows_mtime=0.0,
            row_count=13,
            feature_names=MORPION_CANONICAL_FEATURE_NAMES,
            input_shape=(13, width),
            target_shape=(13, 1),
            dtype="float32",
            created_at_unix_s=0.0,
        ),
        input_tensor=payload["input"],
        target_tensor=payload["target"],
        rebuilt=False,
        materialize_seconds=0.0,
        load_seconds=0.0,
    )


def _assert_exact(actual: Any, expected: Any) -> None:
    """Require exact values, including every Adam moment and RNG byte."""
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            _assert_exact(value, expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_exact(value, reference)
    else:
        assert actual == expected


def _run(
    module: ModuleType,
    args: MorpionTrainingArgs,
    cache: FlatTensorCache,
    *,
    max_rows: int | None,
    chunk_size: int,
    synchronize: bool,
    callback: bool,
) -> dict[str, Any]:
    """Observe two epochs with a real Adam state round-trip between them."""
    trace: list[object] = []
    snapshots: list[object] = []
    tick = itertools.count()
    original_add = timing.PhaseDurations.add_duration
    with pytest.MonkeyPatch.context() as patch, torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(519)
        model = MorpionRegressor(
            MorpionRegressorArgs(
                feature_subset_name="parity_subset", feature_names=FEATURES
            )
        )
        model.net = torch.nn.Sequential(
            torch.nn.Linear(3, 5),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.35),
            torch.nn.Linear(5, 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        # Leave nonzero gradients: the first update must clear them too.
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        model.eval()

        def clock() -> float:
            return float(next(tick))

        def duration(self: timing.PhaseDurations, phase: str, seconds: float) -> None:
            trace.append(("phase", phase, seconds))
            original_add(self, phase, seconds)

        def sync(device: torch.device) -> None:
            trace.append(("sync", str(device)))

        def build(
            *,
            cache: FlatTensorCache,
            row_indices: tuple[int, ...],
            requested_feature_names: tuple[str, ...],
        ) -> TensorSupervisedBatch:
            trace.append((
                "batch",
                row_indices,
                model.training,
                torch.is_grad_enabled(),
            ))
            batch = flat_cache_batch(
                cache=cache,
                row_indices=row_indices,
                requested_feature_names=requested_feature_names,
            )
            assert torch.equal(
                batch.get_input_layer(),
                cache.input_tensor[list(row_indices)][:, FEATURE_INDICES],
            )
            return batch

        def progress(chunk: int, rows: int, epoch: int, epochs: int) -> None:
            steps = tuple(float(s["step"]) for s in optimizer.state.values())
            trace.append((
                "progress",
                chunk,
                rows,
                epoch,
                epochs,
                steps,
                model.training,
            ))

        def log(_self: logging.Logger, record: logging.LogRecord) -> None:
            trace.append(("log", record.name, record.getMessage()))

        # The frozen module has a test-module name; use its original logger identity.
        if module is morpion_flat_cache_baseline:
            patch.setattr(
                module, "LOGGER", logging.getLogger(flat_cache_streaming.__name__)
            )
        patch.setattr(module.LOGGER, "level", logging.INFO)
        patch.setattr(logging.Logger, "handle", log)
        for target in (
            module,
            entity_token_cache_streaming,
            cached,
            timing,
            regression,
        ):
            patch.setattr(target, "perf_counter", clock, raising=False)
        for target in (module, entity_token_cache_streaming, cached, regression):
            patch.setattr(
                target, "synchronize_torch_device_if_needed", sync, raising=False
            )
        patch.setattr(timing.PhaseDurations, "add_duration", duration)
        patch.setattr(module, "flat_cache_batch", build)
        try:
            for epoch in (0, 1):
                stats = module.train_flat_cache_streaming_epoch(
                    model=model,
                    optimizer=optimizer,
                    criterion=torch.nn.MSELoss(),
                    args=args,
                    cache=cache,
                    row_chunk_size=chunk_size,
                    max_rows=max_rows,
                    epoch_index=epoch,
                    progress_callback=progress if callback else None,
                    device=torch.device("cpu"),
                    synchronize_cuda=synchronize,
                )
                metrics = [
                    module.evaluate_flat_cache_streaming_metrics(
                        model=model,
                        args=args,
                        cache=cache,
                        max_rows=max_rows,
                        split=split,
                        device=torch.device("cpu"),
                        synchronize_cuda=synchronize,
                    )
                    for split in ("train", "validation")
                ]
                snapshots.append(
                    copy.deepcopy((
                        asdict(stats),
                        [asdict(m) for m in metrics],
                        model.state_dict(),
                        optimizer.state_dict(),
                        torch.get_rng_state(),
                        random.getstate(),
                        [p.grad for p in model.parameters()],
                        model.training,
                    ))
                )
                # Rebind the restored optimizer to new live parameters, as continuation does.
                state = copy.deepcopy(optimizer.state_dict())
                model = copy.deepcopy(model)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
                optimizer.load_state_dict(state)
        except (ValueError, ZeroDivisionError) as error:
            snapshots.append((type(error).__name__, str(error), model.training))
        return {"snapshots": snapshots, "trace": trace, "rng": torch.get_rng_state()}


@pytest.mark.parametrize("synchronize", [True, False])
@pytest.mark.parametrize("callback", [True, False])
@pytest.mark.parametrize(
    ("shuffle", "fraction", "validation", "max_rows", "batch_size", "chunk_size"),
    [
        (True, 0.2, None, None, 4, 3),
        (False, 0.2, None, 11, 4, 20),
        (True, 0.2, frozenset({1, 4, 9, 20}), 11, 4, 2),
        (True, 0.0, None, None, 5, 1),
        (True, 0.2, frozenset(range(13)), None, 4, 3),
        (True, 0.2, None, 0, 4, 3),
        (True, 0.2, None, None, 0, 3),
        (True, 0.2, None, None, 4, 0),
    ],
)
def test_cached_orchestration_exact_baseline_parity(
    tmp_path: Path,
    shuffle: bool,
    fraction: float,
    validation: frozenset[int] | None,
    max_rows: int | None,
    batch_size: int,
    chunk_size: int,
    synchronize: bool,
    callback: bool,
) -> None:
    """Preserve indices, updates, dropout, continuation, metrics and timed effects."""
    cache = _cache(tmp_path)
    before = cache.paths.tensor_path.read_bytes()
    inputs, targets = cache.input_tensor.clone(), cache.target_tensor.clone()
    args = MorpionTrainingArgs(
        tmp_path / "unused.jsonl",
        tmp_path / "unused-output",
        feature_subset_name="parity_subset",
        feature_names=FEATURES,
        num_epochs=2,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_fraction=fraction,
        validation_seed=37,
        validation_indices=validation,
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_execution_profile("strict_math"):
            expected = _run(
                morpion_flat_cache_baseline,
                args,
                cache,
                max_rows=max_rows,
                chunk_size=chunk_size,
                synchronize=synchronize,
                callback=callback,
            )
            actual = _run(
                flat_cache_streaming,
                args,
                cache,
                max_rows=max_rows,
                chunk_size=chunk_size,
                synchronize=synchronize,
                callback=callback,
            )
        _assert_exact(actual, expected)
        # Independent schedule expectation: the frozen loop imports public helpers,
        # which will themselves be extracted. Do not let a shared scheduling bug
        # turn the new schedule into its own reference.
        if batch_size > 0 and chunk_size > 0:
            limit = 13 if max_rows is None else min(13, max_rows)
            held_out = [
                i
                for i in range(limit)
                if (
                    i in validation
                    if validation is not None
                    else fraction > 0 and (i + 1) % max(2, round(1 / fraction)) == 0
                )
            ]
            training = [i for i in range(limit) if i not in held_out]
            expected_events: list[tuple[str, tuple[int, ...], bool, bool]] = []
            for epoch in (0, 1):
                shuffled = list(training)
                if shuffle:
                    random.Random(37 + epoch).shuffle(shuffled)
                for indices, training_mode in (
                    (shuffled, True),
                    (training, False),
                    (held_out, False),
                ):
                    expected_events.extend(
                        (
                            "batch",
                            tuple(indices[start : start + batch_size]),
                            training_mode,
                            training_mode,
                        )
                        for start in range(0, len(indices), batch_size)
                    )
            assert [
                event for event in actual["trace"] if event[0] == "batch"
            ] == expected_events
        for event in actual["trace"]:
            if event[0] == "log":
                assert event[1] == flat_cache_streaming.__name__
                if validation is not None:
                    assert "split_policy=canonical_state_sha256_v1" in event[2]
        if batch_size > 0 and chunk_size > 0:
            assert len(actual["snapshots"]) == 2
        else:
            assert actual["snapshots"][0][0] in {"ValueError", "ZeroDivisionError"}
        assert cache.paths.tensor_path.read_bytes() == before
        assert torch.equal(cache.input_tensor, inputs)
        assert torch.equal(cache.target_tensor, targets)
    finally:
        torch.set_num_threads(old_threads)
