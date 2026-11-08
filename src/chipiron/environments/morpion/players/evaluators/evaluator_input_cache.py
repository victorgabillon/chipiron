"""Default-off canonical flat inputs for the owned candidate refresh only.

Operational derived data, never admission or prediction authority. Every model
forward remains scalar. The transparent adapter detaches before storage closes.
"""

# Exact types deliberately exclude bool/int coercion and altered subclasses.
# pylint: disable=unidiomatic-typecheck

import hashlib
import itertools
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

import anemone
import atomheart
import torch
import valanga
from atomheart.games.morpion.dynamics import MorpionDynamics as AtomDynamics

import chipiron
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    assert_artifact_writable,
    require,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from chipiron.learning.torch_runtime import torch_execution_profile_metadata

from . import _evaluator_input_storage as storage
from ._evaluator_input_key import state_payload
from .neural_evaluator import MorpionRegressorMasterEvaluator
from .neural_networks.feature_projection import feature_indices_for_subset
from .neural_networks.feature_schema import (
    MORPION_CANONICAL_FEATURE_NAMES,
    full_morpion_feature_subset,
    morpion_feature_subset_from_name,
)
from .neural_networks.model import MorpionRegressor, MorpionRegressorArgs
from .neural_networks.state_to_tensor import MorpionFeatureTensorConverter

ENVIRONMENT = "CHIPIRON_EVALUATOR_INPUT_CACHE"
_REQUEST: ContextVar[tuple[Path, dict[str, Any]] | None] = ContextVar(
    "evaluator_input_cache", default=None
)


@contextmanager
def activation_inputs(work_dir: Path) -> Iterator[dict[str, Any] | None]:
    """Opt in only after accepted-retry/source checks, without scientific metadata."""
    setting = os.environ.get(ENVIRONMENT, "0")
    require(setting in {"0", "1"}, "Evaluator input cache must be 0 or 1")
    observation: dict[str, Any] | None = None
    request = None
    if setting == "1":
        root = Path(
            os.environ.get(
                ENVIRONMENT + "_DIR", str(work_dir / "pipeline" / "evaluator_inputs")
            )
        )
        observation = {
            "enabled": True,
            "outcome": "requested",
            "hits": 0,
            "misses": 0,
            "bypassed_states": 0,
            "key_s": 0.0,
            "lookup_s": 0.0,
            "encoding_s": 0.0,
            "write_s": 0.0,
            "projection_s": 0.0,
        }
        request = (root, observation)
    token = _REQUEST.set(request)
    try:
        yield observation
    finally:
        _REQUEST.reset(token)


def requested() -> bool:
    """Whether this call is inside an opted-in candidate refresh context."""
    return _REQUEST.get() is not None


def _limit(name: str, default: int, maximum: int | None = None) -> int:
    raw = os.environ.get(ENVIRONMENT + name, str(default))
    require(
        raw.isdecimal() and 0 < int(raw) <= (default if maximum is None else maximum),
        "Invalid evaluator cache resource limit",
    )
    return int(raw)


def _network_matches(regressor: MorpionRegressor) -> bool:
    """Admit the exact scalar linear/Linear-ReLU structure described by args."""
    args, model = regressor.args, regressor.net
    if args.model_kind == "linear":
        return (
            args.hidden_sizes is None
            and type(model) is torch.nn.Linear
            and model.in_features == args.input_dim
            and model.out_features == 1
            # Torch annotates bias as Parameter even when bias=False stores None.
            and cast("torch.nn.Parameter | None", model.bias) is not None
        )
    hidden = args.hidden_sizes
    if args.model_kind != "mlp" or type(hidden) is not tuple or not hidden:
        return False
    if (
        any(type(n) is not int or n <= 0 for n in hidden)
        or type(model) is not torch.nn.Sequential
        or len(model) != 2 * len(hidden) + 1
    ):
        return False
    widths = (args.input_dim, *hidden, 1)
    for index, (before, after) in enumerate(itertools.pairwise(widths)):
        layer = model[2 * index]
        if not (
            type(layer) is torch.nn.Linear
            and layer.in_features == before
            and layer.out_features == after
            and cast("torch.nn.Parameter | None", layer.bias) is not None
        ):
            return False
        if index < len(hidden) and type(model[2 * index + 1]) is not torch.nn.ReLU:
            return False
    return True


def _qualified(evaluator: object) -> bool:
    if (
        sys.platform != "linux"
        or type(evaluator) is not MorpionRegressorMasterEvaluator
    ):
        return False
    converter, regressor = evaluator.input_converter, evaluator.regressor
    if (
        type(regressor) is not MorpionRegressor
        or type(regressor.args) is not MorpionRegressorArgs
    ):
        return False
    return (
        type(converter) is MorpionFeatureTensorConverter
        and type(converter.dynamics) is MorpionDynamics
        and type(converter.dynamics.inner) is AtomDynamics
        and type(converter.feature_subset) is type(full_morpion_feature_subset())
        and converter.feature_subset
        in tuple(
            morpion_feature_subset_from_name(name)
            for name in (
                "handcrafted_5_core",
                "handcrafted_10_core",
                "handcrafted_20_core",
                "handcrafted_41",
            )
        )
        and regressor.args.feature_subset == converter.feature_subset
        and regressor.args.legacy_graph_tokens is None
        and not regressor.args.target_transform_enabled
        and _network_matches(regressor)
        and not regressor.training
        and not regressor.target_transform.enabled
        and all(not layer.training for layer in regressor.net.modules())
        and all(
            p.device.type == "cpu" and p.dtype == torch.float32
            for p in regressor.net.parameters()
        )
        and torch.get_default_dtype() == torch.float32
        and str(torch.get_default_device()) == "cpu"
        and torch.get_num_threads() == 1
        and bool(torch_execution_profile_metadata())
    )


def _identity() -> dict[str, object]:
    # Conservative code fence, independent of model weights, node/generation IDs,
    # Dataset admission, source paths or cached predictions. Read source only.
    digest = hashlib.sha256()
    for package in (chipiron, atomheart, anemone, valanga):
        assert package.__file__ is not None
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*.py")):
            digest.update(
                storage.encoded([package.__name__, str(path.relative_to(root))])
            )
            digest.update(path.read_bytes())
    return {
        "format": storage.FORMAT,
        "code_sha256": digest.hexdigest(),
        "python": sys.version,
        "byteorder": sys.byteorder,
        "torch": str(torch.__version__),
        "profile": torch_execution_profile_metadata(),
        "recipe": "canonical-flat-41-v1",
        "representation": "handcrafted_41",
        "features": list(MORPION_CANONICAL_FEATURE_NAMES),
        "shape": [41],
        "dtype": "float32",
        "device": "cpu",
        "layout": "contiguous",
        "state": "complete-eight-fields-marshal2-opaque-no-deserialization",
        "converter": "MorpionFeatureTensorConverter/MorpionDynamics/AtomMorpionDynamics",
    }


class _Converter:
    def __init__(
        self,
        converter: MorpionFeatureTensorConverter,
        store: storage.InputStore,
        observation: dict[str, Any],
    ) -> None:
        self.original = converter
        self.canonical = replace(
            converter, feature_subset=full_morpion_feature_subset()
        )
        self.indices = torch.tensor(
            feature_indices_for_subset(converter.feature_names()), dtype=torch.long
        )
        self.store: storage.InputStore | None = store
        self.observation = observation

    def state_to_model_input_tensors(
        self, state: MorpionState
    ) -> tuple[torch.Tensor, ...]:
        """Convert unchanged after detach; otherwise verify exact raw inputs."""
        if self.store is None:
            return self.original.state_to_model_input_tensors(state)
        started = time.perf_counter()
        try:
            payload = state_payload(state)
        except TypeError, ValueError:
            self.observation["bypassed_states"] += 1
            return self.original.state_to_model_input_tensors(state)
        self.observation["key_s"] += time.perf_counter() - started
        started = time.perf_counter()
        raw = self.store.lookup(payload)
        self.observation["lookup_s"] += time.perf_counter() - started
        if raw is None:
            started = time.perf_counter()
            result = self.canonical.state_to_model_input_tensors(state)
            self.observation["encoding_s"] += time.perf_counter() - started
            require(
                len(result) == 1
                and result[0].dtype == torch.float32
                and tuple(result[0].shape) == (41,)
                and result[0].device.type == "cpu"
                and not result[0].requires_grad
                and result[0].is_contiguous()
                and bool(torch.isfinite(result[0]).all()),
                "Unsupported evaluator cache raw input",
            )
            raw = result[0].numpy().tobytes()
            self.observation["misses"] += 1
        else:
            result = (torch.frombuffer(bytearray(raw), dtype=torch.float32).clone(),)
            self.observation["hits"] += 1
        started = time.perf_counter()
        self.store.append(payload, raw)
        self.observation["write_s"] += time.perf_counter() - started
        started = time.perf_counter()
        projected = (
            result[0]
            if self.original.input_dim == 41
            else result[0].index_select(0, self.indices)
        )
        self.observation["projection_s"] += time.perf_counter() - started
        return (projected,)


@contextmanager
def evaluator_inputs(evaluator: Any, node_count: int) -> Iterator[Any]:
    """Attach for one refresh, detach on every exit, verify before scientific save."""
    request = _REQUEST.get()
    if request is None:
        yield evaluator
        return
    root, observation = request
    requested_rows = _limit("_MAX_ROWS", 20000, maximum=35000)
    limit = _limit("_MAX_MB", 256) * 2**20
    qualified = _qualified(evaluator)
    ceiling = (
        35000
        if qualified
        and evaluator.regressor.args.model_kind == "mlp"
        and evaluator.input_converter.feature_subset
        == morpion_feature_subset_from_name("handcrafted_20_core")
        else 20000
    )
    rows = min(requested_rows, ceiling)
    observation.update(
        requested_max_rows=requested_rows,
        max_rows=rows,
        max_entry_bytes=limit,
        max_coexistence_bytes=2 * limit + 3 * storage.METADATA_BYTES,
        sqlite_page_cache_bytes=2 * 2**20,
        max_payload_bytes=storage.MAX_PAYLOAD,
    )
    if not qualified or node_count > rows:
        observation["outcome"] = "bypass-unsupported-recipe-or-node-budget"
        yield evaluator
        return
    observation.update(
        stored_representation="handcrafted_41",
        consumer_representation=evaluator.input_converter.feature_subset.name,
        consumer_model_kind=evaluator.regressor.args.model_kind,
        consumer_feature_indices=list(
            feature_indices_for_subset(evaluator.input_converter.feature_names())
        ),
    )
    require(
        root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
        "Symlinked evaluator cache path",
    )
    assert_artifact_writable(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    if not storage.resources(root, limit):
        observation["outcome"] = "bypass-resource-admission"
        yield evaluator
        return
    # ownership's finally closes the descriptor on every generator exit; the
    # exception, contention and killed-consumer controls exercise this lifetime.
    with storage.ownership(root) as owned:  # pylint: disable=contextmanager-generator-missing-cleanup
        if not owned:
            observation["outcome"] = "bypass-contention"
            yield evaluator
            return
        started = time.perf_counter()
        store = storage.InputStore(root, _identity(), limit, rows, observation)
        observation["open_verify_s"] = time.perf_counter() - started
        adapter = _Converter(evaluator.input_converter, store, observation)
        try:
            observation["outcome"] = "consuming"
            try:
                yield replace(evaluator, input_converter=adapter)
            finally:
                # The evaluator may remain installed on the SAME live runtime.
                # Detaching before close/prune also makes retained references safe.
                adapter.store = None
            started = time.perf_counter()
            store.finish()
            observation["finish_verify_s"] = time.perf_counter() - started
            observation["outcome"] = "completed"
        finally:
            adapter.store = None
            store.close()
