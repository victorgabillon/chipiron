"""Strict weight continuation inside ordinary, immutable Morpion model bundles."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
    require,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    build_morpion_regressor,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
        MorpionRegressorArgs,
    )


def candidate_training_seed(base_seed: int, name: str, generation: int) -> int:
    """Keep initialization/retry seeds independent of candidate iteration order."""
    return (
        int.from_bytes(
            hashlib.sha256(f"{base_seed}:{name}:{generation}".encode()).digest()[:8],
            "big",
        )
        % 2**63
    )


@dataclass(frozen=True, slots=True)
class MorpionTrainingContinuation:
    """An accepted caller supplies the exact parent, never just a mutable path."""

    candidate_name: str
    seed: int
    parent_bundle: str | None = None
    parent_bundle_sha256: str | None = None
    parent_weights_sha256: str | None = None
    optimizer_state_policy: str = "continue_adam"
    training_policy: str = "continue_from_previous_accepted_weights"

    def __post_init__(self) -> None:
        """Reject incomplete parent identities and unsupported continuation policies."""
        require(bool(self.candidate_name), "Candidate identity is required.")
        require(
            self.training_policy
            in {"continue_from_previous_accepted_weights", "fresh_from_scratch"},
            "Unknown candidate training policy.",
        )
        require(
            self.optimizer_state_policy in {"continue_adam", "reset_each_round"},
            "Unknown optimizer-state policy.",
        )
        identity = (
            self.parent_bundle,
            self.parent_bundle_sha256,
            self.parent_weights_sha256,
        )
        require(
            all(x is None for x in identity) or all(x is not None for x in identity),
            "Incomplete accepted parent identity.",
        )


def weights_digest(model: MorpionRegressor) -> str:
    """Hash actual named tensor values independently of Torch archive serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_state(*, cuda: bool = True) -> dict[str, Any]:
    """Record all RNGs used by these trainers without claiming cross-round replay."""
    # The seeded global RandomState returns MT19937 state; NumPy stubs disagree
    # on the no-argument overload, so retain its observed runtime tuple explicitly.
    numpy_state: Any = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch_cpu_hex": bytes(torch.get_rng_state().tolist()).hex(),
        "torch_cuda_hex": [
            bytes(s.tolist()).hex() for s in torch.cuda.get_rng_state_all()
        ]
        if cuda and torch.cuda.is_available()
        else [],
    }


def initialize_training(
    model_args: MorpionRegressorArgs,
    continuation: MorpionTrainingContinuation | None,
    *,
    output_dir: str | Path,
    device: torch.device,
    learning_rate: float,
) -> tuple[MorpionRegressor, torch.optim.Adam, dict[str, Any]]:
    """Load strict accepted weights before creating an optimizer; never fall back."""
    provenance: dict[str, Any] = {}
    parent_metadata: dict[str, Any] = {}
    if continuation is not None:
        random.seed(continuation.seed)
        np.random.seed(continuation.seed % 2**32)
        if device.type == "cpu":
            # torch.manual_seed also dispatches to accelerator generators.
            torch.default_generator.manual_seed(continuation.seed)
        else:
            torch.manual_seed(continuation.seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(continuation.seed)
        provenance = {
            "candidate_name": continuation.candidate_name,
            "training_policy": continuation.training_policy,
            "seed": continuation.seed,
            "optimizer_state_policy": continuation.optimizer_state_policy,
            "rng_policy": "deterministic_seed_reset_each_round",
            "learning_rate_policy": "constant_configured_rate",
            "rng_before_initialization": rng_state(cuda=device.type == "cuda"),
            "parent_bundle_sha256": continuation.parent_bundle_sha256,
            "parent_weights_sha256": continuation.parent_weights_sha256,
        }
    if continuation is not None and continuation.parent_bundle is not None:
        parent = Path(continuation.parent_bundle)
        require(
            artifact_digest(parent) == continuation.parent_bundle_sha256,
            "Accepted parent bundle is missing or changed.",
        )
        require(
            file_digest(parent / "param.pt") == continuation.parent_weights_sha256,
            "Accepted parent weights are missing or changed.",
        )
        model, loaded_args, manifest = load_morpion_model_bundle(parent)
        require(
            loaded_args == model_args,
            "Accepted parent architecture or features differ.",
        )
        require(
            manifest.target_kind == "backup_value",
            "Accepted parent output semantics differ.",
        )
        parent_metadata = manifest.metadata.get("continuation", {})
        require(
            parent_metadata.get("candidate_name") == continuation.candidate_name,
            "Accepted parent belongs to another candidate lineage.",
        )
        require(
            parent_metadata.get("final_weights_sha256") == weights_digest(model),
            "Accepted parent tensor identity differs.",
        )
        if continuation.training_policy == "fresh_from_scratch":
            model = build_morpion_regressor(model_args)
    else:
        model = build_morpion_regressor(model_args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if continuation is not None:
        inherits = (
            continuation.parent_bundle is not None
            and continuation.training_policy != "fresh_from_scratch"
        )
        provenance["initialization_seed"] = (
            parent_metadata["initialization_seed"] if inherits else continuation.seed
        )
        provenance["shuffle_rng_policy"] = "validation_seed_plus_epoch_index"
        provenance["optimizer_state_loaded"] = (
            inherits and continuation.optimizer_state_policy == "continue_adam"
        )
        provenance["loaded_previous_weights"] = inherits
        provenance["starting_weights_sha256"] = weights_digest(model)
        provenance["parent_optimizer_steps"] = (
            parent_metadata.get("cumulative_optimizer_steps", 0) if inherits else 0
        )
        if inherits and continuation.optimizer_state_policy == "continue_adam":
            assert continuation.parent_bundle is not None
            checkpoint = torch.load(
                Path(continuation.parent_bundle) / "optimizer.pt",
                map_location=device,
                weights_only=True,
            )
            optimizer.load_state_dict(checkpoint)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        if not inherits:
            initial = Path(output_dir) / "initial"
            require(
                not initial.exists(),
                "Initial model must be written only once per attempt.",
            )
            save_morpion_model_bundle(
                model,
                initial,
                model_args=model_args,
                metadata={
                    "candidate_name": continuation.candidate_name,
                    "seed": continuation.seed,
                },
            )
            provenance["initial_bundle_sha256"] = artifact_digest(initial)
        else:
            provenance["initial_bundle_sha256"] = parent_metadata[
                "initial_bundle_sha256"
            ]
        provenance["rng_before_training"] = rng_state(cuda=device.type == "cuda")
        provenance["starting_optimizer_step_count"] = optimizer_step_count(optimizer)
    return model, optimizer, provenance


def optimizer_step_count(optimizer: torch.optim.Adam) -> int:
    """Observe Adam's actual update counter, distinct from cumulative round work."""
    return max(
        (
            int(state["step"].item())
            for state in optimizer.state.values()
            if "step" in state
        ),
        default=0,
    )


def finish_training(
    model: MorpionRegressor,
    optimizer: torch.optim.Adam,
    provenance: dict[str, Any],
    *,
    output_dir: str | Path,
    optimizer_steps: int,
    training_examples_seen: int,
) -> dict[str, Any]:
    """Seal optimizer and measured continuation evidence in the existing bundle."""
    if not provenance:
        return {}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    final_steps = optimizer_step_count(optimizer)
    require(
        final_steps - provenance["starting_optimizer_step_count"] == optimizer_steps,
        "Measured optimizer calls disagree with Adam state.",
    )
    return {
        **provenance,
        "optimizer_steps": optimizer_steps,
        "training_examples_seen": training_examples_seen,
        "final_optimizer_step_count": final_steps,
        "cumulative_optimizer_steps": provenance["parent_optimizer_steps"]
        + optimizer_steps,
        "final_weights_sha256": weights_digest(model),
        "optimizer_sha256": file_digest(directory / "optimizer.pt"),
        "rng_after_training": rng_state(
            cuda=next(model.parameters()).device.type == "cuda"
        ),
    }
