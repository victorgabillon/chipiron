"""Resumable value training over canonical rows with optional state augmentation.

This is reusable training infrastructure. Candidate selection and experiment
orchestration belong outside the package. Validation always uses canonical
states and the deterministic historical row split.
"""
# ruff: noqa: TRY003

from __future__ import annotations

import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader, Sampler

from chipiron.environments.morpion.players.evaluators.datasets.augmentation import (
    AugmentedMorpionDataset,
    CanonicalStates,
)
from chipiron.environments.morpion.players.evaluators.datasets.datasets import (
    collate_morpion_relational_entity_token_supervised_samples,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressorArgs,
    build_morpion_regressor,
)
from chipiron.environments.morpion.symmetry import transform_morpion_state
from chipiron.learning.supervised import TensorSupervisedBatch, regression_quality_stats

from .cached_index_schedule import cached_index_schedule, shuffled_epoch_train_indices

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )


@dataclass(frozen=True)
class EntityValueTrainingConfig:
    """Explicit training recipe; it does not change legacy production defaults."""

    model: MorpionRegressorArgs
    seed: int = 0
    d4_augmentation: bool = False
    num_epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    minimum_learning_rate_ratio: float = 0.01
    workers: int = 2
    device: str = "cuda"
    symmetry_diagnostic_rows: int = 128

    def __post_init__(self) -> None:
        """Reject invalid recipes before opening outputs or constructing models."""
        if (
            self.model.model_kind
            != "relation_biased_entity_token_transformer_value_net"
        ):
            raise ValueError("Value training requires a relational entity model.")
        if (
            self.num_epochs < 1
            or self.batch_size < 1
            or self.workers < 0
            or self.symmetry_diagnostic_rows < 0
        ):
            raise ValueError("Invalid epoch/batch/worker/diagnostic count.")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid learning rate or weight decay.")
        if (
            not 0 <= self.warmup_fraction < 1
            or not 0 <= self.minimum_learning_rate_ratio <= 1
        ):
            raise ValueError("Invalid warmup or minimum learning rate ratio.")


def learning_rate_at_step(
    config: EntityValueTrainingConfig, step: int, total: int
) -> float:
    """Historical linear warmup then cosine, applied before each optimizer step."""
    warmup = int(total * config.warmup_fraction)
    if step < warmup:
        return config.learning_rate * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    ratio = config.minimum_learning_rate_ratio
    return config.learning_rate * (
        ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2
    )


def atomic_json(path: Path, payload: object) -> None:
    """Publish finite JSON only after its complete bytes reach disk."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class EpochRowSampler(Sampler[tuple[int, int]]):
    """Carry epoch and row identity to workers, including after a mid-epoch resume."""

    def __init__(self, indices: tuple[int, ...], seed: int) -> None:
        """Keep the immutable split independent of augmentation and shuffling."""
        self.indices = indices
        self.seed = seed
        self.epoch = -1
        self.start = 0

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Yield canonical validation or the historical global training shuffle."""
        indices = (
            self.indices
            if self.epoch < 0
            else shuffled_epoch_train_indices(
                train_indices=self.indices,
                shuffle=True,
                validation_seed=self.seed,
                epoch_index=self.epoch,
            )
        )
        return iter((index, self.epoch) for index in indices[self.start :])

    def __len__(self) -> int:
        """Return the number of still-requested rows in this epoch."""
        return len(self.indices) - self.start


def _worker_init(_worker_id: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _loader(
    dataset: AugmentedMorpionDataset,
    sampler: EpochRowSampler,
    config: EntityValueTrainingConfig,
) -> DataLoader[TensorSupervisedBatch]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_morpion_relational_entity_token_supervised_samples,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
        multiprocessing_context="spawn" if config.workers else None,
        generator=torch.Generator().manual_seed(config.seed),
        worker_init_fn=_worker_init,
    )


def _inputs(
    batch: TensorSupervisedBatch, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in batch.get_model_input_tensors())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def symmetry_consistency(
    model: MorpionRegressor,
    dataset: AugmentedMorpionDataset,
    indices: tuple[int, ...],
    device: torch.device,
) -> dict[str, object]:
    """Measure all eight views on a fixed subset without changing MSE predictions."""
    spreads = []
    deviations = []
    model.eval()
    with torch.inference_mode():
        for index in indices:
            state, _ = dataset.rows[index]
            predictions = []
            for symmetry in range(8):
                tensors = dataset.converter.state_to_tensors(
                    transform_morpion_state(state, symmetry)
                )
                predictions.append(
                    model(
                        tensors.token_tensor.to(device),
                        tensors.relation_triples.to(device),
                    )
                    .reshape(())
                    .cpu()
                )
            values = torch.stack(predictions)
            spreads.append(float(values.std(unbiased=False)))
            deviations.append(float(values.max() - values.min()))
    return {
        "row_indices": list(indices),
        "all_eight_transforms": True,
        "population_std_mean": sum(spreads) / len(spreads) if spreads else None,
        "range_mean": sum(deviations) / len(deviations) if deviations else None,
        "range_max": max(deviations, default=None),
    }


def train_entity_value(
    config: EntityValueTrainingConfig,
    states: CanonicalStates,
    output_dir: Path,
    *,
    provenance: dict[str, object],
    progress: Callable[[str], None] = print,
    stop_after_steps: int | None = None,
) -> dict[str, object]:
    """Train, resume, and save a normal bundle, with canonical final validation.

    ``stop_after_steps`` is for bounded pilots and restart verification only;
    it produces an interrupted checkpoint, never a completed scientific result.
    SIGINT safely retains the most recent fully applied optimizer step.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    split = cached_index_schedule(row_count=len(states), validation_fraction=0.2)
    if not split.train_indices or not split.validation_indices:
        raise ValueError("Training requires nonempty train and validation splits.")
    identity = {
        "schema": "morpion_entity_value_training_v1",
        "config": asdict(config),
        "optimizer": "AdamW",
        "schedule": "linear_warmup_cosine",
        "loss": "MSE",
        "target": "raw_existing_state_value",
        "row_count": len(states),
        "split_policy": split.split_policy,
        "train_count": len(split.train_indices),
        "validation_count": len(split.validation_indices),
        "provenance": provenance,
    }
    # Normalize tuple/list distinctions once, including on resume.
    identity = json.loads(json.dumps(identity))
    manifest_path = output_dir / "training_config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError("Resume configuration or source provenance differs.")
    atomic_json(manifest_path, identity)
    result_path = output_dir / "result.json"
    if result_path.exists():
        return cast("dict[str, object]", json.loads(result_path.read_text()))
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = build_morpion_regressor(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    dataset = AugmentedMorpionDataset(
        states,
        seed=config.seed,
        augment=config.d4_augmentation,
        max_tokens=config.model.entity_max_tokens,
    )
    sampler = EpochRowSampler(split.train_indices, config.seed)
    loader = _loader(dataset, sampler, config)
    steps_per_epoch = math.ceil(len(split.train_indices) / config.batch_size)
    total_steps = steps_per_epoch * config.num_epochs
    step = 0
    previous_seconds = 0.0
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = cast(
            "dict[str, Any]",
            torch.load(checkpoint_path, map_location=device, weights_only=False),
        )
        if checkpoint["identity"] != identity:
            raise ValueError("Checkpoint configuration differs.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        previous_seconds = float(checkpoint["training_seconds"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([
                value.cpu() for value in checkpoint["cuda_rng"]
            ])
    started = last_log = last_save = time.monotonic()
    initial_step = step

    def save_checkpoint() -> None:
        _synchronize(device)
        _atomic_torch_save(
            checkpoint_path,
            {
                "identity": identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "training_seconds": previous_seconds + time.monotonic() - started,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
            },
        )

    progress(
        f"Training seed={config.seed} D4={config.d4_augmentation}; optimizer steps {step}/{total_steps}"
    )
    interrupted = False

    def request_interrupt(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    previous_handler = signal.signal(signal.SIGINT, request_interrupt)
    status: dict[str, object]
    try:
        model.train()
        while step < total_steps:
            sampler.epoch = step // steps_per_epoch
            sampler.start = (step % steps_per_epoch) * config.batch_size
            for batch in loader:
                learning_rate = learning_rate_at_step(config, step, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                prediction = model(*_inputs(batch, device))
                loss = torch.nn.functional.mse_loss(
                    prediction, batch.target_tensor.to(device)
                )
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
                step += 1
                now = time.monotonic()
                if now - last_log >= 15 or step % steps_per_epoch == 0:
                    _synchronize(device)
                    elapsed = time.monotonic() - started
                    rate = (step - initial_step) / max(elapsed, 1e-9)
                    epoch = (step - 1) // steps_per_epoch + 1
                    loss_value = float(loss.detach())
                    eta = (total_steps - step) / rate
                    status = {
                        "status": "training",
                        "step": step,
                        "total_steps": total_steps,
                        "epoch": epoch,
                        "loss_last_batch": loss_value,
                        "elapsed_seconds": previous_seconds + elapsed,
                        "eta_seconds": eta,
                        "optimizer_steps_per_second": rate,
                    }
                    atomic_json(output_dir / "progress.json", status)
                    progress(
                        f"epoch {epoch}/{config.num_epochs}; step {step}/{total_steps}; loss={loss_value:.4f}; elapsed={(previous_seconds + elapsed) / 60:.1f} min; ETA={eta / 60:.1f} min"
                    )
                    last_log = now
                if now - last_save >= 60 or step % steps_per_epoch == 0:
                    save_checkpoint()
                    last_save = now
                if interrupted or (
                    stop_after_steps is not None
                    and step - initial_step >= stop_after_steps
                ):
                    raise KeyboardInterrupt  # noqa: TRY301 - stop only at a complete optimizer step
    except KeyboardInterrupt:
        save_checkpoint()
        status = {
            "status": "interrupted",
            "step": step,
            "total_steps": total_steps,
            "training_seconds": previous_seconds + time.monotonic() - started,
        }
        atomic_json(output_dir / "progress.json", status)
        progress(f"Saved resumable checkpoint at step {step}/{total_steps}")
        return status
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    _synchronize(device)
    training_seconds = previous_seconds + time.monotonic() - started
    # Release training workers before starting the validation loader.
    del loader
    progress("Final canonical validation (no augmentation or symmetry averaging)")
    validation_sampler = EpochRowSampler(split.validation_indices, config.seed)
    validation_loader = _loader(dataset, validation_sampler, config)
    predictions, targets = [], []
    validation_started = last_log = time.monotonic()
    first_validation_batch_seconds = 0.0
    first_validation_batch_rows = 0
    model.eval()
    with torch.inference_mode():
        for batch in validation_loader:
            predictions.append(model(*_inputs(batch, device)).cpu())
            # Worker tensors retain shared-memory file descriptors. Keep a local
            # copy so validation releases each batch's descriptors immediately.
            targets.append(batch.target_tensor.clone())
            if len(targets) == 1:
                first_validation_batch_seconds = time.monotonic() - validation_started
                first_validation_batch_rows = len(batch.target_tensor)
            if time.monotonic() - last_log >= 15:
                progress(
                    f"Validation {sum(len(x) for x in targets)}/{len(split.validation_indices)} rows"
                )
                last_log = time.monotonic()
    validation_loop_seconds = time.monotonic() - validation_started
    del validation_loader
    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    metrics = asdict(
        regression_quality_stats(predictions=prediction_tensor, targets=target_tensor)
    )
    count = min(config.symmetry_diagnostic_rows, len(split.validation_indices))
    diagnostic_indices = tuple(
        split.validation_indices[i * len(split.validation_indices) // count]
        for i in range(count)
    )
    progress(f"D4 consistency diagnostic on {count} validation rows")
    diagnostic_started = time.monotonic()
    diagnostic = symmetry_consistency(model, dataset, diagnostic_indices, device)
    diagnostic_seconds = time.monotonic() - diagnostic_started
    validation_seconds = time.monotonic() - validation_started
    _atomic_torch_save(
        output_dir / "validation.pt",
        {
            "row_indices": torch.tensor(split.validation_indices),
            "predictions": prediction_tensor,
            "targets": target_tensor,
        },
    )
    if not (output_dir / "bundle").exists():
        bundle_dir = output_dir / "bundle.pending"
        save_morpion_model_bundle(
            model=model,
            output_dir=bundle_dir,
            model_args=config.model,
            metadata={
                "training_config": identity,
                "training_seconds": training_seconds,
                "validation_metrics": metrics,
            },
        )
        bundle_dir.replace(output_dir / "bundle")
    result: dict[str, object] = {
        "status": "complete",
        "seed": config.seed,
        "d4_augmentation": config.d4_augmentation,
        "metrics": metrics,
        "symmetry_consistency": diagnostic,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": training_seconds,
        "validation_and_diagnostics_seconds": validation_seconds,
        "validation_timing": {
            "first_batch_seconds": first_validation_batch_seconds,
            "first_batch_rows": first_validation_batch_rows,
            "following_batches_seconds": validation_loop_seconds
            - first_validation_batch_seconds,
            "following_rows": len(split.validation_indices)
            - first_validation_batch_rows,
            "diagnostic_seconds": diagnostic_seconds,
            "diagnostic_rows": count,
            "other_seconds": validation_seconds
            - validation_loop_seconds
            - diagnostic_seconds,
        },
        "optimizer_steps": step,
        "bundle": str(output_dir / "bundle"),
    }
    atomic_json(result_path, result)
    atomic_json(
        output_dir / "progress.json",
        {"status": "complete", "step": step, "total_steps": total_steps},
    )
    progress(
        f"Complete: seed={config.seed}, D4={config.d4_augmentation}, validation MSE={metrics['mse']:.6f}, training={training_seconds / 60:.1f} min"
    )
    return result
