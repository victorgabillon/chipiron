"""Explicit process ownership of bootstrap Torch policy; never inferred from weights."""

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from chipiron.learning.torch_runtime import (
    configure_torch_process_threads,
    resolve_torch_device,
    torch_execution_profile,
    torch_execution_profile_info,
    torch_execution_profile_metadata,
    validate_torch_execution_profile,
)

from .execution_policy import (
    KEY,
    policy_from_metadata,
    policy_metadata,
    validate_supported,
)
from .generation_publication_io import require

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .config import MorpionBootstrapConfig
    from .launcher import MorpionBootstrapLauncherArgs

LOGGER = logging.getLogger(__name__)


def args_with_policy(
    args: MorpionBootstrapArgs, config: MorpionBootstrapConfig
) -> MorpionBootstrapArgs:
    """Resolve explicit persisted configuration; a CLI override cannot change it."""
    saved = policy_from_metadata(config.metadata)
    requested = policy_metadata(args).get(KEY)
    if args.execution_profile is not None:
        require(requested == saved, "Execution profile changed on resume.")
    if saved is None:
        return args
    from .learning_policy import policy_from_metadata as learning_policy_from_metadata

    return replace(
        args,
        execution_profile="strict_math",
        torch_intra_op_threads=1,
        torch_inter_op_threads=1,
        training_device=config.training_device,
        continual_learning=learning_policy_from_metadata(config.metadata),
    )


def require_workspace_profile(
    paths: MorpionBootstrapPaths, *, require_device: bool = False
) -> None:
    """Manual model-consuming entries must be inside a matching lifetime owner."""
    from .config import load_bootstrap_config
    from .run_state import load_bootstrap_run_state

    if not paths.bootstrap_config_path.is_file():
        return
    config = load_bootstrap_config(paths.bootstrap_config_path)
    saved = policy_from_metadata(config.metadata)
    state = (
        load_bootstrap_run_state(paths.run_state_path)
        if paths.run_state_path.is_file()
        else None
    )
    if state is not None:
        require(
            policy_from_metadata(state.metadata) == saved,
            "Execution policy differs from genesis; legacy adoption is forbidden.",
        )
    if saved is not None:
        require(
            torch_execution_profile_info().effective == "strict_math",
            "Enter bootstrap_execution before using this strict workspace.",
        )
        require(
            torch.get_num_threads() == torch.get_num_interop_threads() == 1,
            "Torch process threads differ from the declared 1/1 recipe.",
        )
        if state is not None:
            validate_torch_execution_profile(state.metadata)
        if require_device:
            resolve_torch_device(config.training_device)


def require_args_profile(args: MorpionBootstrapArgs, *, stage: str) -> None:
    """Validate public/manual stage calls before claims, writes or model operations."""
    from .bootstrap_paths import MorpionBootstrapPaths
    from .config import load_bootstrap_config

    validate_supported(args)
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    requested = policy_metadata(args).get(KEY)
    if paths.bootstrap_config_path.is_file():
        require(
            policy_from_metadata(
                load_bootstrap_config(paths.bootstrap_config_path).metadata
            )
            == requested,
            "Explicit caller execution policy differs from persisted configuration.",
        )
    if requested is not None:
        require(
            torch_execution_profile_info().effective == "strict_math",
            "Enter bootstrap_execution before model work; scoped training alone is insufficient.",
        )
        require(
            torch.get_num_threads() == torch.get_num_interop_threads() == 1,
            "Torch process threads differ from the declared 1/1 recipe.",
        )
        if stage in {
            "training",
            "training_worker",
            "prepare-continuous",
            "initialize",
            "run-continuous",
        }:
            resolve_torch_device(args.training_device)
        if stage not in {"process", "prepare-continuous", "initialize"}:
            require(
                paths.run_state_path.is_file(),
                "Strict stages require an explicitly initialized scientific workspace.",
            )
        LOGGER.info(
            "[torch-runtime] %s",
            json.dumps(
                {
                    "pid": os.getpid(),
                    "stage": stage,
                    "training_device": args.training_device,
                    **torch_execution_profile_metadata(),
                },
                sort_keys=True,
            ),
        )
    require_workspace_profile(paths)


@contextmanager
def bootstrap_execution(
    args: MorpionBootstrapArgs, *, process_start: bool = False
) -> Iterator[None]:
    """Own all model lifetimes; only a fresh process entry may configure threads.

    Inter-op configuration is process lifetime state and is not restored. The
    shared execution-profile helper alone owns/restores deterministic/MHA/SDPA
    flags. Manual callers configure threads before work, then keep this scope
    around initialization, stage runners, activation and their model consumers.
    """
    validate_supported(args)
    if not policy_metadata(args):
        yield
        return
    if process_start:
        configure_torch_process_threads(intra_op=1, inter_op=1)
    require(
        torch.get_num_threads() == torch.get_num_interop_threads() == 1,
        "Configure declared threads at process start before bootstrap_execution.",
    )
    with torch_execution_profile("strict_math"):
        require_args_profile(args, stage="process")
        yield


@contextmanager
def launcher_execution(
    launcher: MorpionBootstrapLauncherArgs, *, process_start: bool = False
) -> Iterator[MorpionBootstrapLauncherArgs]:
    """Resolve configuration and own the entire public launcher invocation."""
    from .bootstrap_paths import MorpionBootstrapPaths
    from .config import load_bootstrap_config

    args = launcher.bootstrap_args
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    config_path = (
        launcher.scientific_config
        if launcher.scientific_operation == "initialize"
        else paths.bootstrap_config_path
    )
    if config_path is not None and config_path.is_file():
        args = args_with_policy(args, load_bootstrap_config(config_path))
    require(
        not policy_metadata(args) or launcher.scientific,
        "Explicit execution profiles require the scientific launcher.",
    )
    launcher = replace(launcher, bootstrap_args=args)
    # Status is metadata-only and must work without GPU visibility or flag changes.
    if launcher.scientific_operation == "inspect":
        yield launcher
        return
    with bootstrap_execution(args, process_start=process_start):
        stage = launcher.scientific_operation or launcher.pipeline_stage
        require_args_profile(args, stage=stage)
        yield launcher
