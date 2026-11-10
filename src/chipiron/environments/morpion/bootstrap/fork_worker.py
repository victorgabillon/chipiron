"""One owned no-descendant genesis helper, killed if its coordinator disappears."""

import ctypes
import fcntl
import json
import os
import signal
import sys
from pathlib import Path


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def main() -> None:
    """Bind death/identity before imports or writes, then use normal fresh genesis."""
    stage, checkpoint, model, name, parent, ticks, inherited_owner = sys.argv[1:]
    parent_pid = int(parent)
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot bind fork helper parent death")
    fields = (
        Path(f"/proc/{parent_pid}/stat")
        .read_text(encoding="utf-8")
        .rsplit(") ", 1)[1]
        .split()
    )
    _require(
        os.getppid() == parent_pid and fields[19] == ticks,
        "Fork coordinator identity changed before helper initialization.",
    )
    root = Path(stage)
    inherited = os.fstat(int(inherited_owner))
    expected = (root / "resource_monitor/owner.lock").stat()
    _require(
        (inherited.st_dev, inherited.st_ino) == (expected.st_dev, expected.st_ino),
        "Wrong inherited fork preparation lease.",
    )
    marker = json.loads((root / "fork-preparation.json").read_text(encoding="utf-8"))
    _require(
        (marker["owner"]["pid"], marker["owner"]["start_ticks"]) == (parent_pid, ticks),
        "Fork helper does not belong to this preparation.",
    )
    with (root / "resource_monitor/owner.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            _require(False, "Fork coordinator no longer owns its preparation.")
    from .bootstrap_args import MorpionBootstrapArgs
    from .config import load_bootstrap_config
    from .derived.provenance import inside
    from .execution_profile import bootstrap_execution
    from .fork_genesis import initialize_fork_genesis
    from .launcher import _scientific_args

    config = load_bootstrap_config(root / "bootstrap_config.json")
    args = _scientific_args(MorpionBootstrapArgs(root), config)
    with bootstrap_execution(args, process_start=True):
        initialize_fork_genesis(
            args,
            config,
            checkpoint=inside(root, checkpoint),
            incumbent=inside(root, model),
            incumbent_name=name,
        )


if __name__ == "__main__":
    main()
