"""Fresh serial synthetic generations using the existing full-checkpoint driver."""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Refuse an existing destination; never operate on historical data."""
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(module: str, *args: str) -> None:
        command = [sys.executable, "-m", module, str(root), *args]
        with (root / f"step-{len(commands):03d}.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=120
            )
        commands.append({"command": command, "returncode": result.returncode})
        (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            message = f"Failed step {len(commands) - 1}; see its log"
            raise SystemExit(message)

    for generation in range(1, 65):
        run(
            "docs.research.durable_closed_snapshot.measure",
            "initial" if generation == 1 else "repeat",
            "--count",
            "4",
            "--history",
        )
        if generation in (8, 16, 32, 64):
            run("docs.research.minimal_rolling_recovery.observe")
            for mode in ("cold", "reuse", "oracle", "continuation"):
                run(
                    "docs.research.durable_closed_snapshot.measure",
                    mode,
                    "--count",
                    "4",
                    "--history",
                )
            print(json.dumps({"completed_generation": generation}), flush=True)


if __name__ == "__main__":
    main()
