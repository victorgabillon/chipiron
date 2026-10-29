"""Serial recipe; fresh output required, no existing fixture is overwritten."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(root: Path) -> None:
    """Run main for the isolated synthetic study."""
    assert root.resolve().is_relative_to(
        Path(
            "/home/pompote/oldata/victor/morpion-scratch/rolling-snapshot-design-20261022"
        )
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "runs").mkdir()
    (root / "logs").mkdir()
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src:.",
        "QT_QPA_PLATFORM": "offscreen",
        "TMPDIR": "/home/pompote/oldata/victor/morpion-scratch/rs-20261022",
        "XDG_CACHE_HOME": str(root / "cache"),
        "MPLCONFIGDIR": str(root / "cache/matplotlib"),
        "XDG_CONFIG_HOME": str(root / "cache/config"),
    }

    def run(name: str, *arguments: object) -> None:
        print(name, flush=True)
        with (root / "logs" / (name + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, *map(str, arguments)],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    driver = "docs/research/rolling_snapshot/measure.py"
    for size in (1000, 5000, 10000):
        prepare = root / "runs" / f"real-{size}-prepare"
        run(f"prepare-{size}", driver, "prepare", "--root", prepare, "--nodes", size)
        run(
            f"checkpoint-{size}",
            driver,
            "checkpoint",
            "--root",
            root / "runs" / f"real-{size}-checkpoint",
            "--source",
            prepare / "checkpoint",
        )
        for mode in ("complete", "shared"):
            run(
                f"real-{size}-{mode}",
                driver,
                "scientific",
                "--root",
                root / "runs" / f"real-{size}-{mode}",
                "--source",
                prepare / "checkpoint",
                "--mode",
                mode,
            )
        for offset in (0, 20):
            for mode in ("complete", "shared"):
                run(
                    f"exact-{size}-offset{offset}-{mode}",
                    driver,
                    "scientific",
                    "--root",
                    root / "runs" / f"exact-{size}-offset{offset}-{mode}",
                    "--nodes",
                    size,
                    "--mode",
                    mode,
                    "--offset",
                    offset,
                )
        source = (
            root
            / "runs"
            / f"real-{size}-shared/shared/generation_000003/generation.json"
        )
        for label, action in (
            ("cold", "indexed"),
            ("reuse", "indexed"),
            ("oracle", "oracle"),
        ):
            run(
                f"reader-{size}-{label}",
                driver,
                action,
                "--root",
                root / "runs" / f"reader-{size}-{label}",
                "--source",
                source,
            )
    run(
        "continuation",
        "docs/research/rolling_snapshot/checkpoint_parity.py",
        root / "runs/continuation",
    )
    run(
        "reachability",
        "docs/research/rolling_snapshot/reachability.py",
        root / "runs/reachability",
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
