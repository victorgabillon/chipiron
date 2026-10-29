"""Plot measured activation/refresh costs without extrapolating beyond the run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Render a standalone scientific SVG from the retained read-only audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    data = json.loads(options.audit.read_text(encoding="utf-8"))
    totals = {
        item["generation"]: item["elapsed_s"]
        for item in data["measurements"]
        if item["stage"] == "activation"
    }
    points = sorted(
        (
            item["completion"]["refresh"]["requested_count"],
            totals[item["generation"]],
            item["completion"]["refresh_performance"]["elapsed_s"],
        )
        for item in data["activations"]
    )
    assert points
    nodes, total, refresh = zip(*points, strict=True)
    figure, axes = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    axes.plot(nodes, total, "o-", label="Atomic activation total", color="#194a72")
    axes.plot(
        nodes, refresh, "s-", label="Full refresh (within total)", color="#c26821"
    )
    axes.set(
        xlabel="Distinct nodes at activation",
        ylabel="Wall time (seconds)",
        title="Morpion continual bootstrap: measured activation cost",
        ylim=(0, None),
    )
    axes.grid(alpha=0.2)
    axes.legend(frameon=False)
    figure.savefig(options.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
