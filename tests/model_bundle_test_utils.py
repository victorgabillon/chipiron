"""Small local model-bundle helpers for deterministic offline tests."""

from __future__ import annotations

from pathlib import Path
import textwrap

import torch


def create_tiny_model_bundle(
    tmp_path: Path,
    *,
    bundle_name: str = "model_bundle",
    input_representation: str = "piece_difference",
    weights_file: str = "weights.pt",
) -> Path:
    """Create a tiny local model bundle under ``tmp_path`` for tests."""
    bundle_dir = tmp_path / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "architecture.yaml").write_text(
        textwrap.dedent(
            """\
            model_output_type:
              point_of_view: player_to_move
            model_type_args:
              list_of_activation_functions:
              - hyperbolic_tangent
              number_neurons_per_layer:
              - 5
              - 1
              type: multi_layer_perceptron
            """
        ),
        encoding="utf-8",
    )
    (bundle_dir / "chipiron_nn.yaml").write_text(
        textwrap.dedent(
            f"""\
            version: 1
            game_kind: chess
            input_representation: {input_representation}
            """
        ),
        encoding="utf-8",
    )

    torch.save({"dummy": True}, bundle_dir / weights_file)
    return bundle_dir
