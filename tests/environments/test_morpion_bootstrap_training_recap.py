"""Tests for compact Morpion training recaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap import training_recap

if TYPE_CHECKING:
    import pytest


def test_collect_latest_training_recap_empty_work_dir_returns_none(
    tmp_path: Path,
) -> None:
    """Empty work dirs should report no available training recap."""
    assert training_recap.collect_latest_training_recap(tmp_path) is None
    assert (
        training_recap.render_training_recap(None)
        == "[TRAINING-RECAP] no training recap available"
    )


def test_collect_latest_training_recap_reads_latest_generation_status(
    tmp_path: Path,
) -> None:
    """The collector should choose the highest readable training generation."""
    _write_training_status(
        tmp_path,
        generation=2,
        evaluator_name="linear_5",
        validation_loss=9.0,
    )
    _write_training_status(
        tmp_path,
        generation=7,
        evaluator_name="mlp_41",
        validation_loss=4.5,
    )

    recap = training_recap.collect_latest_training_recap(tmp_path)

    assert recap is not None
    assert recap.generation == 7
    assert recap.evaluator_name == "mlp_41"
    assert recap.validation_loss == 4.5
    assert recap.model_bundle_path == Path("models/generation_000007/mlp_41")


def test_collect_latest_training_recap_extracts_nested_metrics_and_flags(
    tmp_path: Path,
) -> None:
    """The collector should enrich status recaps from model manifests."""
    _write_training_status(
        tmp_path,
        generation=3,
        evaluator_name="entity_token_transformer_small",
        train_loss=21.91,
        validation_loss=22.77,
    )
    model_dir = (
        tmp_path / "models" / "generation_000003" / "entity_token_transformer_small"
    )
    model_dir.mkdir(parents=True)
    (model_dir / "morpion_manifest.json").write_text(
        json.dumps({
            "metadata": {
                "cached_global_shuffle": "true",
                "graph_token_cache": {"used": True},
                "graph_token_cache_used": "true",
                "regression_quality": {
                    "validation": {
                        "r2_vs_mean_baseline": 0.5123,
                        "pearson_correlation": 0.7319,
                    }
                },
            }
        }),
        encoding="utf-8",
    )
    (model_dir / "morpion_regressor_args.json").write_text(
        json.dumps({"graph_output_tanh": False}),
        encoding="utf-8",
    )

    recap = training_recap.collect_latest_training_recap(tmp_path)

    assert recap is not None
    assert recap.train_loss == 21.91
    assert recap.validation_loss == 22.77
    assert recap.validation_quality_r2_vs_mean_baseline == 0.5123
    assert recap.validation_quality_pearson_correlation == 0.7319
    assert recap.graph_output_tanh is False
    assert recap.graph_token_cache_used is True
    assert recap.cached_global_shuffle is True


def test_collect_latest_training_recap_skips_corrupt_json(
    tmp_path: Path,
) -> None:
    """Partial or corrupt status files should be skipped without crashing."""
    corrupt_dir = tmp_path / "pipeline" / "generation_000009"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "training_status.json").write_text("{", encoding="utf-8")
    _write_training_status(
        tmp_path,
        generation=4,
        evaluator_name="linear_10",
        validation_loss=1.25,
    )

    recap = training_recap.collect_latest_training_recap(tmp_path)

    assert recap is not None
    assert recap.generation == 4
    assert recap.evaluator_name == "linear_10"


def test_render_training_recap_outputs_compact_text(tmp_path: Path) -> None:
    """Rendered recaps should fit in a small terminal-friendly block."""
    _write_training_status(
        tmp_path,
        generation=5,
        evaluator_name="mlp_20",
        train_loss=2.123456,
        validation_loss=3.987654,
    )
    recap = training_recap.collect_latest_training_recap(tmp_path)

    rendered = training_recap.render_training_recap(recap)

    assert "[TRAINING-RECAP] generation=5 evaluator=mlp_20 status=done" in rendered
    assert "train_loss=2.123" in rendered
    assert "validation_loss=3.988" in rendered
    assert "model=models/generation_000005/mlp_20" in rendered


def test_collect_latest_training_recap_table_reads_multiple_evaluators(
    tmp_path: Path,
) -> None:
    """The table collector should include every evaluator result sorted by loss."""
    _write_training_status_with_results(
        tmp_path,
        generation=37,
        selected_evaluator_name="entity_token_transformer_small",
        evaluator_results={
            "mlp_41": {"train_loss": 30.0, "validation_loss": 31.0},
            "entity_token_transformer_small": {
                "train_loss": 22.05,
                "validation_loss": 22.5,
            },
            "linear_41": {"train_loss": 48.0, "validation_loss": 49.0},
        },
    )

    table = training_recap.collect_latest_training_recap_table(tmp_path)

    assert table is not None
    assert table.generation == 37
    assert table.selected_evaluator_name == "entity_token_transformer_small"
    assert [row.evaluator_name for row in table.rows] == [
        "entity_token_transformer_small",
        "mlp_41",
        "linear_41",
    ]


def test_render_training_recap_table_includes_all_evaluators(
    tmp_path: Path,
) -> None:
    """The table renderer should show all evaluator losses and mark selected."""
    _write_training_status_with_results(
        tmp_path,
        generation=37,
        selected_evaluator_name="entity_token_transformer_small",
        evaluator_results={
            "mlp_41": {"train_loss": 30.0, "validation_loss": 31.0},
            "entity_token_transformer_small": {
                "train_loss": 22.05,
                "validation_loss": 22.5,
            },
            "linear_41": {"train_loss": 48.0, "validation_loss": 49.0},
        },
    )

    table = training_recap.collect_latest_training_recap_table(tmp_path)
    rendered = training_recap.render_training_recap_table(table)

    assert "generation=37" in rendered
    assert "* entity_token_transformer_small" in rendered
    assert "mlp_41" in rendered
    assert "linear_41" in rendered
    assert rendered.index("entity_token_transformer_small") < rendered.index("mlp_41")


def test_render_training_recap_table_operator_plain_fallback(
    tmp_path: Path,
) -> None:
    """Operator rendering should keep a readable non-Rich fallback."""
    _write_training_status_with_results(
        tmp_path,
        generation=37,
        selected_evaluator_name="entity_token_transformer_small",
        evaluator_results={
            "entity_token_transformer_small": {
                "train_loss": 22.05,
                "validation_loss": 22.5,
            },
        },
    )

    table = training_recap.collect_latest_training_recap_table(tmp_path)
    rendered = training_recap.render_training_recap_table_operator(
        table,
        force_plain=True,
    )

    assert "[Morpion Training Recap]" in rendered
    assert "selected entity_token_transformer_small" in rendered
    assert "* entity_token_transformer_small" in rendered


def test_training_recap_warns_about_stale_training_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Status recap should show stale training before an empty table."""
    _write_training_status_with_results(
        tmp_path,
        generation=37,
        selected_evaluator_name="mlp_37",
        evaluator_results={"mlp_37": {"train_loss": 2.0, "validation_loss": 3.0}},
    )
    generation_dir = tmp_path / "pipeline" / "generation_000038"
    generation_dir.mkdir(parents=True, exist_ok=True)
    (generation_dir / "manifest.json").write_text(
        json.dumps({
            "created_at_utc": "2026-07-08T13:00:00Z",
            "dataset_status": "done",
            "generation": 38,
            "metadata": {},
            "model_bundle_paths": {},
            "rows_path": "rows/generation_000038.jsonl",
            "runtime_checkpoint_path": None,
            "selected_evaluator_name": None,
            "training_status": "training",
            "tree_snapshot_path": None,
        })
        + "\n",
        encoding="utf-8",
    )
    (generation_dir / "training_status.json").write_text(
        json.dumps({
            "evaluator_results": {},
            "generation": 38,
            "metadata": {},
            "selected_evaluator_name": None,
            "selection_policy": None,
            "status": "training",
            "updated_at_utc": "2026-07-08T13:00:00Z",
        })
        + "\n",
        encoding="utf-8",
    )
    (generation_dir / "training_claim.json").write_text(
        json.dumps({
            "claim_id": "claim-38",
            "claimed_at_utc": "2020-01-01T13:00:00Z",
            "expires_at_utc": "2020-01-01T14:00:00Z",
            "generation": 38,
            "metadata": {},
            "owner": None,
            "stage": "training",
        })
        + "\n",
        encoding="utf-8",
    )

    exit_code = training_recap.main(["--work-dir", str(tmp_path), "--no-rich"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "TRAINING BLOCKED / STALE" in output
    assert "generation=38" in output
    assert "evaluator_results=0" in output
    assert "Last completed training" in output
    assert "mlp_37" in output


def test_collect_latest_training_recap_table_uses_final_loss_fallback(
    tmp_path: Path,
) -> None:
    """Table rows should use final_loss when validation_loss is missing."""
    _write_training_status_with_results(
        tmp_path,
        generation=8,
        selected_evaluator_name="linear_10",
        evaluator_results={
            "linear_10": {"train_loss": 5.0, "final_loss": 6.75},
        },
    )

    table = training_recap.collect_latest_training_recap_table(tmp_path)

    assert table is not None
    assert table.rows[0].validation_loss == 6.75


def test_training_recap_json_cli_emits_valid_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default CLI JSON mode should emit table JSON data."""
    _write_training_status(
        tmp_path,
        generation=6,
        evaluator_name="linear_20",
        validation_loss=6.25,
    )

    exit_code = training_recap.main(["--work-dir", str(tmp_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["generation"] == 6
    assert payload["selected_evaluator_name"] == "linear_20"
    assert payload["rows"][0]["evaluator_name"] == "linear_20"
    assert payload["rows"][0]["validation_loss"] == 6.25


def test_training_recap_single_json_cli_preserves_old_shape(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The --single JSON mode should preserve the old single-recap shape."""
    _write_training_status(
        tmp_path,
        generation=9,
        evaluator_name="mlp_20",
        validation_loss=2.5,
    )

    exit_code = training_recap.main(["--work-dir", str(tmp_path), "--single", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["generation"] == 9
    assert payload["evaluator_name"] == "mlp_20"
    assert payload["validation_loss"] == 2.5


def test_find_key_finds_nested_values() -> None:
    """The recursive key helper should find nested JSON-like values."""
    payload = {"outer": [{"inner": {"train_loss": 3.5}}]}

    assert training_recap.find_key(payload, "train_loss") == 3.5
    assert training_recap.find_key(payload, "missing") is None


def _write_training_status(
    work_dir: Path,
    *,
    generation: int,
    evaluator_name: str,
    train_loss: float | None = None,
    validation_loss: float,
) -> None:
    status_dir = work_dir / "pipeline" / f"generation_{generation:06d}"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluator_results": {
            evaluator_name: {
                "final_loss": validation_loss,
                "model_bundle_path": (
                    f"models/generation_{generation:06d}/{evaluator_name}"
                ),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        },
        "generation": generation,
        "selected_evaluator_name": evaluator_name,
        "status": "done",
        "updated_at_utc": "2026-07-08T10:00:00Z",
    }
    (status_dir / "training_status.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_training_status_with_results(
    work_dir: Path,
    *,
    generation: int,
    selected_evaluator_name: str,
    evaluator_results: dict[str, dict[str, float]],
) -> None:
    status_dir = work_dir / "pipeline" / f"generation_{generation:06d}"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluator_results": {
            evaluator_name: {
                "final_loss": result.get("final_loss", result.get("validation_loss")),
                "model_bundle_path": (
                    f"models/generation_{generation:06d}/{evaluator_name}"
                ),
                **result,
            }
            for evaluator_name, result in evaluator_results.items()
        },
        "generation": generation,
        "selected_evaluator_name": selected_evaluator_name,
        "status": "done",
        "updated_at_utc": "2026-07-08T10:00:00Z",
    }
    (status_dir / "training_status.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
