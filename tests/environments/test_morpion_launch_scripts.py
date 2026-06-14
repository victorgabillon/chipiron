"""Static checks for Morpion bootstrap launch scripts."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cluster_scripts_expose_training_export_mode_env() -> None:
    """Cluster launch scripts should surface the training export mode knob."""
    for script_name in (
        "launch_morpion_gnome_cluster.sh",
        "launch_morpion_tmux_cluster.sh",
    ):
        script_text = (_REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")

        assert 'TRAINING_EXPORT_MODE="${TRAINING_EXPORT_MODE:-sharded}"' in script_text
        assert "--training-export-mode" in script_text
        assert "$TRAINING_EXPORT_MODE" in script_text


def test_cluster_scripts_enable_rollout_defaults() -> None:
    """Cluster launch scripts should default to rollout-enabled Morpion growth."""
    for script_name in (
        "launch_morpion_gnome_cluster.sh",
        "launch_morpion_tmux_cluster.sh",
    ):
        script_text = (_REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")

        assert (
            'MORPION_ROLLOUT_AFTER_OPENING="${MORPION_ROLLOUT_AFTER_OPENING:-1}"'
            in script_text
        )
        assert "--rollout-after-opening" in script_text
        assert "--rollout-max-extra-steps" in script_text
        assert "--rollout-action-selector-kind" in script_text
