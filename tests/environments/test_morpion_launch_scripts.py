"""Static checks for Morpion bootstrap launch scripts."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _script_text(script_name: str) -> str:
    """Read one cluster launch script for static assertions."""
    return (_REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")


def _line_starting(script_text: str, prefix: str) -> str:
    """Return the first script line with the requested assignment prefix."""
    return next(line for line in script_text.splitlines() if line.startswith(prefix))


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
        assert (
            'MORPION_ROLLOUT_ACTION_SELECTOR_KIND="${MORPION_ROLLOUT_ACTION_SELECTOR_KIND:-random_legal_prefer_openable}"'
            in script_text
        )
        assert (
            "MORPION_ROLLOUT_ACTION_SELECTOR_KIND=random_legal_prefer_openable"
            in script_text
        )
        assert "--rollout-after-opening" in script_text
        assert "--rollout-max-extra-steps" in script_text
        assert "--rollout-action-selector-kind" in script_text


def test_growth_cluster_supervisors_stop_on_exhausted_budget() -> None:
    """Growth supervisors should not restart after the branch budget is exhausted."""
    for script_name in (
        "launch_morpion_gnome_cluster.sh",
        "launch_morpion_tmux_cluster.sh",
    ):
        script_text = (_REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")

        assert "growth_budget_already_exhausted" in script_text
        assert "not restarting" in script_text
        assert "break" in script_text


def test_cluster_growth_worker_max_cycles_is_growth_only() -> None:
    """Growth workers should amortize startup without changing other workers."""
    expected_default = (
        'MORPION_GROWTH_WORKER_MAX_CYCLES="${MORPION_GROWTH_WORKER_MAX_CYCLES:-20}"'
    )
    for script_name in (
        "launch_morpion_gnome_cluster.sh",
        "launch_morpion_tmux_cluster.sh",
    ):
        script_text = _script_text(script_name)

        assert expected_default in script_text
        assert "worker_max_cycles=$MORPION_GROWTH_WORKER_MAX_CYCLES" in script_text
        assert "--max-cycles $MORPION_GROWTH_WORKER_MAX_CYCLES" in script_text

        if script_name == "launch_morpion_gnome_cluster.sh":
            growth_command = _line_starting(script_text, "GROWTH_ARGS=")
            other_worker_commands = (
                _line_starting(script_text, "DATASET_ARGS="),
                _line_starting(script_text, "TRAINING_ARGS="),
                _line_starting(script_text, "REEVALUATION_ARGS="),
            )
        else:
            growth_command = _line_starting(script_text, "growth_loop=")
            other_worker_commands = (
                _line_starting(script_text, "dataset_loop="),
                _line_starting(script_text, "training_loop="),
                _line_starting(script_text, "reevaluation_loop="),
            )

        assert "--max-cycles $MORPION_GROWTH_WORKER_MAX_CYCLES" in growth_command
        for command in other_worker_commands:
            assert "--max-cycles" not in command
            assert "MORPION_GROWTH_WORKER_MAX_CYCLES" not in command


def test_cluster_scripts_expose_operator_output_controls() -> None:
    """Cluster launchers should expose human-output knobs without coloring logs."""
    for script_name in (
        "launch_morpion_gnome_cluster.sh",
        "launch_morpion_tmux_cluster.sh",
    ):
        script_text = _script_text(script_name)

        assert 'MORPION_OPERATOR_RICH="${MORPION_OPERATOR_RICH:-auto}"' in script_text
        assert 'MORPION_COLOR_RAW_LOGS="${MORPION_COLOR_RAW_LOGS:-0}"' in script_text
        assert 'MORPION_GROWTH_SHOW_RECAP="${MORPION_GROWTH_SHOW_RECAP:-1}"' in (
            script_text
        )
        assert "chipiron.environments.morpion.bootstrap.growth_recap" in script_text
        assert "--worker-max-cycles" in script_text
        assert "tee -a" in script_text


def test_cluster_scripts_expose_training_recap_controls() -> None:
    """Cluster launch scripts should expose compact training recap controls."""
    for script_name in (
        "launch_morpion_gnome_cluster.sh",
        "launch_morpion_tmux_cluster.sh",
    ):
        script_text = (_REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")

        assert "training_recap" in script_text
        assert "MORPION_CLUSTER_SHOW_RECAP" in script_text
        assert "MORPION_CLUSTER_IDLE_LOG_EVERY" in script_text
        assert "--all-evaluators" in script_text


def test_cluster_python_diagnostics_use_valid_quotes() -> None:
    """Cluster launcher Python diagnostics should not escape heredoc quotes."""
    for script_name in (
        "launch_morpion_gnome_cluster.sh",
        "launch_morpion_tmux_cluster.sh",
    ):
        script_text = (_REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")

        assert r"print(\"" not in script_text
        assert 'print("anemone:", anemone.__file__)' in script_text
        assert 'print("atomheart:", atomheart.__file__)' in script_text
        assert 'print("chipiron:", chipiron.__file__)' in script_text
        assert 'print("coral:", coral.__file__)' in script_text


def test_gnome_cluster_exposes_streaming_training_chunk_size_env() -> None:
    """The GNOME cluster launcher should expose the JSONL training chunk knob."""
    script_text = (
        _REPO_ROOT / "scripts" / "launch_morpion_gnome_cluster.sh"
    ).read_text(encoding="utf-8")

    assert (
        'MORPION_TRAINING_ROW_CHUNK_SIZE="${MORPION_TRAINING_ROW_CHUNK_SIZE:-8192}"'
        in script_text
    )
    assert "--training-row-chunk-size $MORPION_TRAINING_ROW_CHUNK_SIZE" in script_text
    assert "row_chunk_size=$MORPION_TRAINING_ROW_CHUNK_SIZE" in script_text


def test_gnome_cluster_exposes_evaluator_diagnostics_max_rows_env() -> None:
    """The GNOME cluster launcher should bound evaluator diagnostics by default."""
    script_text = (
        _REPO_ROOT / "scripts" / "launch_morpion_gnome_cluster.sh"
    ).read_text(encoding="utf-8")

    assert (
        'MORPION_EVALUATOR_DIAGNOSTICS_MAX_ROWS="${MORPION_EVALUATOR_DIAGNOSTICS_MAX_ROWS:-60}"'
        in script_text
    )
    assert (
        "--evaluator-diagnostics-max-rows $MORPION_EVALUATOR_DIAGNOSTICS_MAX_ROWS"
    ) in script_text
    assert (
        "evaluator_diagnostics_max_rows=$MORPION_EVALUATOR_DIAGNOSTICS_MAX_ROWS"
        in script_text
    )


def test_gnome_cluster_exposes_growth_memory_profile_env() -> None:
    """The GNOME cluster launcher should expose opt-in growth profiling knobs."""
    script_text = (
        _REPO_ROOT / "scripts" / "launch_morpion_gnome_cluster.sh"
    ).read_text(encoding="utf-8")

    assert 'MORPION_GROWTH_MEMORY_PROFILE="${MORPION_GROWTH_MEMORY_PROFILE:-0}"' in (
        script_text
    )
    assert (
        'MORPION_GROWTH_MEMORY_PROFILE_TOP_N="${MORPION_GROWTH_MEMORY_PROFILE_TOP_N:-20}"'
        in script_text
    )
    assert (
        'MORPION_GROWTH_MEMORY_PROFILE_SAMPLE_NODES="${MORPION_GROWTH_MEMORY_PROFILE_SAMPLE_NODES:-2000}"'
        in script_text
    )
    assert "--growth-memory-profile --growth-memory-profile-top-n" in script_text
    assert "memory_profile=$MORPION_GROWTH_MEMORY_PROFILE" in script_text
    assert "memory_profile_top_n=$MORPION_GROWTH_MEMORY_PROFILE_TOP_N" in script_text
    assert (
        "memory_profile_sample_nodes=$MORPION_GROWTH_MEMORY_PROFILE_SAMPLE_NODES"
        in script_text
    )


def test_gnome_cluster_exposes_candidate_checkpoint_load_headroom_env() -> None:
    """The GNOME cluster launcher should expose checkpoint load forecast knobs."""
    script_text = (
        _REPO_ROOT / "scripts" / "launch_morpion_gnome_cluster.sh"
    ).read_text(encoding="utf-8")

    assert (
        'MORPION_CANDIDATE_CHECKPOINT_LOAD_HEADROOM_FACTOR="${MORPION_CANDIDATE_CHECKPOINT_LOAD_HEADROOM_FACTOR:-60}"'
        in script_text
    )
    assert (
        'MORPION_CANDIDATE_CHECKPOINT_LOAD_MIN_HEADROOM_MB="${MORPION_CANDIDATE_CHECKPOINT_LOAD_MIN_HEADROOM_MB:-512}"'
        in script_text
    )
    assert (
        "--candidate-checkpoint-load-headroom-factor "
        "$MORPION_CANDIDATE_CHECKPOINT_LOAD_HEADROOM_FACTOR"
    ) in script_text
    assert (
        "--candidate-checkpoint-load-min-headroom-mb "
        "$MORPION_CANDIDATE_CHECKPOINT_LOAD_MIN_HEADROOM_MB"
    ) in script_text
    assert (
        "candidate_checkpoint_load_headroom_factor="
        "$MORPION_CANDIDATE_CHECKPOINT_LOAD_HEADROOM_FACTOR"
    ) in script_text
    assert (
        "candidate_checkpoint_load_min_headroom_mb="
        "$MORPION_CANDIDATE_CHECKPOINT_LOAD_MIN_HEADROOM_MB"
    ) in script_text
