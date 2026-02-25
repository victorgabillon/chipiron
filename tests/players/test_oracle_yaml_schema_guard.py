from pathlib import Path


PLAYER_CONFIG_ROOT = Path("src/chipiron/data/players/player_config")


def test_no_legacy_syzygy_keys_in_player_yaml_configs() -> None:
    yaml_files = sorted(PLAYER_CONFIG_ROOT.rglob("*.yaml"))
    assert yaml_files, "Expected player YAML configs to exist"

    for yaml_file in yaml_files:
        text = yaml_file.read_text(encoding="utf-8")
        assert "syzygy_play" not in text, f"Legacy key found in {yaml_file}"
        assert "syzygy_evaluation" not in text, f"Legacy key found in {yaml_file}"
