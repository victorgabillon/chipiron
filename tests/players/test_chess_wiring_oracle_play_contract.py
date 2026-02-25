import ast
from pathlib import Path

SOURCE_PATH = Path("src/chipiron/players/wirings/chess_wiring.py")
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def test_build_chess_game_player_checks_oracle_play_flag() -> None:
    function_node = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_chess_game_player"
    )

    function_source = ast.get_source_segment(SOURCE, function_node) or ""
    assert "args.player_factory_args.player_args.oracle_play" in function_source
