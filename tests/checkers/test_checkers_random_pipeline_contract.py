import ast
from pathlib import Path

SOURCE = Path("src/chipiron/players/wirings/checkers_wiring.py").read_text()
TREE = ast.parse(SOURCE)


def _find_function(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found")


def test_build_checkers_game_player_uses_pipeline_and_adapter() -> None:
    build_fn = _find_function("build_checkers_game_player")
    function_source = ast.get_source_segment(SOURCE, build_fn) or ""

    assert "create_player_with_pipeline" in function_source
    assert "CheckersAdapter" in function_source


def test_build_checkers_game_player_routes_selector_with_checkers_game_kind() -> None:
    build_fn = _find_function("build_checkers_game_player")

    found_selector_call = False
    for node in ast.walk(build_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "create_main_move_selector":
                found_selector_call = True
                for kw in node.keywords:
                    if kw.arg == "game_kind":
                        assert isinstance(kw.value, ast.Attribute)
                        assert isinstance(kw.value.value, ast.Name)
                        assert kw.value.value.id == "GameKind"
                        assert kw.value.attr == "CHECKERS"
                        break
                else:
                    raise AssertionError(
                        "create_main_move_selector call missing game_kind keyword"
                    )

    assert found_selector_call, (
        "Expected create_main_move_selector call in checkers wiring"
    )
