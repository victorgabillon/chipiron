import ast
from pathlib import Path

SOURCE = Path(
    "src/chipiron/environments/checkers/players/wiring/checkers_wiring.py"
).read_text()
TREE = ast.parse(SOURCE)


def _find_function(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found")


def test_build_checkers_game_player_uses_pipeline_and_adapter() -> None:
    build_fn = _find_function("build_checkers_game_player")
    function_source = ast.get_source_segment(SOURCE, build_fn) or ""

    assert "create_player_with_standard_adapter_pipeline" in function_source
    assert "CheckersAdapter" in function_source


def test_build_checkers_game_player_routes_pipeline_with_checkers_game_kind() -> None:
    build_fn = _find_function("build_checkers_game_player")

    found_pipeline_call = False
    for node in ast.walk(build_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "create_player_with_standard_adapter_pipeline":
                found_pipeline_call = True
                for kw in node.keywords:
                    if kw.arg == "game_kind":
                        assert isinstance(kw.value, ast.Attribute)
                        assert isinstance(kw.value.value, ast.Name)
                        assert kw.value.value.id == "GameKind"
                        assert kw.value.attr == "CHECKERS"
                        break
                else:
                    raise AssertionError(
                        "create_player_with_standard_adapter_pipeline call missing game_kind keyword"
                    )

    assert found_pipeline_call, (
        "Expected create_player_with_standard_adapter_pipeline call in checkers wiring"
    )
