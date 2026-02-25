import ast
from pathlib import Path


SOURCE = Path("src/chipiron/players/factory_higher_level.py").read_text()
TREE = ast.parse(SOURCE)


def _find_function(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found")


def test_public_helper_delegates_to_select_wiring() -> None:
    helper = _find_function("get_observer_wiring_for_game_kind")
    returns = [node for node in ast.walk(helper) if isinstance(node, ast.Return)]
    assert returns, "helper should return selected wiring"
    return_expr = returns[0].value
    assert isinstance(return_expr, ast.Call)
    assert isinstance(return_expr.func, ast.Name)
    assert return_expr.func.id == "_select_wiring"


def test_checkers_wiring_is_registered_in_select_wiring() -> None:
    selector = _find_function("_select_wiring")

    has_checkers_case = False
    for node in ast.walk(selector):
        if isinstance(node, ast.match_case):
            pattern = node.pattern
            if (
                isinstance(pattern, ast.MatchValue)
                and isinstance(pattern.value, ast.Attribute)
                and isinstance(pattern.value.value, ast.Name)
                and pattern.value.value.id == "GameKind"
                and pattern.value.attr == "CHECKERS"
            ):
                has_checkers_case = True
                returns = [stmt for stmt in node.body if isinstance(stmt, ast.Return)]
                assert returns, "CHECKERS case should return a wiring"
                return_value = returns[0].value
                assert isinstance(return_value, ast.Call)
                assert any(
                    isinstance(arg, ast.Name) and arg.id == "CHECKERS_WIRING"
                    for arg in return_value.args
                )

    assert has_checkers_case, "_select_wiring should include GameKind.CHECKERS"
