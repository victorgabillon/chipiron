"""Direct tests for the tiny deterministic canary games added in PR1."""

from __future__ import annotations

from test_support.canary_games import (
    RoleCycleCanaryGame,
    make_solo_canary_game,
    make_three_role_canary_game,
    make_two_role_canary_game,
)


def run_to_completion(game: RoleCycleCanaryGame) -> list[str]:
    """Collect the deterministic actor sequence for a canary game."""
    actors: list[str] = []
    state = game.initial_state
    while not state.is_terminal():
        actor = game.current_actor(state)
        assert actor is not None
        actors.append(actor)
        assert game.legal_actions(state) == ("advance",)
        state = game.step(state, "advance")
    assert game.current_actor(state) is None
    assert game.legal_actions(state) == ()
    assert state.history == tuple(actors)
    return actors


def test_solo_canary_game_has_one_actor_and_a_predictable_terminal_path() -> None:
    """Keep the single-role future fixture tiny, explicit, and deterministic."""
    game = make_solo_canary_game(total_steps=3)

    assert game.roles == ("solo",)
    assert run_to_completion(game) == ["solo", "solo", "solo"]


def test_two_role_canary_game_alternates_roles_without_engine_complexity() -> None:
    """Keep the two-role canary focused on explicit alternation only."""
    game = make_two_role_canary_game(total_steps=4)

    assert game.roles == ("white", "black")
    assert run_to_completion(game) == ["white", "black", "white", "black"]


def test_three_role_canary_game_cycles_three_roles_in_fixed_order() -> None:
    """Prepare a future multi-role fixture without wiring it into production yet."""
    game = make_three_role_canary_game(total_steps=4)

    assert game.roles == ("alpha", "beta", "gamma")
    assert run_to_completion(game) == ["alpha", "beta", "gamma", "alpha"]
