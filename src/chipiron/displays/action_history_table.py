"""Helpers for presenting flat action history as a participant-column table."""

from __future__ import annotations

import typing


def build_action_history_table(
    *,
    action_name_history: typing.Sequence[str],
    participant_labels: typing.Sequence[str],
) -> tuple[list[str], list[list[str]]]:
    """Convert flat action history into participant columns.

    When participants are known, rows are filled chronologically left-to-right using
    the provided participant order. If no participants are known, the caller should
    fall back to a simpler display.
    """
    if not participant_labels:
        return ["Ply", "Action"], [
            [str(half_move), str(action_name)]
            for half_move, action_name in enumerate(action_name_history, start=1)
        ]

    participant_count = len(participant_labels)
    headers = list(participant_labels)
    rows: list[list[str]] = []
    for start in range(0, len(action_name_history), participant_count):
        round_actions = list(action_name_history[start : start + participant_count])
        padded_actions = round_actions + [""] * (participant_count - len(round_actions))
        rows.append(padded_actions)

    return headers, rows
