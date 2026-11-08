"""Pure entry selection; domain adapters retain all ownership and persistence."""

from typing import Literal

type RunMode = Literal["auto", "resume", "fresh"]
type RunAction = Literal["create", "resume", "no-work"]


def select_action(
    mode: RunMode, *, empty: bool, committed: bool, terminal: bool = False
) -> RunAction:
    """Never reinterpret incomplete existing state as a fresh run."""
    if mode not in {"auto", "resume", "fresh"}:
        message = "Unsupported lifecycle mode."
        raise ValueError(message)
    message = "Unsupported lifecycle state."
    if empty:
        if mode == "resume":
            message = "Resume requires an existing committed workspace."
        else:
            return "create"
    elif mode == "fresh":
        message = (
            "Fresh requires an absent or empty destination; preserve existing work."
        )
    elif not committed:
        message = (
            "Partial, foreign or unsupported workspace; explicit recovery is required."
        )
    else:
        return "no-work" if terminal else "resume"
    raise ValueError(message)
