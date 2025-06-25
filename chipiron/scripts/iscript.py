"""
This module defines the IScript interface, which serves as the interface for scripts in the application.
"""

from typing import Any, Protocol

from .script import Script


class IScript(Protocol):
    """
    The interface for scripts in the application.
    """

    def __init__(self, base_script: Script[Any]) -> None:
        """
        Initializes the IScript object.

        Args:
            base_script (Script): The base script object.

        Returns:
            None
        """

    def run(self) -> None:
        """
        Runs the script.

        Returns:
            None
        """

    def terminate(self) -> None:
        """
        Finishes the script. Performs profiling or timing.

        Returns:
            None
        """
