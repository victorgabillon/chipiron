"""
This module defines the IScript interface, which serves as the interface for scripts in the application.
"""

from typing import Protocol

from .default_script_args import DataClassWithBaseScriptArgs, DefaultScriptArgs
from .script import AnyScript


class IScript(Protocol):
    """
    The interface for scripts in the application.
    """

    base_script: AnyScript

    @classmethod
    def get_args_dataclass_name(cls) -> type[DataClassWithBaseScriptArgs]:
        """
        Returns the dataclass type that holds the arguments for the script.

        Returns:
            type: The dataclass type for the script's arguments.
        """
        return DefaultScriptArgs  # could be overridden by subclasses

    def __init__(self, base_script: AnyScript) -> None:
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
