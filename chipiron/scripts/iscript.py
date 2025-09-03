"""
This module defines the IScript interface, which serves as the interface for scripts in the application.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.dataclass import IsDataclass

from .script import Script


@dataclass
class DefaultScriptArgs:
    """Default arguments for the script."""

    base_script_args: BaseScriptArgs


class HasBaseScriptArgs(Protocol):
    """Mixin class that provides access to the base_script_args attribute."""

    base_script_args: BaseScriptArgs


class DataClassWithBaseScriptArgs(HasBaseScriptArgs, IsDataclass, Protocol):
    """Data class that includes base_script_args."""


class IScript(Protocol):
    """
    The interface for scripts in the application.
    """

    base_script: Script[Any]

    @classmethod
    def get_args_dataclass_name(cls) -> type[DataClassWithBaseScriptArgs]:
        """
        Returns the dataclass type that holds the arguments for the script.

        Returns:
            type: The dataclass type for the script's arguments.
        """
        return DefaultScriptArgs  # could be overridden by subclasses

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
