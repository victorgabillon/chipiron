"""
Default arguments for the script.
"""

from dataclasses import dataclass
from typing import Protocol

from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.dataclass import IsDataclass


@dataclass
class DefaultScriptArgs:
    """Default arguments for the script."""

    base_script_args: BaseScriptArgs


class HasBaseScriptArgs(Protocol):
    """Mixin class that provides access to the base_script_args attribute."""

    base_script_args: BaseScriptArgs


class DataClassWithBaseScriptArgs(HasBaseScriptArgs, IsDataclass, Protocol):
    """Data class that includes base_script_args."""
