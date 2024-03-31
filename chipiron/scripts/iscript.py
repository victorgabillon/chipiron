"""
The interface script
"""

from typing import Protocol, Any

from .script import Script


class IScript(Protocol):
    """ the interface for scripts"""
    args_dataclass_name: Any

    def __init__(
            self,
            base_script: Script
    ) -> None:
        ...

    def run(self) -> None:
        """ Running the script"""
        ...

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
        ...
