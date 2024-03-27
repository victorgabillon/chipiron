"""
The interface script
"""

from typing import Protocol


class IScript(Protocol):
    """ the interface for scripts"""

    def run(self) -> None:
        """ Running the script"""

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
