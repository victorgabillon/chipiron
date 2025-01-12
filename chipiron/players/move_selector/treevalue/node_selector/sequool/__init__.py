"""
This module provides functionality for creating and managing Sequool objects.

A Sequool object represents a sequence of operations that can be applied to a dataset.

Example usage:
    from .factory import create_sequool, SequoolArgs

    # Create a Sequool object
    sequool = create_sequool()

    # Define the arguments for the Sequool object
    sequool_args = SequoolArgs()

"""

from .factory import SequoolArgs, create_sequool

__all__ = ["create_sequool", "SequoolArgs"]
