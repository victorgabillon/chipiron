"""
Init file for scripts module
"""

from .iscript import IScript
from .script import Script
from .script_type import ScriptType

# tkinter problems if the following is not commenting we should get rid of tkinter probably
# from .script_gui_custom import script_gui

__all__ = [
    "IScript",
    "ScriptType",
    "Script",
]
