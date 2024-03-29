from .factory import ScriptType, create_script
from .iscript import IScript
from .script import Script

# tkinter problems if the folowing is not commenting we should get rid of tkinter probably
# from .script_gui_custom import script_gui

__all__ = [
    "IScript",
    "ScriptType",
    "Script",
    "create_script"
]
