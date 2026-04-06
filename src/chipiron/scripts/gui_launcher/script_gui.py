"""Entry point for customtkinter script GUI."""

from typing import Any

import customtkinter as _customtkinter  # type: ignore[reportMissingImports]

from chipiron import scripts
from chipiron.utils.dataclass import IsDataclass

from .builders import generate_inputs
from .models import ArgsChosenByUser
from .ui_ctk import build_script_gui

ctk: Any = _customtkinter


def script_gui() -> tuple[scripts.ScriptType, IsDataclass | None, str]:
    """Create and run the customtkinter GUI."""
    root = ctk.CTk()
    args_chosen_by_user = ArgsChosenByUser()
    build_script_gui(root=root, args_chosen_by_user=args_chosen_by_user)
    root.mainloop()
    return generate_inputs(args_chosen_by_user=args_chosen_by_user)
