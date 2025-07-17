import shakmaty_python_binding

from .utils import moveUci


class RustMove:
    move: shakmaty_python_binding.MyMove
    uci_: moveUci

    def __init__(self, move: shakmaty_python_binding.MyMove, uci: moveUci) -> None:
        self.move = move
        self.uci_ = uci

    def is_zeroing(self) -> bool:
        return self.move.is_zeroing()

    def uci(self) -> moveUci:
        return self.uci_
