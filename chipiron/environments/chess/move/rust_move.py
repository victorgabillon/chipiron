import shakmaty_python_binding


class RustMove:
    move: shakmaty_python_binding.MyMove
    uci_: str

    def __init__(
            self,
            move: shakmaty_python_binding.MyMove,
            uci: str
    ) -> None:
        self.move = move
        self.uci_ = uci

    def is_zeroing(self) -> bool:
        return self.move.is_zeroing()

    def uci(self) -> str:
        return self.uci_
