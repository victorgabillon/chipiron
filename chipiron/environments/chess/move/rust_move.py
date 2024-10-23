import shakmaty_python_binding


class RustMove:
    move: shakmaty_python_binding.MyMove

    def is_zeroing(self)-> bool:
        return self.move.is_zeroing()