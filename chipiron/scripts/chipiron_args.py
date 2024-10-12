from dataclasses import dataclass


@dataclass
class ImplementationArgs:
    # whether to use the speedup given by board_modification.
    # If True the modifications are recorded when a move is played to update fast the tensor
    # representation in pytorch
    use_board_modification: bool = False

    # whether to use the speedup given using the rust version of the chess boards.
    # If True rust is used
    use_rust_boards: bool = False
