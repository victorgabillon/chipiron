"""Module to convert an ascii board to a FEN string."""

from typing import AnyStr


def convert_line(line: AnyStr, index: int) -> str:
    """
    Convert a line of the ascii board to a FEN string.

    Args:
        line (AnyStr): The line of the ascii board.
        index (int): The starting index of the line.

    Returns:
        str: The converted FEN string.
    """
    if len(line) == 0:
        return ""

    count: int = 0
    while index + count < 8 and line[count] == "1":
        count = count + 1

    if count == 0:
        return str(line[0]) + convert_line(line[1:], index + 1)
    else:
        return str(count) + convert_line(line[count:], index + count)


def convert_to_fen(ascii_board: str | bytes) -> str:
    """
    Convert an ascii board to a FEN string.
    Args:
        ascii_board (str | bytes): The ascii board.
    Returns:
        str: The converted FEN string.
    """
    # Handle bytes input by converting to string
    if isinstance(ascii_board, bytes):
        ascii_board = ascii_board.decode("utf-8")

    list_ascii_board: list[str] = ascii_board.splitlines()
    fen: str = ""
    list_ascii_board2: list[str] = list_ascii_board[:-1]

    for line in list_ascii_board2:
        fen = fen + convert_line(line, 0) + "/"

    fen = fen[:-1]
    fen = fen + " " + str(list_ascii_board[-1])
    return fen
