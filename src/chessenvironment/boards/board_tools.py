def convert_line(line, index):
    if len(line) == 0:
        return ''

    count = 0
    while index + count < 8 and line[count] == '1':
        count = count + 1

    if count == 0:
        return line[0] + convert_line(line[1:], index + 1)
    else:
        return str(count) + convert_line(line[count:], index + count)


def convert_to_fen(ascii_board):
    list_ascii_board = ascii_board.splitlines()
    fen = ''
    list_ascii_board2 = list_ascii_board[:-1]
    for line in list_ascii_board2:
        fen = fen + convert_line(line, 0) + '/'
    fen = fen[:-1]
    fen = fen + ' ' + list_ascii_board[-1]
    return fen
