import chess
import chess.svg

from cairosvg import svg2png


# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#
#         self.setGeometry(100, 100, 1100, 1100)
#
#         self.widgetSvg = QSvgWidget(parent=self)
#         self.widgetSvg.setGeometry(10, 10, 1080, 1080)
#
#         self.chessboard = chess.Board()
#
#     def paintEvent(self, event):
#         self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
#         self.widgetSvg.load(self.chessboardSvg)


class DisplayOneGame:

    def __init__(self):
        super().__init__()

        # self.app = QApplication([])
        # self.window = MainWindow()
        # self.window.show()
        # self.app.exec()

    def displayBoard(self, board, round_, last_move, color_to_play):
        color = 'white' if  not color_to_play else 'black'
        if round_ == 1:
            color='starting'
        filenamepng = 'runs/BoardDisplays/output' + str(int(round_/2))+'-'+color+ '.png'
        filenametxt = 'runs/BoardDisplays/output' + str(int(round_/2))+'-'+color + '.txt'

        if board.chess_board.is_check():
            square_king_checked = board.chess_board.king(color_to_play)
        else:
            square_king_checked = None

        self.chessboardSvg = chess.svg.board(board.chess_board, lastmove=last_move, check=square_king_checked).encode(
            "UTF-8")
        svg2png(bytestring=self.chessboardSvg, write_to=   filenamepng )
        svg2png(bytestring=self.chessboardSvg, write_to='output2.png')

        with open(filenametxt, "w") as f:
            f.write(board.chess_board.unicode())

    #  img = mpimg.imread('output.png')
    #  imgplot = plt.imshow(img)
    # plt.show()
    # self.window.chessboard = board.chessboard
    # self.window .paintEvent()
