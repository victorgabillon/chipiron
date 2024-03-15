import chess.syzygy
from chipiron.players.boardevaluators.over_event import Winner, HowOver, OverTags
import chipiron.players.move_selector.treevalue.nodes as nodes
import chipiron.environments.chess.board as boards
from chipiron.utils import path


class SyzygyTable:

    def __init__(
            self,
            path_to_table: path
    ):
        path_to_table_str: str = str(path_to_table)
        self.table_base = chess.syzygy.open_tablebase(path_to_table_str)

    def fast_in_table(
            self,
            board: boards.BoardChi
    ) -> bool:
        return board.number_of_pieces_on_the_board() < 6

    def in_table(
            self,
            board: boards.BoardChi
    ) -> bool:
        try:
            self.table_base.probe_wdl(board.board)
        except KeyError:
            return False
        return True

    def set_over_event(
            self,
            node: nodes.AlgorithmNode
    ) -> None:
        val: int = self.val(node.board)

        who_is_winner_ = Winner.NO_KNOWN_WINNER
        if val != 0:
            how_over_ = HowOver.WIN
            if val > 0:
                who_is_winner_ = node.player_to_move
            if val < 0:
                who_is_winner_ = Winner.WHITE if node.player_to_move == chess.BLACK else Winner.BLACK
        else:
            how_over_ = HowOver.DRAW

        node.minmax_evaluation.over_event.becomes_over(
            how_over=how_over_,
            who_is_winner=who_is_winner_
        )

    def val(
            self,
            board: boards.BoardChi
    ) -> int:
        # tablebase.probe_wdl Returns 2 if the side to move is winning, 0 if the position is a draw and -2 if the side to move is losing.
        val: int = self.table_base.probe_wdl(board.board)
        return val

    def get_over_tag(
            self,
            board: boards.BoardChi
    ) -> OverTags:
        val = self.table_base.probe_wdl(board.board)
        if val > 0:
            if board.turn == chess.WHITE:
                return OverTags.TAG_WIN_WHITE
            else:
                return OverTags.TAG_WIN_BLACK
        elif val == 0:
            return OverTags.TAG_DRAW
        else:
            if board.turn == chess.WHITE:
                return OverTags.TAG_WIN_BLACK
            else:
                return OverTags.TAG_WIN_WHITE

    def string_result(
            self,
            board: boards.BoardChi
    ) -> str:
        val = self.table_base.probe_wdl(board.board)
        player_to_move = 'white' if board.turn == chess.WHITE else 'black'
        if val > 0:
            return 'WIN for player ' + player_to_move
        elif val == 0:
            return 'DRAW'
        else:
            return 'LOSS for player ' + player_to_move

    def dtz(
            self,
            board: boards.BoardChi
    ) -> int:
        dtz: int = self.table_base.probe_dtz(board.board)
        return dtz

    def best_move(
            self,
            board: boards.BoardChi
    ) -> chess.Move:
        all_moves: list[chess.Move] = list(board.legal_moves)

        # avoid draws by 50 move rules in winning position, # otherwise look
        # for it to make it last and preserve pieces in case of mistake by opponent

        best_value = -1000000000000000000000

        assert all_moves
        best_move: chess.Move = all_moves[0]
        for move in all_moves:
            board_copy: boards.BoardChi = board.copy(stack=True)
            board_copy.board.push(move)
            val_player_next_board = self.val(board_copy)
            val_player_node = -val_player_next_board
            dtz_player_next_board = self.dtz(board_copy)
            dtz_player_node = -dtz_player_next_board
            if val_player_node > 0:  # winning position
                new_value = board.board.is_zeroing(move) * 100 - dtz_player_node + 1000
            elif val_player_node == 0:
                new_value = - board.board.is_zeroing(move) * 100 + dtz_player_node
            elif val_player_node < 0:
                new_value = - board.board.is_zeroing(move) * 100 + dtz_player_node - 1000
            # print('edeswswswaqq',str(move),val_player_node,new_value ,node.board.is_zeroing(move) ,self.dtz(next_board))

            if new_value > best_value:
                best_value = new_value
                best_move = move
        return best_move
