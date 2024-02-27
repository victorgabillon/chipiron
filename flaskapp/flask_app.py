from flask import Flask, render_template
import chess
from chipiron.environments.chess.board import BoardChi
from chipiron.players.factory import create_chipiron_player
app = Flask(__name__)
from chipiron.players.move_selector.move_selector import  MoveRecommendation

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/move/<int:depth>/<path:fen>/')
def get_move(depth, fen):
    print('depth',depth,type(depth))
    print("Calculating...")
    board : BoardChi = BoardChi()
    print('fen',fen,type(fen))
    board.set_starting_position(fen=fen)
    player = create_chipiron_player(depth)
    move_reco: MoveRecommendation = player.select_move(
            board=board,
            seed=0
    ) 
    print("Move found!", move_reco.move)
    print()
    return str(move_reco.move)


@app.route('/test/<string:tester>')
def test_get(tester):
    return tester


if __name__ == '__main__':
    app.run(debug=True)
