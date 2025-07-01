from encodings.punycode import T
import json
import random

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap4

from chipiron.environments.chess.board import BoardChi, create_board_chi
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.players.factory import create_chipiron_player
from chipiron.players.move_selector.move_selector import MoveRecommendation
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.environments.chess.board.utils import fen


# Uncomment and populate this variable in your code:
PROJECT = "chipironchess"

# Build structured log messages as an object.
global_log_fields = {}

# Add log correlation to nest all log messages.
# This is only relevant in HTTP-based contexts, and is ignored elsewhere.
# (In particular, non-HTTP-based Cloud Functions.)
request_is_defined = "request" in globals() or "request" in locals()
if request_is_defined and request:
    trace_header = request.headers.get("X-Cloud-Trace-Context")

    if trace_header and PROJECT:
        trace = trace_header.split("/")
        global_log_fields["logging.googleapis.com/trace"] = (
            f"projects/{PROJECT}/traces/{trace[0]}"
        )

# Complete a structured log entry.
entry = dict(
    severity="NOTICE",
    message="This is the default display field.",
    # Log viewer accesses 'component' as jsonPayload.component'.
    component="arbitrary-property",
    **global_log_fields,
)

print(json.dumps(entry))


app = Flask(__name__)
bootstrap = Bootstrap4(app)


@app.route("/")
def index():
    return render_template("index.html")


# @app.route('/move/<int:depth>/')
@app.route("/move/<path:fen_>/", methods=["GET", "POST"])
def get_move(fen_: fen):
    print("Calculating...")
    print("fen", fen_, type(fen_))
    random_generator = random.Random()
    player = create_chipiron_player(
        implementation_args=ImplementationArgs(),
        universal_behavior=False,
        random_generator=random_generator,
        tree_move_limit=500,
    )

    move_reco: MoveRecommendation = player.select_move(
        fen_plus_history=FenPlusHistory(current_fen=fen_), seed_int=0
    )
    print("Move found!", move_reco.move)
    print()
    res = str(move_reco.move)
    del player
    del move_reco
    return res


@app.route("/test/<string:tester>")
def test_get(tester):
    return tester


@app.route("/about")
def about():
    """About route."""
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
