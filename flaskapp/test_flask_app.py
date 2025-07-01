from flaskapp.flask_app import get_move


def test_get_move():
    get_move(fen_="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")


if __name__ == "__main__":
    test_get_move()
    print("test passed!")
