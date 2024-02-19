import os

from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

@app.route("/")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello {name}!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))