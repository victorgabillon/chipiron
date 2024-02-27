import os

from flask import Flask, render_template, url_for, flash, redirect, jsonify
from flask_cors import CORS
from flask_wtf import FlaskForm

from forms import PlayForm, WatchForm
from flask_bootstrap import Bootstrap

from threading import Thread
from queue import Queue

commands = Queue()
import time

from flask_socketio import SocketIO,emit

def game_loop():
    while True:
        try:
            command = commands.get_nowait()
            print('tt', command)
            matchscript(command)

        except Exception:
            pass
        time.sleep(.2)  # TODO poll other things


def matchscript(command):
    config = {'seed': 11, 'gui': False, 'file_name_player_one': f'{command["white_player"]}.yaml',
              'file_name_player_two': f'{command["black_player"]}.yaml',
              'file_name_match_setting': 'setting_jime.yaml', 'profiling': False, 'print_svg_board_to_file': True}
    script_object: scripts.Script = scripts.create_script(
        script_type=scripts.ScriptType.OneMatch,
        extra_args=config
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


Thread(target=game_loop, daemon=True).start()

app = Flask(__name__)
#CORS(app)  # Enable CORS for all routes and origins
CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")

app.config['SECRET_KEY'] = 'vico'

boostrap = Bootstrap(app)


@app.route("/", methods=["GET", "POST"])
def hello_world():
    """Home route."""
    watch_form: FlaskForm = WatchForm()
    play_form: FlaskForm = PlayForm()

    if watch_form.validate_on_submit():
        watch = {
            "white_player": watch_form.white_player.data,
            "black_player": watch_form.black_player.data,
            "strength": watch_form.strength.data
        }
        commands.put(watch)
        flash(f'white_player {watch["white_player"]} '
              f'black_player {watch["black_player"]} strength {watch["strength"]}', 'success')
        return redirect(url_for('watch'))

    return render_template(
        'home.html',
        watch_form=watch_form,
        play_form=play_form
    )


@app.route("/watch")
def watch():
    """About route."""
    image_file = url_for('static', filename=os.path.join('images', 'my.svg'))
    return render_template('watch.html', image_file=image_file)


@app.route("/about")
def about():
    """About route."""
    return render_template('about.html')


from celery import shared_task
import scripts

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
