from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField


class PlayForm(FlaskForm):
    color = SelectField(u'color', choices=[('White', "White",), ('Black', "Black")])
    player = SelectField(u'player', choices=[('RecurZipfBase3', "RecurZipfBase3",), ('Uniform', "Uniform"), ('Sequool', "Sequool")])
    strength = SelectField(u'strength', choices=[('1', "1",), ('2', "2"), ('3', "3")])
    submit = SubmitField("Play")


class WatchForm(FlaskForm):
    white_player = SelectField(u'player', choices=[('RecurZipfBase3', "RecurZipfBase3",), ('Uniform', "Uniform"), ('Sequool', "Sequool")])
    black_player = SelectField(u'player', choices=[('RecurZipfBase3', "RecurZipfBase3",), ('Uniform', "Uniform"), ('Sequool', "Sequool")])
    strength = SelectField(u'strength', choices=[('1', "1",), ('2', "2"), ('3', "3")])
    submit = SubmitField("Watch")
