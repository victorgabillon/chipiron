# Chipiron

Chipiron is a Python library that plays chess


## Requirements

* Python 3.9
* pip

## Installation
The following command will install the python packaged listed in requirements.txt download syzygy tables and install stockfish.

```console
make init
```

## Usage

### Base case with a gui to play against chipiron or watch it play 
Starts a gui to choose your options of play.
```console
python3 chipiron/main-chipiron.py 
```

### Script: one match
This allows to play a match between two players.
```console
python3 chipiron/main-chipiron.py one_match --config_file_name chipiron/scripts/one_match/exp_options.yaml
```

### Script: learn nn from supervised datasets 
This learns Neural Networks from a database of labelled boards.
```console
python3 chipiron/main-chipiron.py learn_nn --config_file_name chipiron/scripts/learn_nn_supervised/exp_options.yaml
```

## Contributing


## License
[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)