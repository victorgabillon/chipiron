# Chipiron

Chipiron is a Python library that plays chess

## Installation

```console
pip -r ./requirements.txt
```

## Usage



### one match
This allows to play a match between two players.
```console
python chipiron/main-chipiron.py one_match --config_file_name chipiron/scripts/one_match/exp_options.yaml
```

### learn nn from supervised datasets 
This learns Neural Networks from a database of labelled boards.
```console
python chipiron/main-chipiron.py learn_nn --config_file_name chipiron/scripts/learn_nn_supervised/exp_options.yaml
```

## Contributing


## License
[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)