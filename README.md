# Chipiron

Chipiron is a Python library that plays chess


## Requirements

* Python 3.9
* pip
* curl
* tkinter

## Installation
The following command will install the python packaged listed in requirements.txt, download syzygy tables, install stockfish, retrieve datasets and gui images...

```console
make init
```

so you might want to git clone and create a virtual environment as follows:
<details>
<summary> click to see more </summary>
<br>

```console
git clone https://github.com/victorgabillon/chipiron.git
cd chipiron
virtualenv chipiron_env_test
source chipiron_env_test/bin/activate
make init
```

</details>


## Usage

### Base case with a gui to play against chipiron or watch it play 
Starts a gui to choose your options of play.
```console
python3 main_chipiron.py 
```

### Script: one match
This allows to play a match between two players.
```console
python3 main_chipiron.py --script_name one_match --config_file_name scripts/one_match/exp_options.yaml
```

### Script: learn nn from supervised datasets 
This learns Neural Networks from a database of labelled boards.
```console
python3 main_chipiron.py --script_name learn_nn --config_file_name scripts/learn_nn_supervised/exp_options.yaml
```

## Contributing


## License
[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)