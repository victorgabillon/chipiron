[![python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Formatted with isort](https://img.shields.io/badge/isort-checked-green)](https://pycqa.github.io/isort/index.html)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
![Tests](https://github.com/victorgabillon/chipiron/actions/workflows/ci.yaml/badge.svg)
[![pytorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# Chipiron

Chipiron is a Python library that plays chess

## Acknowledgments

This code is build on top of the chess library https://github.com/niklasf/python-chess
and its rust version :https://github.com/niklasf/shakmaty

It also integrates code from : https://github.com/oakmac/chessboardjs/

## Play Online against Chipiron

Play at
[this website](https://chipiron-759534873716.europe-west1.run.app/) !


## Requirements

* Python 3.12
* pip
* curl
* tkinter
* make

## Installation

The following command will install the python packaged listed in requirements.txt, download syzygy tables, install
stockfish, retrieve datasets and gui images...

```console
make init
```

so you might want to git clone and create a virtual environment as follows:
<details>
<summary> Click here to see more detailed instructions with conda</summary>
<br>

```console
git clone https://github.com/victorgabillon/chipiron.git
cd chipiron
conda create --name chipiron3.12 python==3.12
`conda activate chipiron3.12
`conda install -c conda-forge tk=*=xft_*
make init
```

</details>

## Usage

### Base case with a GUI to play against chipiron or watch it play

Starts a gui to choose your options of play.

```console
python3 chipiron/scripts/main_chipiron.py
```

### Other Scripts:

#### One match:

This allows to play a match between two players.

```console
python3 chipiron/scripts/main_chipiron.py --script_name one_match --config_file_name chipiron/scripts/one_match/inputs/base/exp_options.yaml
```

#### League

This allows to run a league of selected players and compute their Elo ratings.





```console
python3 chipiron/scripts/main_chipiron.py --script_name league
```


#### Lean a NN from a labelled dataset

This learns Neural Networks from a database of labelled boards.
```console
python3 chipiron/scripts/main_chipiron.py --script_name learn_nn --config_file_name chipiron/scripts/learn_nn_supervised/exp_options.yaml
```


<!---### Script: learn nn from supervised datasets
This learns Neural Networks from a database of labelled boards.
```console
python3 main_chipiron.py --script_name learn_nn --config_file_name scripts/learn_nn_supervised/exp_options.yaml
```
-->





## Contributing

## License

[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)