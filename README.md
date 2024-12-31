[![python](https://img.shields.io/badge/Python-3.11|3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
![Tests](https://github.com/victorgabillon/chipiron/actions/workflows/ci.yaml/badge.svg)
[![pytorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

# Chipiron

Chipiron is a Python library that plays chess

## Acknowledgments

This code is build on top of the chess library https://github.com/niklasf/python-chess
and its rust version :https://github.com/niklasf/shakmaty

It also integrates code from : https://github.com/oakmac/chessboardjs/

## Requirements

* Python 3.12
* pip
* curl
* tkinter
* make
* gdown

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
conda create chipiron3.1 python==3.11
conda activate chipiron3.11
conda install -c conda-forge tk=*=xft_*
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
python3 main_chipiron.py --script_name one_match --config_file_name scripts/one_match/inputs/base/exp_options.yaml
```

### Script: league

This allows to run a league of selected players and compute their Elo ratings.

```console
python3 main_chipiron.py --script_name league
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