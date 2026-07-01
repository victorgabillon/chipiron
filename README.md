[![python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
![Tests](https://github.com/victorgabillon/chipiron/actions/workflows/ci.yaml/badge.svg)
[![pytorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Pyright](https://img.shields.io/badge/pyright-type%20checked-blue)](https://github.com/microsoft/pyright)

# Chipiron

Chipiron is a Python library that plays chess

## Acknowledgments

- Built on top of [python-chess](https://github.com/niklasf/python-chess) and its Rust version [shakmaty](https://github.com/niklasf/shakmaty)
- Integrates code from [chessboardjs](https://github.com/oakmac/chessboardjs/) in the website



## Play Online against Chipiron

Play at
[this website](https://chipiron-759534873716.europe-west1.run.app/) !


---

## Related Projects

Chipiron integrates and builds upon other personal repositories:

- [shakmaty_python_binding](https://github.com/victorgabillon/shakmaty_python_binding):
  Python bindings for the Rust chess library [shakmaty](https://github.com/niklasf/shakmaty).

- [parsley-coco](https://github.com/victorgabillon/parsley_coco):
  A generic parsing library, installed from PyPI as `parsley-coco` and
  imported in Python as `parsley`.

- [chipiron-website](https://github.com/victorgabillon/chipiron-website):
  Source code for the website where you can play against Chipiron online.



---

## Quickstart

```bash
git clone https://github.com/victorgabillon/chipiron.git
cd chipiron
make init
python3 -m chipiron.scripts.main_chipiron
```

---

## Requirements

* Python 3.13
* pip
* curl
* tkinter
* make
* libxcb-cursor0
* zstd (for decompressing Lichess PGN databases)
* cc (or else to build some rust dependencies)

**Optional:**
* Stockfish chess engine (can be installed with `make stockfish`)



## Installation

### Using Make

This will install Python dependencies, download syzygy tables, retrieve datasets, and GUI images:

```bash
make init
```

#### Optional: Install Stockfish Chess Engine

If you want to use the integrated Stockfish player for stronger AI opponents:

```bash
make stockfish
```

This downloads and installs Stockfish 16 (~40MB) to the correct location for chipiron integration.


#### (Optional) Using Conda

<details>
<summary>Click here for detailed instructions with conda</summary>

```bash
git clone https://github.com/victorgabillon/chipiron.git
cd chipiron
conda create --name chipiron3.13 python==3.13
conda activate chipiron3.13
conda install -c conda-forge tk=*=xft_* # to fix graphical problems with tkinter
make init
```
</details>

---

### Docker

The Chipiron Docker image is  published to [Docker Hub](https://hub.docker.com/r/victorgabillon/chipiron).

Simply pull the image

```bash
docker pull victorgabillon/chipiron:latest
```

Or, build the Docker image:
```bash
docker build -t chipiron-x11 .
```

To run the Docker container with GUI:
```bash
sudo docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network=host \
  -u $(id -u):$(id -g) \
  <image_name>
```

## Usage

Every script should be run via `main_chipiron.py` and run from the ROOT_FOLDER. The type of script can be specified via the argument --script_name. Yaml config file can be specified via --config_file_name.

### Base case with a GUI to play against chipiron or watch it play

Starts a gui to choose your options of play.

```console
python3 -m chipiron.scripts.main_chipiron
```

### Other Scripts:

#### One match:

This allows to play a match between two players.

```console
python3 -m chipiron.scripts.main_chipiron --script_name one_match --config_file_name src/chipiron/scripts/one_match/inputs/base/exp_options.yaml
```

#### League

This allows to run a league of selected players and compute their Elo ratings.





```console
python3 -m chipiron.scripts.main_chipiron --script_name league
```


#### Lean a NN from a labelled dataset

This learns Neural Networks from a database of labelled boards.
```console
python3 -m chipiron.scripts.main_chipiron --script_name learn_nn --config_file_name src/chipiron/scripts/learn_nn_supervised/exp_options.yaml
```


<!---### Script: learn nn from supervised datasets
This learns Neural Networks from a database of labelled boards.
```console
python3 main_chipiron.py --script_name learn_nn --config_file_name scripts/learn_nn_supervised/exp_options.yaml
```
-->




---

## Documentation

Full documentation is available at [chipiron.readthedocs.io](https://chipiron.readthedocs.io/en/latest/).

To build and view the documentation locally:

```bash
cd docs
make clean
make html
xdg-open _build/html/index.html
```

---

## Testing

The canonical quality gate is tox:

```bash
tox
```

If tox is not installed in your active Python 3.13 environment:

```bash
python -m pip install -e '.[dev]'
```

Useful focused runs:

```bash
tox -e tooling
tox -e py313
tox -e lint
tox -e typecheck
```

Tool versions for Ruff, Pylint, mypy, Pyright, tox, Black, build, and
pre-commit are pinned in `pyproject.toml`. `tox -e tooling` checks that CI and
pre-commit stay aligned with those pins.

For local IDEs, pre-commit, or manual commands outside tox, install the same
development extras into your active Python 3.13 environment:

```bash
python -m pip install -e '.[test,lint,typecheck,dev]'
```

Then you can run pytest directly:

```bash
python -m pytest
```

Ruff does not auto-fix in the default quality gates. To rewrite files
intentionally, run explicit fix commands:

```bash
python -m ruff check --fix src/chipiron tests
python -m ruff format src/chipiron tests
```

### Coverage

Some integration tests spawn subprocesses, including full Chipiron runs. To
ensure coverage is collected from those processes, coverage is configured in
`pyproject.toml` with `patch = ["subprocess"]`.

The pytest defaults already enable coverage. To spell out the same command:

```bash
python -m pytest --cov=chipiron --cov-config=pyproject.toml --cov-report=term-missing
```

If coverage appears empty, ensure:

- tests are not bypassing Python execution
- subprocesses are standard Python processes

### Integration Testing

For complete end-to-end validation, use the integration test scripts that test the entire installation process in a clean environment:

```bash
# Python version (comprehensive logging and reporting)
python scripts/integration_test.py

# Shell version (lightweight and fast)
./scripts/integration_test.sh

# Test with custom repository
python scripts/integration_test.py --repo-url https://github.com/your-fork/chipiron.git

# Keep temporary files for debugging
python scripts/integration_test.py --keep-temp --verbose
```

These scripts:
- Create a temporary conda environment
- Clone the repository fresh
- Run complete makefile installation
- Test all components (Stockfish, Syzygy tables, GUI, chess functionality)
- Generate detailed reports

See [`scripts/README.md`](scripts/README.md) for detailed documentation.

---

## Contributing

Please open issues or pull requests on GitHub.

**Development Requirements:**

For development and testing, use the extras declared in `pyproject.toml`:

```bash
python -m pip install -e '.[test,lint,typecheck,dev]'
```

This includes tools for:
- Type checking (mypy)
- Linting (pylint, ruff)
- Testing (pytest, tox)
- Documentation building (sphinx)
- Code formatting and development tools

The pre-commit hooks use the pinned Ruff from your active Python environment,
so install the development extras before installing the hooks:

```bash
pre-commit install
```


---

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or support, please open an [issue](https://github.com/victorgabillon/chipiron/issues).
