# Chipiron Scripts

Every script in this folder should be run via `main_chipiron.py` and run from the ROOT_FOLDER. The type of script can be specified via the argument --script_name. Yaml config file can be specified via --config_file_name.



## Base case with a GUI to play against chipiron or watch it play

Starts a gui to choose your options of play.

```console
python3 chipiron/scripts/main_chipiron.py
```

## Other Scripts:

## Available Scripts

- **one_match**: Play a match between two players.
- **league**: Run a league of selected players and compute Elo ratings.
- **learn_nn**: Learn Neural Networks from a database of labelled boards.
- **learn_from_scratch_value_and_fixed_boards**: Train a neural network from scratch using value and fixed board datasets.
- **base_tree_exploration**: Run tree exploration benchmarks.
- **runtheleague**: Run a custom league script.
- **script_gui_custom**: Start a custom GUI for Chipiron (if enabled).

> For details and usage, see the examples below or refer to the main

### One match:

This allows to play a match between two players.

```console
python3 chipiron/scripts/main_chipiron.py --script_name one_match --config_file_name chipiron/scripts/one_match/inputs/base/exp_options.yaml
```

### League

This allows to run a league of selected players and compute their Elo ratings.





```console
python3 chipiron/scripts/main_chipiron.py --script_name league
```


### Lean a NN from a labelled dataset

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

