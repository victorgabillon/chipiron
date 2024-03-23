from dataclasses import dataclass


@dataclass
class NeuralNetBoardEvalArgs:
    nn_type: str
    nn_param_folder_name: str
