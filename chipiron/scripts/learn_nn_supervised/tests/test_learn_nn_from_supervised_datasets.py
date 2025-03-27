import chipiron.scripts as scripts
from chipiron.scripts.factory import create_script

configs = [
    {
        "nn_trainer_args": {
            "neural_network_folder_path": "chipiron/scripts/learn_nn_supervised/board_evaluators_common_training_data/"
            + "nn_pytorch/test_to_keep",
            "reuse_existing_model": False,
            "nn_architecture_file_if_not_reusing_existing_one": "chipiron/scripts/learn_nn_supervised/"
            + "board_evaluators_common_training_data/nn_pytorch/architectures/"
            + architecture_file,
        },
        "stockfish_boards_train_file_name": "chipiron/scripts/learn_nn_supervised/tests/small_dataset.pi",
        "stockfish_boards_test_file_name": "chipiron/scripts/learn_nn_supervised/tests/small_dataset.pi",
        "test": True,
    }
    for architecture_file in [
        "architecture_p1.yaml",
        "architecture_prelu_bug.yaml",
        "architecture_prelu_nobug.yaml",
        "architecture_transformerone.yaml",
    ]
]


def test_learn_nn() -> None:
    for config in configs:
        script_object: scripts.IScript = create_script(
            script_type=scripts.ScriptType.LearnNN,
            extra_args=config,
            should_parse_command_line_arguments=False,
        )

        # run the script
        script_object.run()

        # terminate the script
        script_object.terminate()


if __name__ == "__main__":
    test_learn_nn()
