import chipiron.scripts as scripts
from chipiron.scripts.factory import create_script


def test_learn_nn() -> None:
    script_object: scripts.IScript = create_script(
        script_type=scripts.ScriptType.LearnNN,
        extra_args={
            "config_file_name": "chipiron/scripts/learn_nn_supervised/tests/test_exp_options.yaml"
        },
        should_parse_command_line_arguments=False,
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    test_learn_nn()
