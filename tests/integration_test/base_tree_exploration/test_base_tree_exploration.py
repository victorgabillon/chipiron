import chipiron.scripts as scripts
from chipiron.scripts.factory import create_script


def test_base_tree_exploration() -> None:
    print(f"Running the SCRIPT with config ")
    script_object: scripts.IScript = create_script(
        script_type=scripts.ScriptType.BaseTreeExploration,
        should_parse_command_line_arguments=False,
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    test_base_tree_exploration()
