"""Module for test base tree exploration."""

from parsley import make_partial_dataclass_with_optional_paths

from chipiron.players.player_ids import PlayerConfigTag
import chipiron.scripts as scripts
from chipiron.scripts.base_tree_exploration.base_tree_exploration import (
    BaseTreeExplorationArgs,
)
from chipiron.scripts.factory import create_script


PartialOpMatchScriptArgs = make_partial_dataclass_with_optional_paths(
    cls=BaseTreeExplorationArgs
)


def test_base_tree_exploration() -> None:
    """Test base tree exploration."""
    print("Running the SCRIPT with config ")
    script_object: scripts.IScript = create_script(
        script_type=scripts.ScriptType.BASE_TREE_EXPLORATION,
        extra_args=PartialOpMatchScriptArgs(
            player_config_tag=PlayerConfigTag.UNIFORM_DEPTH,
        ),
        should_parse_command_line_arguments=False,
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()


if __name__ == "__main__":
    test_base_tree_exploration()
