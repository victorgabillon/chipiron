import logging
import time

from parsley_coco import make_partial_dataclass_with_optional_paths

import chipiron.scripts as scripts
from chipiron.scripts.factory import create_script
from chipiron.scripts.replay_game.replay_game import ReplayScriptArgs
from chipiron.utils.logger import chipiron_logger, suppress_logging

# we need to not use multiprocessing to be able to use pytest therefore use setting_cubo  and not setting_jime


PartialOpReplayScriptArgs = make_partial_dataclass_with_optional_paths(
    cls=ReplayScriptArgs
)

start_time = time.time()
configs_base = [
    PartialOpReplayScriptArgs(
        file_game_report="tests/integration_test/replay_games/games_0_W:RecurZipfBase3-vs-B:RecurZipfBase3_game_report.yaml",
        gui=False,
    )
]


def test_replay_match(configs=configs_base) -> None:
    for config in configs:
        chipiron_logger.info(f"Running the SCRIPT with config {config}")

        with suppress_logging(
            chipiron_logger, level=logging.ERROR
        ):  # only warning and error logging during script run
            script_object: scripts.IScript = create_script(
                script_type=scripts.ScriptType.ReplayMatch,
                extra_args=config,
                should_parse_command_line_arguments=False,
            )

            # run the script
            script_object.run()

            # terminate the script
            script_object.terminate()


if __name__ == "__main__":
    test_replay_match()
    chipiron_logger.info("ALL OK for REPLAY")
