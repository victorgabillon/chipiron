import time

import chipiron.scripts as scripts

# we need to not use multiprocessing to be able to use pytest therefore use setting_cubo  and not setting_jime

start_time = time.time()
configs_base = [
    {
        'file_path_game_pickle': 'tests/integration_test/replay_games/games_0_W:RecurZipfBase3-vs-B:RecurZipfBase3.obj'
    }
]


def test_replay_match(
        configs=configs_base
):
    for config in configs:
        print(f'Running the SCRIPT with config {config}')
        script_object: scripts.IScript = scripts.create_script(
            script_type=scripts.ScriptType.ReplayMatch,
            extra_args=config,
            should_parse_command_line_arguments=False
        )

        # run the script
        script_object.run()

        # terminate the script
        script_object.terminate()


if __name__ == "__main__":
    test_replay_match()
    print('ALL OK for REPLAY')
