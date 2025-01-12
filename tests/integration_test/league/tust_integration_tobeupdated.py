import os

from chipiron import scripts

os.chdir("../../")
print(os.getcwd())
configs = [
    {
        "seed": 11,
        "gui": False,
        "file_name_player_one": "Sequool.yaml",
        "file_name_player_two": "RecurZipfBase3.yaml",
        "file_name_match_setting": "setting_jime.yaml",
        "profiling": False,
    },
]

for config in configs:
    script_object: scripts.Script = scripts.create_script(
        script_type=scripts.ScriptType.League, extra_args=config
    )

    # run the script
    script_object.run()

    # terminate the script
    script_object.terminate()

print("ALL OK fOR ONE MATCH")
