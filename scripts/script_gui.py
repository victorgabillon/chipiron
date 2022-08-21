import tkinter as tk


def script_gui() -> tuple[str, dict]:
    root = tk.Tk()
    root.output = None
    # place a label on the root window
    root.title('chipiron')

    window_width: int = 800
    window_height: int = 300

    # get the screen dimension
    screen_width: int = root.winfo_screenwidth()
    screen_height: int = root.winfo_screenheight()

    # find the center point
    center_x: int = int(screen_width / 2 - window_width / 2)
    center_y: int = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    # root.iconbitmap('download.jpeg')

    message = tk.Label(root, text="What to do?")
    message.grid(column=0, row=0)

    # exit button
    exit_button = tk.Button(
        root,
        text='Exit',
        command=lambda: root.quit()
    )

    message = tk.Label(root, text="Play Black against chipiron strength:")
    message.grid(column=0, row=2)

    # Create the list of options
    options_list = ["1", "2", "3", "4"]

    # Variable to keep track of the option
    # selected in OptionMenu
    strength_value = tk.StringVar(root)

    # Set the default value of the variable
    strength_value.set("1")

    # Create the option menu widget and passing
    # the options_list and value_inside to it.
    strength_menu = tk.OptionMenu(root, strength_value, *options_list)
    strength_menu.grid(column=1, row=2)

    # plat button
    play_against_chipiron_button = tk.Button(
        root,
        text='!Play!',
        command=lambda: [play_against_chipiron(root, strength_value), root.destroy()]
    )

    # exit button
    watch_a_game_button = tk.Button(
        root,
        text='Watch a game',
        command=lambda: [watch_a_game(root), root.destroy()]
    )

    play_against_chipiron_button.grid(row=2, column=3)
    watch_a_game_button.grid(row=4, column=0)
    exit_button.grid(row=6, column=0)

    root.mainloop()
    gui_args: dict
    script_name: str
    match root.output['type']:
        case 'play_against_chipiron':
            tree_move_limit = 4 * 10 ** root.output['strength']
            gui_args = {'config_file_name': 'scripts/one_match/exp_options.yaml',
                        'seed': 0,
                        'gui': True,
                        'file_name_player_one': 'Stockfish.yaml',
                        'file_name_player_two': 'Human.yaml',
                        'file_name_match_setting': 'setting_duda.yaml',
                        'player_one': {'tree_builder': {'stopping_criterion': {'tree_move_limit': tree_move_limit}}}}
            script_name = 'one_match'
        case 'watch_a_game':
            gui_args = {'config_file_name': 'scripts/one_match/exp_options.yaml',
                        'seed': 0,
                        'gui': True,
                        'file_name_player_one': 'RecurZipfBase3.yaml',
                        'file_name_player_two': 'RecurZipfBase4.yaml',
                        'file_name_match_setting': 'setting_duda.yaml'}
            script_name = 'one_match'
        case other:
            raise f'Not a good name: {other}'

    return script_name, gui_args


def play_against_chipiron(root, strength_str):
    root.output = {'type': 'play_against_chipiron', 'strength': int(strength_str.get())}


def watch_a_game(root):
    root.output = {'type': 'watch_a_game'}
