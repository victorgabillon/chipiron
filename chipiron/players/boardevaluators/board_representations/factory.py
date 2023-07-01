from .board_representation import Representation364Factory

def create_board_representation(args: dict) -> object:
    board_representation: object
    print('args', args)
    if args['neural_network']['representation'] == '364':
        board_representation = Representation364Factory()
    else:
        raise Exception(f'trying to create {args["representation"]}')

    return board_representation

