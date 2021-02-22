import settings
from Players.BoardEvaluators.NN4_pytorch import transform_board, real_f, real_l

class Player():
    # the only difference between player and mover is the color? is this necessary and symplifiable?

    def __init__(self):
        pass

    def get_move(self, board, time):
        """ returns the best move computed by the player.
        The player has the option to ask the syzygy table to play it"""

        # if there is only one possible legal move in the position, do not think, choose it.
        all_legal_moves = list(board.get_legal_moves())
        if len(all_legal_moves) == 1:
            return all_legal_moves[0]

        # if the play with syzygy option is on test if the position is in the database to play syzygy
        if self.syzygy_play and self.syzygy_player.fast_in_table(board):
            print('Playing with Syzygy')
            best_move = self.syzygy_player.best_move(board)

            if False : # settings.learning_nn_bool:

                input_layer = transform_board(board)

                # target_value_0_1 = value_player_to_move(board)
                target_value_0_1 = min(max(self.syzygy_player.val(board), -1), 1) * .5 + .5
                print('**', target_value_0_1, input_layer)
                assert (target_value_0_1 == 0. or target_value_0_1 == .5 or target_value_0_1 == 1.)
                target_input_layer = None
                print('!!',input_layer,target_value_0_1)

                settings.nn_to_train.train_one_example(input_layer,
                                                       target_value_0_1,
                                                       target_input_layer)
                replay_buffer = settings.nn_replay_buffer_manager.replay_buffer
                replay_buffer.print_info()
                replay_buffer.add_element((input_layer,
                                           target_value_0_1,
                                           target_input_layer))

                for count, element in enumerate(replay_buffer.choose_random_element(1)):
                    print('replay buffer element', count)
                    input_layer_buffer = element[0]
                    target_value_0_1_buffer = element[1]
                    target_input_layer_buffer = element[2]
                    settings.nn_to_train.train_one_example(input_layer_buffer, target_value_0_1_buffer,
                                                           target_input_layer_buffer)
                #
                settings.nn_replay_buffer_manager.save()



        else:
            print('Playing with player (not Syzygy)')
            best_move = self.get_move_from_player(board, time)





        return best_move

    def set_color(self, color):
        self.color = color

    def print_info(self):
        pass
        # print('------------\nPlayer ',self.color)
