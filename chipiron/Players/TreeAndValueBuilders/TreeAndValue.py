from Players.Player import Player
from Players.TreeAndValueBuilders.Trees.opening_instructions import OpeningInstructor
import random
import settings
from Players.BoardEvaluators.NN4_pytorch import transform_board, real_f, real_l
from Players.TreeAndValueBuilders.notations_and_statics import softmax
import time

class TreeAndValue(Player):
    # at the moment it look like i dont need this class i could go directly
    # for the tree builder no? think later bout that? maybe one is for multi round and the other is not?

    def __init__(self, arg):
        self.tree = None
        self.arg = arg
        self.tree_move_limit = arg['tree_move_limit'] if 'tree_move_limit' in arg else None
        self.opening_instructor = OpeningInstructor(arg['opening_type']) if 'opening_type' in arg else None

    def continue_exploring(self):
        if self.tree.root_node.is_over():
            return False
        bool_ = True
        if self.tree_move_limit is not None:
            bool_ = self.tree.move_count < self.tree_move_limit
        return bool_

    def print_info_during_move_computation(self):
        if self.tree.root_node.best_node_sequence:
            current_best_child = self.tree.root_node.best_node_sequence[0]
            current_best_move = self.tree.root_node.moves_children.inverse[current_best_child]
        else:
            current_best_move = '?'
        print('========= tree move counting:', self.tree.move_count, 'out of', self.tree_move_limit,
              '|',
              "{0:.0%}".format(self.tree.move_count / self.tree_move_limit),
              '| current best move:', current_best_move, '| current white value:',
              self.tree.root_node.value_white)  # ,end='\r')

    def recommend_move_after_exploration(self):

        # for debug we fix the choice in the next lines
        if settings.deterministic_behavior:
            print(' FIXED CHOICE FOR DEBUG')
            best_child = self.tree.root_node.get_all_of_the_best_moves(how_equal='considered_equal')[-1]
            print('We have as best: ', self.tree.root_node.moves_children.inverse[best_child])

        else:  # normal behavior
            print('~~', self.arg)
            if self.arg['move_selection_rule']['type'] == 'softmax':
                temperature = self.arg['move_selection_rule']['temperature']
                values = [self.tree.root_node.subjective_value_of(node) for node in
                          self.tree.root_node.moves_children.values()]
                softmax_ = softmax(values, temperature)
                print('££!!', values, len(values), type(values))
                print('##', softmax_, len(softmax_), type(softmax_))
                move_as_list = random.choices(list(self.tree.root_node.moves_children.keys()), weights=softmax_, k=1)
                best_move = move_as_list[0]
                print('$$',best_move)
            elif self.arg['move_selection_rule']['type'] == 'almost_equal':
                # find best first move allowing for random choice for almost equally valued moves.
                best_root_children = self.tree.root_node.get_all_of_the_best_moves(how_equal='almost_equal')
                print('We have', len(best_root_children), 'moves considered as equally best:', end='')
                for child in best_root_children:
                    print(self.tree.root_node.moves_children.inverse[child], end=' ')
                best_child = random.choice(best_root_children)
                best_move = self.tree.root_node.moves_children.inverse[best_child]
        return best_move

    def get_move_from_player(self, board, time):
        self.tree = self.create_tree(board)
        self.tree.add_root_node(board)

        self.count = 0
        while self.continue_exploring():
            assert (not self.tree.root_node.is_over())
            self.print_info_during_move_computation()

            opening_instructions_batch = self.choose_node_and_move_to_open()

            # self.tree.save_raw_data_to_file(self.count)
            # self.count += 1
            #
            # input("Press Enter to continue...")

            # opening_instructions_batch.print_info()
            while opening_instructions_batch and self.continue_exploring():
                #  opening_instructions_batch.print_info()

                if self.tree_move_limit is not None:
                    opening_instructions_subset = opening_instructions_batch.pop_items(
                        self.tree_move_limit - self.tree.move_count)
                else:
                    opening_instructions_subset = opening_instructions_batch

                self.tree.open_and_update(opening_instructions_subset)
                if settings.testing_bool:
                    self.tree.test_the_tree()

        if settings.testing_bool:
            self.tree.test_the_tree()

        if self.tree_move_limit is not None:
            assert self.tree_move_limit == self.tree.move_count or self.tree.root_node.is_over()
        #self.tree.save_raw_data_to_file()
        self.tree.print_some_stats()
        for move, child in self.tree.root_node.moves_children.items():
            print(move, self.tree.root_node.moves_children[move].value_white, child.over_event.simple_string())
        print('evaluation for white: ', self.tree.root_node.value_white)

        best_move = self.recommend_move_after_exploration()
        self.tree.print_best_line()  # todo maybe almost best choosen line no?

        if False:  # settings.learning_nn_bool:
            value_0_1 = min(max(self.tree.root_node.value_white, -1), 1) * .5 + .5

            input_layer = transform_board(board)

            if self.tree.root_node.is_over():
                # target_value_0_1 = value_player_to_move(board)
                target_value_0_1 = min(max(self.tree.root_node.subjective_value(), -1), 1) * .5 + .5
                print('**', target_value_0_1, input_layer)
                assert (value_0_1 == 0. or value_0_1 == .5 or value_0_1 == 1.)
                target_input_layer = None
                # assert(2==45)

                settings.nn_to_train.train_one_example(input_layer,
                                                       target_value_0_1,
                                                       target_input_layer)
                replay_buffer = settings.nn_replay_buffer_manager.replay_buffer
                replay_buffer.print_info()
                replay_buffer.add_element((input_layer,
                                           target_value_0_1,
                                           target_input_layer))
            else:
                target_value_0_1 = None
                next_best_board = self.tree.root_node.best_node_sequence[0]
                assert (self.tree.root_node.player_to_move != next_best_board.player_to_move)
                target_input_layer = transform_board(next_best_board.board)
                print('£$%^', input_layer, target_input_layer, self.tree.root_node.subjective_value(),
                      next_best_board.subjective_value())

            replay_buffer = settings.nn_replay_buffer_manager.replay_buffer
            replay_buffer.print_info()

            for count, element in enumerate(replay_buffer.choose_random_element(5)):
                replay_buffer.print_info()

                print('replay buffer element', count)
                input_layer_buffer = element[0]
                target_value_0_1_buffer = element[1]
                target_input_layer_buffer = element[2]
                settings.nn_to_train.train_one_example(input_layer_buffer, target_value_0_1_buffer,
                                                       target_input_layer_buffer)
            #

            print('^^', self.tree.root_node.board.chess_board, self.tree.root_node.value_white)
            print('&&&&', input_layer, value_0_1)
            # assert(2==4)
            settings.nn_replay_buffer_manager.save()

            values = [self.tree.root_node.subjective_value_of(node) for node in
                      self.tree.root_node.moves_children.values()]
            softmax_ = softmax(values, 20)
            print('££!!', values, len(values), type(values))
            print('##', softmax_, len(softmax_), type(softmax_))
            vh = random.choices(list(self.tree.root_node.moves_children.keys()), weights=softmax_, k=1)
            best_move = vh[0]
            print(best_move, type(best_move), vh, list(self.tree.root_node.moves_children.keys()),
                  random.choices(list(self.tree.root_node.moves_children.keys()), weights=softmax_, k=1))

        return best_move

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')
