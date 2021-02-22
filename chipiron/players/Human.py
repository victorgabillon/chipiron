from players.Player import Player
import chess
import time
import settings

class Human(Player):
    # at the momentit look like i dont need this calss i could go directly for the tree builder no? think later bout that? maybe one is for multi round and the other is not?

    def __init__(self, options, environment):
        self.human_played = False

    def get_move_from_player(self, board, timetoplay):

        settings.global_lock.release()
        while True:
            settings.global_lock.acquire()
            try:
                if  self.human_played:
                    break
            finally:
                settings.global_lock.release()

            time.sleep(.05)
           # print('y',self.human_played)
        #print('yx',self.human_played)
        self.human_played = False
        #print('ys',self.lock.locked())
        settings.global_lock.acquire()
        return None

    def print_info(self):
        super().print_info()
        print('type: Human')
