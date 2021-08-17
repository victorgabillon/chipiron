import time
import global_variables

class Human:
    # at the momentit look like i dont need this calss i could go directly for the tree builder no? think later bout that? maybe one is for multi round and the other is not?

    def __init__(self, options, environment):
        self.human_played = False

    def get_move_from_player(self, board, timetoplay):

        global_variables.global_lock.release()
        while True:
            global_variables.global_lock.acquire()
            try:
                if  self.human_played:
                    break
            finally:
                global_variables.global_lock.release()

            time.sleep(.05)
           # print('y',self.human_played)
        #print('yx',self.human_played)
        self.human_played = False
#        print('yseded',self.lock.locked())
        global_variables.global_lock.acquire()
        return None

    def print_info(self):
        super().print_info()
        print('type: Human')
