from src.players.boardevaluators.table_base.syzygy import SyzygyTable
import multiprocessing
import queue

def create_syzygy_thread():
    syzygy_table = SyzygyTable('')
   # syzygy_mailbox = multiprocessing.Manager().Queue()
    #syzygy_table_thread = SyzygyTableThread(syzygy_table, syzygy_mailbox)
    return syzygy_table