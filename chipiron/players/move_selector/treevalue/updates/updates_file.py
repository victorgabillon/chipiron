from dataclasses import dataclass

from chipiron.utils.dict_of_numbered_dict_with_pointer_on_max import DictOfNumberedDictWithPointerOnMax
from .index_block import IndexUpdateInstructionsBlock
from .value_block import ValueUpdateInstructionsBlock


class UpdateInstructionsBatch:
    # todo probably what we need to do is is that we should split the dic between half mmoves instaead of sorting it

    def __init__(self, dictionary={}):
        # batch is a dictionary of all the node from which a backward update should be started
        # it is a SortedDict where the keys involve the depth as the main sorting argument
        # this permits to easily give priority of update to the nodes with higher depth.
        # it should be less time-consuming because and less a redundant update depth per depth from the back
        # self.batch = MySortedDict()
        self.batch = DictOfNumberedDictWithPointerOnMax()
        for node in dictionary:
            self.batch[node] = dictionary[node]

    def __setitem__(self, node, value):
        self.batch[node] = value

    def __getitem__(self, key):
        return self.batch[key]

    def __contains__(self, node):
        return node in self.batch

    def __iter__(self):
        raise Exception(f'fail in {__name__}')

    def popitem(self):
        node, value = self.batch.popitem()
        return node, value  # return the node

    def __bool__(self):
        return bool(self.batch)

    def print_info(self):
        print('UpdateInstructionsBatch: batch contains')
        raise Exception(f'not implemented in {__name__}')

        # for key, update_information in self.batch:
        #    key.print_info()
        #    update_information.print_info()

    def merge(self, update_instructions_batch):
        if update_instructions_batch is not None:
            for half_move in update_instructions_batch.batch.half_moves:
                for node in update_instructions_batch.batch.half_moves[half_move]:
                    if half_move in self.batch.half_moves and node in self.batch.half_moves[half_move]:
                        new_update_information = UpdateInstructions()
                        new_update_information.merge(
                            self.batch.half_moves[half_move][node],
                            update_instructions_batch.batch.half_moves[half_move][node]
                        )
                        self.batch[node] = new_update_information
                    else:
                        self.batch[node] = update_instructions_batch[node]


@dataclass(slots=True)
class UpdateInstructions:
    value_block: ValueUpdateInstructionsBlock | None = None
    index_block: IndexUpdateInstructionsBlock | None = None

    def merge(
            self,
            an_update_instruction,
            another_update_instruction
    ):

        if an_update_instruction.value_block:
            if another_update_instruction.value_block:
                self.value_block = ValueUpdateInstructionsBlock()
                self.value_block.merge(
                    an_update_instruction.value_block,
                    another_update_instruction.value_block
                )
            else:
                self.value_block = an_update_instruction.value_block
        else:
            self.value_block = another_update_instruction.value_block

        if an_update_instruction.index_block:
            if another_update_instruction.index_block:
                self.index_block = IndexUpdateInstructionsBlock()
                self.index_block.merge(
                    an_update_instruction.index_block,
                    another_update_instruction.index_block
                )
            else:
                self.index_block = an_update_instruction.index_block
        else:
            self.index_block = another_update_instruction.index_block

    def print_info(self):
        print('printing info of update instructions')
        assert (self.index_block is not None and self.value_block is not None)
        self.value_block.print_info()
        self.index_block.print_info()

    def empty(self):
        assert (self.value_block is not None)
        return self.value_block.empty() and (self.index_block is None or self.index_block.empty())
