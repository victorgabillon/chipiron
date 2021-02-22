from sortedcontainers import SortedDict


class UpdateInstructionsBatch:

    def __init__(self, dictionary={}):
        # batch is a dictionary of all the node from which a backward update should be started
        # it is a SortedDict where the keys involves the depth as the main sorting argument
        # this permits to easily give priority of update to the nodes with higher depth.
        # it should be less time consuming because and less a redundant update depth per depth from the back
        self.batch = SortedDict()

        for key in dictionary:
            self[key] = dictionary[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.batch[key] = value
        else:  # key is a node
            self.batch[(key.half_move, key.id, key)] = value

    def __getitem__(self, key):
        # assert(0==1)
        return self.batch[key]

    def __iter__(self):
        return iter(self.batch)

    def popitem(self):
        key, value = self.batch.popitem()
        return key[2], value  # return the node

    def __bool__(self):
        return bool(self.batch)

    def print_info(self):
        print('UpdateInstructionsBatch: batch contains')
        for key, update_information in self.batch:
            key.print_info()
            update_information.print_info()

    def merge(self, update_instructions_batch):
        if update_instructions_batch is not None:
            for key in update_instructions_batch:
                if key in self.batch:
                    new_update_information = UpdateInstructions()
                    new_update_information.merge(self.batch[key], update_instructions_batch[key])
                    self.batch[key] = new_update_information
                else:
                    self.batch[key] = update_instructions_batch[key]


class UpdateInstructions:

    def __init__(self):
        # all the instructions
        self.all_instructions_blocks = {}

    def __setitem__(self, key, value):
        self.all_instructions_blocks[key] = value

    def __getitem__(self, key):
        return self.all_instructions_blocks[key]

    def __iter__(self):
        return iter(self.all_instructions_blocks)

    def keys(self):
        return self.all_instructions_blocks.keys()

    def add_update_instructions_block(self, key, update_instructions_block):
        self.all_instructions_blocks[key] = update_instructions_block

    def merge(self, an_update_instruction, another_update_instruction):
        #an_update_instruction.print_info()
        #another_update_instruction.print_info()

        an_keys = set(an_update_instruction.keys())
        another_keys = set(another_update_instruction.keys())
        all_keys = an_keys | another_keys
        for key in all_keys:
            if key in an_keys:
                if key in another_keys:
                    new_update_block_type = type(an_update_instruction[key])
                    self[key] = new_update_block_type()
                    self[key].merge(an_update_instruction[key], another_update_instruction[key])
                else:
                    self[key] = an_update_instruction[key]
            else:
                self[key] = another_update_instruction[key]

    def print_info(self):
        print('printing info of update instructions')
        for block_key in self.all_instructions_blocks:
            print('key',block_key)
            self[block_key].print_info()

    def empty(self):
        for block_key in self.all_instructions_blocks:
            if not self[block_key].empty():
                return False
        return True


class BaseUpdateInstructionsBlock:
    def __init__(self,
                 node_sending_update=None,  # node(or None)
                 is_node_newly_over=None,  # boolean
                 new_value_for_node=None,  # boolean
                 new_best_move_for_node=None):  # boolean

        self.instruction_dict = {}
        self['children_with_updated_over'] = {node_sending_update} if is_node_newly_over else set()
        self['children_with_updated_value'] = {node_sending_update} if new_value_for_node else set()
        self['children_with_updated_best_move'] = self.children_with_updated_best_move = {
            node_sending_update} if new_best_move_for_node else set()

    def __setitem__(self, key, value):
        self.instruction_dict[key] = value

    def __getitem__(self, key):
        return self.instruction_dict[key]

    def __iter__(self):
        return iter(self.instruction_dict)

    def merge(self, an_update_instruction, another_update_instruction):
        self.instruction_dict = {}
        self['children_with_updated_value'] = \
            an_update_instruction['children_with_updated_value'] | another_update_instruction[
                'children_with_updated_value']
        self['children_with_updated_best_move'] = \
            an_update_instruction['children_with_updated_best_move'] | another_update_instruction[
                'children_with_updated_best_move']
        self['children_with_updated_over'] = \
            an_update_instruction['children_with_updated_over'] | another_update_instruction[
                'children_with_updated_over']

    def print_info(self):
        print('upInstructions printing')
        print(len(self['children_with_updated_value']), 'children_with_updated_value', end=' ')
        for child in self['children_with_updated_value']:
            print(child.id, end=' ')
        print('\n', len(self['children_with_updated_best_move']), 'children_with_updated_best_move:', end=' ')
        for child in self['children_with_updated_best_move']:
            print(child.id, end=' ')
        print('\n', len(self['children_with_updated_over']), 'children_with_updated_over', end=' ')
        for child in self['children_with_updated_over']:
            print(child.id, end=' ')
        print()

    def empty(self):
        """ returns if all the components are simultaneously empty"""
        empty_bool = not bool(self['children_with_updated_value']) \
                     and not bool(self['children_with_updated_best_move']) \
                     and not bool(self['children_with_updated_over'])
        return empty_bool
