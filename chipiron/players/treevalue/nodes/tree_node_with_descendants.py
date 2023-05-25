from chipiron.players.treevalue.nodes.tree_node_with_values import TreeNodeWithValue
from chipiron.players.treevalue.trees.descendants import RangedDescendants


class UpdateInstructionsDescendantsBlock:

    def __init__(self, new_descendants=None):
        self.new_descendants = new_descendants

    def merge(self, an_update_instruction, another_update_instruction):
        # set union is | in python 3.9 #todo change at some point
        # self.new_descendants = **an_update_instruction.new_descendants | **another_update_instruction.new_descendants
        self.new_descendants = RangedDescendants()
        self.new_descendants.merge(an_update_instruction.new_descendants, another_update_instruction.new_descendants)

    def print_info(self):
        print('printing update block descendants')
        self.new_descendants.print_info()

    def empty(self):
        return self.new_descendants.empty()


class NodeWithDescendants(TreeNodeWithValue):

    def __init__(self, board, half_move, id_number, parent_node):
        super().__init__(board, half_move, id_number, parent_node)

        # all the descendants as a list of set. one set per depth
        self.descendants = RangedDescendants()
        self.descendants.add_descendant(self)

    def get_descendants_at_depth(self, depth):
        return self.descendants[depth]

    def print_descendants(self):
        self.descendants.print_info()

    def update_descendants(self, new_descendants):
        really_new_descendants = self.descendants.update(new_descendants)
        return really_new_descendants

    def test(self):
        super().test()
        self.test_visit()

    def test_visit(self):
        #   print('testvisist')
        self.descendants.test()
        self.descendants.test_2(self)

    def dot_description(self):
        super_description = super().dot_description()
        return super_description + '\n num_desc: ' + str(self.descendants.number_of_descendants)

    def create_update_instructions_after_node_birth(self):
        update_instructions = super().create_update_instructions_after_node_birth()
        new_descendants_ = self.descendants
        update_instructions['descendants'] = UpdateInstructionsDescendantsBlock(new_descendants=new_descendants_)
        return update_instructions

    def perform_updates(self, updates_instructions):
        new_update_instructions = super().perform_updates(updates_instructions)

        updates_instructions.print_info()

        print('frgt',updates_instructions.all_instructions_blocks)
        if 'descendants' in updates_instructions:
            print('ppppppppppppppppppppppppppppppppppppr')
            # get the base block
            updates_instructions_block = updates_instructions['descendants']  # todo create a variable for the tag

            if not updates_instructions_block.empty():
                really_new_descendants = self.update_descendants(updates_instructions_block.new_descendants)

                if really_new_descendants.number_of_descendants > 0:
                    # create the new instructions for the parents
                    descendants_update_instructions_block = UpdateInstructionsDescendantsBlock(really_new_descendants)
                    new_update_instructions.all_instructions_blocks[
                        'descendants'] = descendants_update_instructions_block

        return new_update_instructions


class NodeWithDescendantsNoUpdate(NodeWithDescendants):
    def perform_updates(self, updates_instructions):
        return super().perform_updates(updates_instructions)

    def create_update_instructions_after_node_birth(self):
        return super().create_update_instructions_after_node_birth()
