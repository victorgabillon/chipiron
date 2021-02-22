import settings

class FirstMoves:

    def __init__(self):
        self.first_moves = {}

    def __iter__(self):
        return iter(self.first_moves)

    def __setitem__(self, node, value):
        self.first_moves[node] = value

    def __getitem__(self, node):
        return self.first_moves[node]

    def add_first_move(self, node, parent_node):

       # print('~',node.id)
        #if parent_node is not None:
          #  print('@', parent_node.id,self.first_moves[parent_node])
        if parent_node is None:
            assert (node.id ==0)
            self.first_moves[node] = set()  # base case
        elif self.first_moves[parent_node] == set():
            self.first_moves[node] = {node}
        elif node in self.first_moves:
#            print('~s', self.first_moves[node])
            previous_first_move = self.first_moves[node].copy()

            self.first_moves[node].update(self.first_moves[parent_node].copy())
            node_descendants = node.get_descendants()
            for descendant in node_descendants:
                self.first_moves[descendant].update(self.first_moves[parent_node].copy())

            new_first_moves = self.first_moves[node].difference(previous_first_move)
           # print('@',new_first_moves,self.first_moves[node],previous_first_move)
            #assert(new_first_moves == set())
        else:
           # print('~as')
            self.first_moves[node] = self.first_moves[parent_node].copy()

        if settings.testing_bool:
            self.test_first_move_node(node)
            #self.test_first_move()

    def test_first_move(self):
        for node in self.first_moves:
            self.test_first_move_node(node)

    def test_first_move_node(self,node):
           # print('[', node.id)
           # print('o', self.first_moves[node])
            # if len(self.first_moves[node]):
            #     print('os',type(list(self.first_moves[node])[0]))
            #     print('ow',list(self.first_moves[node])[0].id)
           # print('@', self.test_get_first_moves(node))
            # if self.test_get_first_moves(node):
            #     print('@s', type(self.test_get_first_moves(node)))
            #     print('@w',self.test_get_first_moves(node),list(self.test_get_first_moves(node))[0].id)
            assert (self.first_moves[node] == self.test_get_first_moves(node))

    def test_get_first_moves(self, node):
      #  print('dddddddddddddddddddddddddddddddd',node.id,node.parent_nodes)
        des = set()
        if node.id == 0:
            return set()
        generation = node.parent_nodes
       # print('@##',generation)
        found =False
        while not found:
            #print(':',list(generation)[0].id)
            next_depth_generation = set()
            for node_ in generation:
               # print('e:', node_.id,node_.parent_nodes)
                    for next_generation_parent in node_.parent_nodes:
                        if next_generation_parent is not  None:
                            #print('s:', next_generation_parent.id, next_depth_generation, node_.parent_nodes)
                            next_depth_generation.add(next_generation_parent)
                            if None in next_generation_parent.parent_nodes:
                                des.add(node_)
                                found = True
                        else:
                            des.add(node)
                            found = True

            generation = next_depth_generation
           # assert(2==3)
        return des
