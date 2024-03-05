from sortedcollections import ValueSortedDict
from chipiron.environments import HalfMove
from chipiron.players.move_selector.treevalue.nodes import ITreeNode


class Descendants:
    descendants_at_half_move: dict[HalfMove, dict[str, ITreeNode]]
    number_of_descendants: int
    number_of_descendants_at_half_move: dict[HalfMove, int]
    min_half_move: int | None
    max_half_move: int | None

    def __init__(self):
        self.descendants_at_half_move = {}
        self.number_of_descendants = 0
        self.number_of_descendants_at_half_move = {}
        self.min_half_move = None
        self.max_half_move = None

    def keys(self):
        return self.descendants_at_half_move.keys()

    def __setitem__(self, half_move, value):
        self.descendants_at_half_move[half_move] = value

    def __getitem__(self, half_move):
        return self.descendants_at_half_move[half_move]

    def __iter__(self):
        return iter(self.descendants_at_half_move)

    def get_count(self):
        return self.number_of_descendants

    def contains_node(self, node):
        if node.half_move in self.descendants_at_half_move and node.fast_rep in self[node.half_move]:
            return True
        else:
            return False

    def remove_descendant(self, node):
        half_move = node.half_move
        fen = node.fast_rep

        self.number_of_descendants -= 1
        self[half_move].pop(fen)
        self.number_of_descendants_at_half_move[half_move] -= 1
        if self.number_of_descendants_at_half_move[half_move] == 0:
            self.number_of_descendants_at_half_move.pop(half_move)
            self.descendants_at_half_move.pop(half_move)

    def empty(self):
        return self.number_of_descendants == 0

    def add_descendant(self, node):
        half_move: HalfMove = node.half_move
        fen: str = node.fast_rep

        if half_move in self.descendants_at_half_move:
            assert (fen not in self.descendants_at_half_move[half_move])
            self.descendants_at_half_move[half_move][fen] = node
            self.number_of_descendants_at_half_move[half_move] += 1
        else:
            self.descendants_at_half_move[half_move] = {fen: node}
            self.number_of_descendants_at_half_move[half_move] = 1
        self.number_of_descendants += 1

    def __len__(self):
        return len(self.descendants_at_half_move)

    def print_info(self):
        print('---here are the ', self.get_count(), ' descendants.')
        for half_move in self:
            print('half_move: ', half_move, '| (', self.number_of_descendants_at_half_move[half_move],
                  'descendants)')  # ,                  end='| ')
            for descendant in self[half_move].values():
                print(descendant.id, descendant.fast_rep, end=' ')
            print('')

    def print_stats(self):
        print('---here are the ', self.get_count(), ' descendants')
        for half_move in self:
            print('half_move: ', half_move, '| (', self.number_of_descendants_at_half_move[half_move], 'descendants)')

    def test(self):
        assert (set(self.descendants_at_half_move.keys()) == set(self.number_of_descendants_at_half_move))
        sum_ = 0
        for half_move in self:
            sum_ += len(self[half_move])
        assert (self.number_of_descendants == sum_)

        for half_move in self:
            assert (self.number_of_descendants_at_half_move[half_move] == len(self[half_move]))

    def test_2(self, root_node):

        all_descendants = root_node.get_descendants()

        # self.print_info()
        for d in all_descendants:
            if d.half_move not in self.descendants_at_half_move:
                assert (d.half_move in self.descendants_at_half_move)
            if d.fast_rep not in self.descendants_at_half_move[d.half_move]:
                assert (d.fast_rep in self.descendants_at_half_move[d.half_move])

        for half_move in self.descendants_at_half_move:
            for d in self[half_move].values():
                # print('P',d)
                assert (d in all_descendants)

    def test_2_nod(self, root_node):

        all_descendants = root_node.get_not_opened_descendants()
        # print(':sssssssss')
        # for i in all_descendants:
        #    print(':',i.id,i)
        # self.print_info()
        for d in all_descendants:
            if d.half_move not in self.descendants_at_half_move:
                print('{{', d.id, root_node.id)
                assert (d.fast_rep in self.descendants_at_half_move[d.half_move])

        for half_move in self.descendants_at_half_move:
            for d in self[half_move].values():
                # print('P',d)
                assert (d in all_descendants)


class RangedDescendants(Descendants):

    def __init__(
            self
    ) -> None:
        super().__init__()
        self.min_half_move = None
        self.max_half_move = None

    def __str__(self):
        string: str = ''
        for half_move in self:
            string += f'half_move: {half_move} | ({self.number_of_descendants_at_half_move[half_move]} descendants)\n'
            for descendant in self[half_move].values():
                string += f'{descendant.id} '
            string += '\n'
        return string

    def is_new_generation(
            self,
            half_move: HalfMove
    ) -> bool:
        if self.min_half_move is not None:
            return half_move == self.max_half_move + 1
        else:
            return True

    def is_in_the_current_range(self, half_move):
        if self.min_half_move is not None:
            return self.max_half_move >= half_move >= self.min_half_move
        else:
            return False

    def is_in_the_acceptable_range(self, half_move):
        # print('?',half_move)
        if self.min_half_move is not None and self.max_half_move is not None:
            # print('?w', half_move,self.max_half_move , self.min_half_move)
            return self.max_half_move + 1 >= half_move >= self.min_half_move
        else:
            #   print('s?', half_move)

            return True

    def add_descendant(self, node):
        half_move = node.half_move
        fen = node.fast_rep

        assert (self.is_in_the_acceptable_range(half_move))
        if self.is_in_the_current_range(half_move):
            # print('half move',half_move, self.max_half_move, self.min_half_move,
            #      [(len(self.descendants_at_half_move[half_move]),half_move) for half_move in self.descendants_at_half_move])
            if half_move in self.descendants_at_half_move[half_move]:
                assert (fen not in self.descendants_at_half_move[half_move])
            self.descendants_at_half_move[half_move][fen] = node
            self.number_of_descendants_at_half_move[half_move] += 1
        else:  # half_move == len(self.descendants_at_half_move)
            assert (self.is_new_generation(half_move))
            self.descendants_at_half_move[half_move] = {fen: node}
            self.number_of_descendants_at_half_move[half_move] = 1
            if self.max_half_move is not None:
                self.max_half_move += 1
            else:
                self.min_half_move = half_move
                self.max_half_move = half_move
        self.number_of_descendants += 1

    def remove_descendant(self, node):
        half_move = node.half_move
        fen = node.fast_rep

        self.number_of_descendants -= 1
        self[half_move].pop(fen)
        self.number_of_descendants_at_half_move[half_move] -= 1
        if self.number_of_descendants_at_half_move[half_move] == 0:
            self.number_of_descendants_at_half_move.pop(half_move)
            self.descendants_at_half_move.pop(half_move)
            if half_move == self.max_half_move:
                self.max_half_move -= 1
            if half_move == self.min_half_move:
                self.min_half_move += 1
            if self.max_half_move < self.min_half_move:
                self.max_half_move = None
                self.min_half_move = None
                assert (self.number_of_descendants == 0)

    def range(self):
        return range(self.min_half_move, self.max_half_move + 1)

    def update(self, new_descendants):
        really_new_descendants = RangedDescendants()

        for half_move in new_descendants.range():
            if half_move in self:
                really_new_descendants_keys = set(new_descendants[half_move].keys()).difference(
                    set(self[half_move].keys()))
            else:
                really_new_descendants_keys = new_descendants[half_move].keys()
            for key in really_new_descendants_keys:
                really_new_descendants.add_descendant(new_descendants[half_move][key])
                self.add_descendant(new_descendants[half_move][key])

        # really_new_descendants.print_info()

        return really_new_descendants

    def merge(self, descendant_1, descendant_2):

        half_moves_range = set(descendant_1.keys()) | set(descendant_2.keys())
        assert (len(half_moves_range) > 0)
        self.min_half_move = min(half_moves_range)
        self.max_half_move = max(half_moves_range)
        for half_move in half_moves_range:
            if descendant_1.is_in_the_current_range(half_move):
                if descendant_2.is_in_the_current_range(half_move):
                    #  print('dd',type(self.descendants_at_half_move),type())
                    # in python 3.9 we can use a |
                    self.descendants_at_half_move[half_move] = {**descendant_1[half_move], **descendant_2[half_move]}
                    self.number_of_descendants_at_half_move[half_move] = len(self[half_move])
                    assert (self.number_of_descendants_at_half_move[half_move] == len(
                        {**descendant_1[half_move], **descendant_2[half_move]}))
                else:
                    self.descendants_at_half_move[half_move] = descendant_1[half_move]
                    self.number_of_descendants_at_half_move[half_move] = \
                        descendant_1.number_of_descendants_at_half_move[half_move]
            else:
                self.descendants_at_half_move[half_move] = descendant_2[half_move]
                self.number_of_descendants_at_half_move[half_move] = descendant_2.number_of_descendants_at_half_move[
                    half_move]
            self.number_of_descendants += self.number_of_descendants_at_half_move[half_move]

    def test(self):
        super().test()
        if self.min_half_move is None:
            assert (self.max_half_move is None)
            assert (self.number_of_descendants == 0)
        else:
            for i in range(self.min_half_move, self.max_half_move + 1):
                assert (i in self.descendants_at_half_move.keys())
        for half_move in self:
            assert (self.is_in_the_current_range(half_move))

    def print_info(self):
        super().print_info()
        print('---here are the ', self.get_count(), ' descendants. min:', self.min_half_move, '. max:',
              self.max_half_move)


class SortedDescendants(Descendants):
    # todo is there a difference between sorted descendant nd sorted value descendant? below?
    def __init__(self):
        super().__init__()
        self.sorted_descendants_at_half_move = {}

    def update_value(self, node, value):
        #    print('###lll',self.sorted_descendants_at_half_move[node.half_move],value)
        # print('xsx',value,node,node.half_move)
        # self.print_info()
        # if node.half_move == 113:
        #  print('^^',self.descendants_at_half_move[node.half_move])

        self.sorted_descendants_at_half_move[node.half_move][node] = value

    def add_descendant(self, node, value):

        super().add_descendant(node)
        half_move = node.half_move

        if half_move in self.sorted_descendants_at_half_move:
            assert (node not in self.sorted_descendants_at_half_move[half_move])
            self.sorted_descendants_at_half_move[half_move][node] = value
        else:  # half_move == len(self.descendants_at_half_move)
            self.sorted_descendants_at_half_move[half_move] = {node: value}

        assert (self.contains_node(node))

    def test(self):
        super().test()
        # print('defge',len(self.sorted_descendants_at_half_move), len(self.descendants_at_half_move),self.sorted_descendants_at_half_move,self.descendants_at_half_move)
        assert (len(self.sorted_descendants_at_half_move) == len(self.descendants_at_half_move))

        assert (self.sorted_descendants_at_half_move.keys() == self.descendants_at_half_move.keys())
        for half_move in self.sorted_descendants_at_half_move:
            assert (len(self.sorted_descendants_at_half_move[half_move]) == len(
                self.descendants_at_half_move[half_move]))

    def print_info(self):
        super().print_info()
        print('sorted')
        for half_move in self:
            print('half_move: ', half_move, '| (', self.number_of_descendants_at_half_move[half_move],
                  'descendants)')  # ,                  end='| ')
            for descendant, value in self.sorted_descendants_at_half_move[half_move].items():
                print(descendant.id, descendant.fast_rep, '(' + str(value) + ')', end=' ')
            print('')

    def remove_descendant(self, node):

        super().remove_descendant(node)
        half_move = node.half_move
        self.sorted_descendants_at_half_move[half_move].pop(node)
        if half_move not in self.number_of_descendants_at_half_move:
            self.sorted_descendants_at_half_move.pop(half_move)
        # self.print_info()

        assert (not self.contains_node(node))

    def contains_node(self, node):
        reply_base = super().contains_node(node)
        if node.half_move in self.descendants_at_half_move and node in self.sorted_descendants_at_half_move[
            node.half_move]:
            rep = True
        else:
            rep = False
        assert (reply_base == rep)
        return rep


class SortedValueDescendants(Descendants):
    def __init__(self):
        super().__init__()
        self.sorted_descendants_at_half_move = {}

    def update_value(self, node, value):
        #    print('###lll',self.sorted_descendants_at_half_move[node.half_move],value)
        #   print('xsx',value)
        self.sorted_descendants_at_half_move[node.half_move][node] = value

    def add_descendant(self, node, value):
        super().add_descendant(node)
        half_move = node.half_move

        if half_move in self.sorted_descendants_at_half_move:
            assert (node not in self.sorted_descendants_at_half_move[half_move])
            self.sorted_descendants_at_half_move[half_move][node] = value
        #  print(type(self.sorted_descendants_at_half_move[half_move]))
        else:  # half_move == len(self.descendants_at_half_move)
            self.sorted_descendants_at_half_move[half_move] = ValueSortedDict({node: value})
        # print(type(self.sorted_descendants_at_half_move[half_move]))
        # self.print_info()

    def test(self):
        super().test()
        # print('defge',len(self.sorted_descendants_at_half_move), len(self.descendants_at_half_move),self.sorted_descendants_at_half_move,self.descendants_at_half_move)
        assert (len(self.sorted_descendants_at_half_move) == len(self.descendants_at_half_move))

        assert (self.sorted_descendants_at_half_move.keys() == self.descendants_at_half_move.keys())
        for half_move in self.sorted_descendants_at_half_move:
            assert (len(self.sorted_descendants_at_half_move[half_move]) == len(
                self.descendants_at_half_move[half_move]))

    def print_info(self):
        super().print_info()
        print('sorted')
        for half_move in self:
            print('half_move: ', half_move, '| (', self.number_of_descendants_at_half_move[half_move],
                  'descendants)')  # ,                  end='| ')
            for descendant, value in self.sorted_descendants_at_half_move[half_move].items():
                print(descendant.id, +'(' + value + ')', end=' ')
            print('')

    def remove_descendant(self, node):
        super().remove_descendant(node)
        half_move = node.half_move

        self.sorted_descendants_at_half_move[half_move].pop(node)
        if half_move not in self.number_of_descendants_at_half_move:
            self.sorted_descendants_at_half_move.pop(half_move)
