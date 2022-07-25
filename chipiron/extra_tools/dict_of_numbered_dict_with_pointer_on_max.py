class DictOfNumberedDictWithPointerOnMax:

    def __init__(self):
        self.half_moves = {}
        self.max_half_move = None

    def __setitem__(self, node, value):
        half_move = node.half_move
        if self.max_half_move is None:
            self.max_half_move = half_move
        else:
            self.max_half_move = max(half_move, self.max_half_move)
        if half_move in self.half_moves:
            self.half_moves[half_move][node] = value
        else:
            self.half_moves[half_move] = {node: value}

        assert(self.max_half_move == max(self.half_moves))

    def __getitem__(self, node):
        return self.half_moves[node.half_move][node]

    def __bool__(self):
        return bool(self.half_moves)

    #def items(self):
    #    return self.dic.items()

    #def __len__(self):
    #    return len(self.dic)

    #def __iter__(self):
    #    return iter(self.dic)

    def __contains__(self, node):
        if node.half_move not in self.half_moves:
            return False
        else:
            return node in self.half_moves[node.half_move]


    def popitem(self):
        popped = self.half_moves[self.max_half_move].popitem()
        if not self.half_moves[self.max_half_move]:
            del self.half_moves[self.max_half_move]
            if  self.half_moves:
                self.max_half_move = max(self.half_moves.keys())
            else:
                self.max_half_move = None

        return popped

    #def sort_dic(self):
    #    self.dic = dict(sorted(self.dic.items(), key=lambda item: item[0]))
        # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
