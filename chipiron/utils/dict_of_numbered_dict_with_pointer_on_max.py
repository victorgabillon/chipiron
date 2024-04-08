from typing import TypeVar, Generic

from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode

T_Key = TypeVar('T_Key', bound=ITreeNode)
T_Value = TypeVar('T_Value')


class DictOfNumberedDictWithPointerOnMax(Generic[T_Key, T_Value]):

    def __init__(self) -> None:
        self.half_moves: dict[int, dict[T_Key, T_Value]] = {}
        self.max_half_move: int | None = None

    def __setitem__(
            self,
            node: T_Key,
            value: T_Value
    ) -> None:
        half_move = node.half_move
        if self.max_half_move is None:
            self.max_half_move = half_move
        else:
            self.max_half_move = max(half_move, self.max_half_move)
        if half_move in self.half_moves:
            self.half_moves[half_move][node] = value
        else:
            self.half_moves[half_move] = {node: value}

        assert (self.max_half_move == max(self.half_moves))

    def __getitem__(self, node: T_Key) -> T_Value:
        return self.half_moves[node.half_move][node]

    def __bool__(self) -> bool:
        return bool(self.half_moves)

    def __contains__(self, node: T_Key) -> bool:
        if node.half_move not in self.half_moves:
            return False
        else:
            return node in self.half_moves[node.half_move]

    def popitem(self) -> tuple[T_Key, T_Value]:
        assert self.max_half_move is not None
        popped: tuple[T_Key, T_Value] = self.half_moves[self.max_half_move].popitem()
        if not self.half_moves[self.max_half_move]:
            del self.half_moves[self.max_half_move]
            if self.half_moves:
                self.max_half_move = max(self.half_moves.keys())
            else:
                self.max_half_move = None

        return popped

    # def sort_dic(self):
    #    self.dic = dict(sorted(self.dic.items(), key=lambda item: item[0]))
    # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
