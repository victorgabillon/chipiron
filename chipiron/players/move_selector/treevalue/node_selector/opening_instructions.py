import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Self, ItemsView, ValuesView

import chess

import chipiron.players.move_selector.treevalue.nodes as nodes
from chipiron.players.move_selector.treevalue.nodes.utils import a_move_sequence_from_root


@dataclass(slots=True)
class OpeningInstruction:
    node_to_open: nodes.ITreeNode
    move_to_play: chess.Move

    def print_info(self) -> None:
        print(f'OpeningInstruction: node_to_open {self.node_to_open.id} at hm {self.node_to_open.half_move} | '
              f'a path from root to node_to_open is {a_move_sequence_from_root(self.node_to_open)} | '
              f'self.move_to_play {self.move_to_play}')


class OpeningInstructions:
    # todo do we need a dict? why not a set? verify

    batch: dict[Any, OpeningInstruction]

    def __init__(
            self,
            dictionary: dict[Any, OpeningInstruction] | None = None
    ) -> None:

        # here i use a dictionary because they are insertion ordered until there is an ordered set in python
        # order is important because some method give a batch where the last element in the batch are prioritary
        self.batch = {}

        if dictionary is not None:
            for key in dictionary:
                self[key] = dictionary[key]

    def __setitem__(
            self,
            key: Any,
            value: OpeningInstruction
    ) -> None:
        # key is supposed to be a tuple with (node_to_open,  move_to_play)
        assert (len(key) == 2)
        assert (isinstance(key, tuple))
        # print(key, type(key[1]))
        assert (isinstance(key[1], chess.Move))
        self.batch[key] = value

    def __getitem__(
            self,
            key: Any
    ) -> OpeningInstruction:
        # assert(0==1)
        return self.batch[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.batch)

    def __bool__(self) -> bool:
        return bool(self.batch)

    def merge(
            self,
            another_opening_instructions_batch: Self
    ) -> None:
        for opening_instruction_key, opening_instruction in another_opening_instructions_batch.items():
            if opening_instruction_key not in self.batch:
                self.batch[opening_instruction_key] = opening_instruction

    def pop_items(
            self,
            how_many: int,
            popped: Self
    ) -> None:
        how_many = min(how_many, len(self.batch))
        for pop in range(how_many):
            key, value = self.batch.popitem()
            popped[key] = value

    def values(self) -> ValuesView[OpeningInstruction]:
        return self.batch.values()

    def items(self) -> ItemsView[Any, OpeningInstruction]:
        return self.batch.items()

    def print_info(self) -> None:
        print('OpeningInstructionsBatch: batch contains', len(self.batch), 'elements:')
        for key, opening_instructions in self.batch.items():
            opening_instructions.print_info()

    def __len__(self) -> int:
        return len(self.batch)


def create_instructions_to_open_all_moves(
        moves_to_play: list[chess.Move],
        node_to_open: nodes.ITreeNode
) -> OpeningInstructions:
    opening_instructions_batch = OpeningInstructions()

    for move_to_play in moves_to_play:
        # at the moment it looks redundant keys are almost the same as values but its clean
        # the keys are here for fast and redundant proof insertion
        # and the values are here for clean data processing
        opening_instructions_batch[(node_to_open.id, move_to_play)] = OpeningInstruction(node_to_open,
                                                                                         move_to_play)
    #  node_to_open.non_opened_legal_moves.add(move_to_play)
    return opening_instructions_batch


class OpeningType(Enum):
    AllChildren: str = 'all_children'


class OpeningInstructor:

    def __init__(
            self,
            opening_type: OpeningType,
            random_generator: random.Random
    ) -> None:
        self.opening_type = opening_type
        self.random_generator = random_generator

    def all_moves_to_open(
            self,
            node_to_open: nodes.ITreeNode
    ) -> list[chess.Move]:
        if self.opening_type == OpeningType.AllChildren:
            node_to_open.all_legal_moves_generated = True
            moves_to_play = list(node_to_open.legal_moves)

            # this shuffling add randomness to the playing style
            self.random_generator.shuffle(moves_to_play)

        else:
            raise Exception('Hello-la')
        return moves_to_play
