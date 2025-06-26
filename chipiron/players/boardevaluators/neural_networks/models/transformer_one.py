from dataclasses import dataclass
from typing import Any, Literal

import chess
import torch
import torch.nn as nn
from torch.nn import functional as F

from chipiron.players.boardevaluators.neural_networks.NNModelType import NNModelType
from chipiron.utils.chi_nn import ChiNN

number_of_squares = len(chess.SQUARES)
number_pieces_types = len(chess.PIECE_TYPES)
number_colors = len(chess.COLORS)
number_occupancy_types = (
    number_pieces_types * number_colors + 1
)  # could be empty (+1) or one of the pieces in black or white
len_square_tensor = number_of_squares * number_occupancy_types
len_all_possible_tensor_input = (
    len_square_tensor + 1
)  # +1 for the vector that will bear the output embedding
number_parallel_tracks = number_of_squares + 1


@dataclass()
class TransformerArgs:
    number_occupancy_types: int = number_occupancy_types
    len_square_tensor: int = len_square_tensor
    number_pieces_types: int = number_pieces_types

    type: Literal[NNModelType.Transformer] = NNModelType.Transformer
    n_embd: int = 27
    n_head: int = 3
    n_layer: int = 2
    dropout_ratio: float = 0.0

    def __str__(self) -> str:
        """
        Returns a string representation of the TransformerArgs instance.
        """
        return (
            f"TransformerArgs(n_embd={self.n_embd}, "
            f"n_head={self.n_head}, "
            f"n_layer={self.n_layer}, "
            f"dropout_ratio={self.dropout_ratio})"
        )

    def filename(self) -> str:
        """
        Generates a filename based on the TransformerArgs instance.
        """
        return (
            f"transformer_{self.n_embd}embd_"
            f"{self.n_head}head_{self.n_layer}layer_"
            f"{self.dropout_ratio:.2f}dropout"
        )


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, n_embd: int, head_size: int, dropout_ratio: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=True)
        self.query = nn.Linear(n_embd, head_size, bias=True)
        self.value = nn.Linear(n_embd, head_size, bias=True)

        self.dropout = nn.Dropout(dropout_ratio)
        # todo investigate gap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out: torch.Tensor = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(
        self, num_heads: int, head_size: int, dropout_ratio: float, n_embd: int
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size=head_size, n_embd=n_embd, dropout_ratio=dropout_ratio)
                for _ in range(num_heads)
            ]
        )  # (B, T, hs) * num_heads
        self.proj = nn.Linear(
            head_size * num_heads, n_embd
        )  # (B, T, hs) * num_heads -> (B, T, n_embd)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.empty(1)  # to make mypy and jit happy
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd: int, dropout_ratio: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_ratio),
        )

    # self.lin =   nn.Linear(n_embd, 4 * n_embd)
    # self.re1 =     nn.GELU()
    # self.lin2 =    nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # xx=self.re1(self.lin(x))
        # print('xxxx',xx)
        # a=self.lin2(xx)
        a: torch.Tensor = self.net(x)

        # print('x',x, a, 'yy',self.net[0].weight, 'rr', self.net[0].weight.grad)
        return a


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd: int, n_head: int, dropout_ratio: float) -> None:
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head, head_size, dropout_ratio=dropout_ratio, n_embd=n_embd
        )  # (B, T, n_embd) -> (B, T, n_embd)
        self.ffwd = FeedFoward(
            n_embd, dropout_ratio=dropout_ratio
        )  # (B, T, n_embd) -> (B, T, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('zzyyy',x,self.ln1(x))

        y = self.sa(self.ln1(x))
        x = x + y
        # print('yyy',y,x,self.ln2(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerOne(ChiNN):

    def __init__(self, args: TransformerArgs) -> None:
        super(TransformerOne, self).__init__()

        self.board_embedding_table = nn.Parameter(
            torch.randn(len_all_possible_tensor_input, args.n_embd)
        )

        self.blocks = nn.Sequential(
            *[
                Block(args.n_embd, n_head=args.n_head, dropout_ratio=args.dropout_ratio)
                for _ in range(args.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(args.n_embd)  # final layer norm
        self.lm_head = nn.Linear(number_parallel_tracks * args.n_embd, 1)

        self.tan_h = nn.Tanh()

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module: Any) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, indices: list[list[int]]) -> torch.Tensor:

        # idx and targets are both (B,T) tensor of integers
        y = self.board_embedding_table[
            indices, :
        ]  # (B,len_all_possible_tensor_input,n_embd)
        z = self.blocks(y)  # (B,T,C)
        # w = self.ln_f(z)  # (B,T,C)
        x: torch.Tensor = self.lm_head(z.flatten(start_dim=1))  # (B,T,vocab_size)
        # print("z",z,"oo", z.sum())

        x = self.tan_h(x)

        return x

    def compute_representation(
        self, node: Any, parent_node: Any, board_modifications: Any
    ) -> None:
        """
        Compute the input representation for the given node.

        Args:
            node (Any): Current node.
            parent_node (Any): Parent node.
            board_modifications (Any): Board modifications.
        """
        ...
        raise Exception(f"to be recoded in {__name__}")

    def get_nn_input(self, node: Any) -> None:
        """
        Get the input tensor for the given node.

        Args:
            node (Any): Current node.

        Returns:
            None
        """
        raise Exception(f"to be recoded in {__name__}")

    def print_param(self) -> None:
        """
        Print the parameters of the model.
        """
        print(
            "print param not implmeented but is it necessary given the summary thingy in pythorch?"
        )
