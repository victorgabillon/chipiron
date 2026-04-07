"""Factory for game-specific SVG adapters."""

from atomheart.games.chess.board import BoardFactory

from chipiron.displays.checkers_svg_adapter import CheckersSvgAdapter
from chipiron.displays.chess_svg_adapter import ChessSvgAdapter
from chipiron.displays.integer_reduction_svg_adapter import IntegerReductionSvgAdapter
from chipiron.displays.morpion_svg_adapter import MorpionSvgAdapter
from chipiron.displays.svg_adapter_errors import UnregisteredSvgAdapterError
from chipiron.displays.svg_adapter_protocol import SvgGameAdapter
from chipiron.environments.types import GameKind


def make_svg_adapter(
    *, game_kind: GameKind, board_factory: BoardFactory
) -> SvgGameAdapter:
    """Construct an SVG adapter for the requested game kind."""
    match game_kind:
        case GameKind.CHESS:
            return ChessSvgAdapter(board_factory=board_factory)
        case GameKind.CHECKERS:
            return CheckersSvgAdapter()
        case GameKind.INTEGER_REDUCTION:
            return IntegerReductionSvgAdapter()
        case GameKind.MORPION:
            return MorpionSvgAdapter()
        case _:
            raise UnregisteredSvgAdapterError(game_kind=game_kind)
