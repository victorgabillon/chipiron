"""Tests that Parsley resolves PlayerConfigTag to PlayerArgs."""

from dataclasses import dataclass

from parsley_coco import Parsley, create_parsley

from chipiron.games.match.match_settings_args import MatchSettingsArgs
from chipiron.games.match.match_tag import MatchConfigTag
from chipiron.players.player_args import PlayerArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.utils.small_tools import get_package_root_path


@dataclass
class _PlayerArgContainer:
    player: PlayerArgs | PlayerConfigTag = PlayerConfigTag.RANDOM


@dataclass
class _MatchArgsContainer:
    match_args: MatchSettingsArgs | MatchConfigTag = MatchConfigTag.CUBO


def _create_parser() -> Parsley[_PlayerArgContainer]:
    return create_parsley(
        args_dataclass_name=_PlayerArgContainer,
        should_parse_command_line_arguments=False,
        package_name=get_package_root_path("chipiron"),
        verbosity=0,
    )


def _create_match_parser() -> Parsley[_MatchArgsContainer]:
    return create_parsley(
        args_dataclass_name=_MatchArgsContainer,
        should_parse_command_line_arguments=False,
        package_name=get_package_root_path("chipiron"),
        verbosity=0,
    )


def test_player_config_tags_parse_to_player_args() -> None:
    parser = _create_parser()

    tag: PlayerConfigTag
    for tag in PlayerConfigTag:
        print(f"Testing tag: {tag}, type: {type(tag)}")
        args = parser.parse_arguments(extra_args=_PlayerArgContainer(player=tag))
        assert isinstance(args.player, PlayerArgs), (
            f"{tag} did not resolve to PlayerArgs"
        )


def test_match_config_tags_parse_to_match_args() -> None:
    parser = _create_match_parser()

    tag: MatchConfigTag
    for tag in MatchConfigTag:
        print(f"Testing tag: {tag}, type: {type(tag)}")
        args = parser.parse_arguments(extra_args=_MatchArgsContainer(match_args=tag))
        assert isinstance(args.match_args, MatchSettingsArgs), (
            f"{tag} did not resolve to MatchSettingsArgs"
        )


if __name__ == "__main__":
    test_player_config_tags_parse_to_player_args()
    test_match_config_tags_parse_to_match_args()
    print("all tests passed")
