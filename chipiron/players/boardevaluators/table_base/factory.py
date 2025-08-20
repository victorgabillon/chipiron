"""
Module to create a SyzygyTable object if the path to the syzygy tables exists,
otherwise return None.
"""

import os
from typing import Any, Protocol

from chipiron.players.boardevaluators.table_base.syzygy_python import SyzygyChiTable
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.path_variables import SYZYGY_TABLES_DIR

from .syzygy_rust import SyzygyRustTable
from .syzygy_table import SyzygyTable


def create_syzygy_python() -> SyzygyChiTable | None:
    """
    Create a SyzygyTable object if the path to the syzygy tables exists,
    otherwise return None.

    Returns:
        SyzygyTable | None: The created SyzygyTable object or None if the path does not exist.
    """
    syzygy_table: SyzygyChiTable | None
    is_exist: bool = os.path.exists(SYZYGY_TABLES_DIR)

    if is_exist:
        syzygy_table = SyzygyChiTable(path_to_table=SYZYGY_TABLES_DIR)
    else:
        chipiron_logger.warning("no folder found for syzygy tables so NOT using it")
        syzygy_table = None

    return syzygy_table


def create_syzygy_rust() -> SyzygyRustTable | None:
    """
    Create a SyzygyTable object if the path to the syzygy tables exists,
    otherwise return None.

    Returns:
        SyzygyTable | None: The created SyzygyTable object or None if the path does not exist.
    """
    syzygy_table: SyzygyRustTable | None
    is_exist: bool = os.path.exists(SYZYGY_TABLES_DIR)

    if is_exist:
        syzygy_table = SyzygyRustTable(path_to_table=SYZYGY_TABLES_DIR)
    else:
        chipiron_logger.warning("no folder found for syzygy tables so NOT using it")
        syzygy_table = None

    return syzygy_table


class SyzygyProvider(Protocol):
    def provide(
        self,
    ) -> SyzygyChiTable | None: ...


class SyzygyFactory(Protocol):
    def __call__(
        self,
    ) -> SyzygyTable[Any] | None: ...


def create_syzygy_factory(use_rust: bool) -> SyzygyFactory:
    """
    Create a SyzygyTable object
    """
    syzygy_factory: SyzygyFactory
    if use_rust:
        syzygy_factory = create_syzygy_rust
    else:
        syzygy_factory = create_syzygy_python

    return syzygy_factory


def create_syzygy(use_rust: bool) -> SyzygyTable[Any] | None:
    """
    Create a SyzygyTable object
    """
    syzygy_table: SyzygyTable[Any] | None
    if use_rust:
        syzygy_table = create_syzygy_rust()
    else:
        syzygy_table = create_syzygy_python()

    return syzygy_table
