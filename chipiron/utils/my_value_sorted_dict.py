"""
Module for sorting a dictionary by ascending order
"""

from typing import Any

from .comparable import Comparable


# todo 3.12 vartype
def sort_dic[CT: Comparable](dic: dict[Any, CT]) -> dict[Any, CT]:
    """
    Sorts a dictionary by ascending order of values.

    Args:
            dic (dict[Any, CT]): The dictionary to be sorted.

    Returns:
            dict[Any, CT]: The sorted dictionary.
    """
    z = dic.items()
    a = sorted(z, key=lambda item: item[1])
    sorted_dic = dict(a)
    return sorted_dic
