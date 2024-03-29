from typing import Any

from .comparable import CT


# todo 3.12 vartype
def sort_dic(
        dic: dict[Any, CT]
) -> dict[Any, CT]:
    """ sorting a dictionary by ascending order"""
    z = dic.items()
    a = sorted(z, key=lambda item: item[1])
    sorted_dic = dict(a)
    return sorted_dic
