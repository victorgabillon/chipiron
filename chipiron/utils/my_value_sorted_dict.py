from typing import Any


# todo 3.12 vartype
def sort_dic(
        dic: dict[Any, Any]
):
    """ sorting a dic   tionary by ascending order"""
    sorted_dic = dict(sorted(dic.items(), key=lambda item: item[1]))
    return sorted_dic
