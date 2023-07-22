def sort_dic(dic: dict) -> dict:
    """ sorting a dictionary by ascending order"""

    sorted_dic: dict = dict(sorted(dic.items(), key=lambda item: item[1]))
    return sorted_dic



