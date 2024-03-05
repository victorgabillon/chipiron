import copy
import sys
import yaml
from itertools import islice
import numpy as np
from chipiron.utils.is_dataclass import IsDataclass
from typing import Type
from enum import Enum
import dacite
import chipiron as ch
import os
import typing
from typing import Any

import chipiron


path = typing.Annotated[str | os.PathLike[str], 'path']
seed = typing.Annotated[int, "seed"]


def mkdir(
        folder_path: path
) -> None:
    try:
        os.mkdir(folder_path)
    except FileNotFoundError as error:
        sys.exit(
            f"Creation of the directory {folder_path} failed with error {error} in file {__name__}\n with pwd {os.getcwd()}")
    except FileExistsError as error:
        print(f'the file already exists so no creation needed for {folder_path} ')
    else:
        print(f"Successfully created the directory {folder_path} ")


def yaml_fetch_args_in_file(path_file: path) -> dict[Any, Any]:
    with open(path_file, 'r', encoding="utf-8") as file:
        args: dict[Any, Any] = yaml.load(file, Loader=yaml.FullLoader)
    return args


def dict_alphabetic_str(dic):
    string = ''
    for key, value in sorted(dic.items()):
        string += ' {:>30} : {}\n'.format(key, value)
    return string


def unique_int_from_list(a_list: list[int | None]) -> int | None:
    # only coded for a list of 2 atm probably can be done recursively for larger lists
    assert (len(a_list) == 2)
    x = a_list[0]
    y = a_list[1]
    if x is None or y is None:
        return None
    else:
        return int(.5 * (x + y) * (x + y + 1) + y)  # Cantor pairing function


def rec_merge_dic(a, b):
    """recursively merges two dictionaries"""
    merged = copy.deepcopy(b)
    for key in a:
        if key in merged:
            if isinstance(a[key], dict) and isinstance(merged[key], dict):
                merged[key] = rec_merge_dic(a[key], merged[key])
        else:
            merged[key] = a[key]

    return merged


def nth_key(dct, n):
    it = iter(dct)
    # Consume n elements.
    next(islice(it, n, n), None)
    # Return the value at the current position.
    # This raises StopIteration if n is beyond the limits.
    # Use next(it, None) to suppress that exception.
    return next(it)


def softmax(x, temperature):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) * temperature)
    return e_x / e_x.sum(axis=0)  # only difference


# before 3.12
from typing import TypeVar

_T_co = TypeVar("_T_co", covariant=True, bound=IsDataclass)


def fetch_args_modify_and_convert(
        path_to_file: path,  # path to a yaml file
        dataclass_name: Type[_T_co],  # the dataclass into which the dictionary will be converted
        modification: dict[Any, Any] | None = None,  # modification to the dict extracted from the yaml file
) -> _T_co:
    if modification is None:
        modification = {}
    file_args: dict[Any, Any] = chipiron.utils.yaml_fetch_args_in_file(path_to_file)
    merged_args_dict: dict[Any, Any] = ch.tool.rec_merge_dic(file_args, modification)

    print('merged_args_dict', merged_args_dict)
    # formatting the dictionary into the corresponding dataclass
    dataclass_args: _T_co = dacite.from_dict(data_class=dataclass_name,
                                             data=merged_args_dict,
                                             config=dacite.Config(cast=[Enum]))

    return dataclass_args


# after 3.12


# def fetch_args_modify_and_convert[ADataclass: IsDataclass](
#         path_to_file: str | bytes | os.PathLike,  # path to a yaml file
#         dataclass_name: Type[ADataclass],  # the dataclass into which the dictionary will be converted
#         modification: dict | None = None,  # modification to the dict extracted from the yaml file
# ) -> ADataclass:
#     if modification is None:
#         modification = {}
#     file_args: dict = ch.tool.yaml_fetch_args_in_file(path_to_file)
#     merged_args_dict: dict = ch.tool.rec_merge_dic(file_args, modification)
#
#     print('merged_args_dict', merged_args_dict)
#     # formatting the dictionary into the corresponding dataclass
#     dataclass_args: ADataclass = dacite.from_dict(data_class=dataclass_name,
#                                                   data=merged_args_dict,
#                                                   config=dacite.Config(cast=[Enum]))
#
#     return dataclass_args

from dataclasses import dataclass


@dataclass
class Interval:
    min_value: float | None = None
    max_value: float | None = None


def intersect_intervals(
        interval_1: Interval,
        interval_2: Interval
) -> Interval | None:
    assert (interval_1.max_value is not None and interval_1.min_value is not None)
    assert (interval_2.max_value is not None and interval_2.min_value is not None)
    min_value: float = max(interval_1.min_value, interval_2.min_value)
    max_value: float = min(interval_1.max_value, interval_2.max_value)
    if max_value < min_value:
        return None
    else:
        interval_res = Interval(max_value=max_value, min_value=min_value)
        return interval_res


def distance_number_to_interval(
        value: float,
        interval: Interval
) -> float:
    assert (interval.max_value is not None and interval.min_value is not None)
    if value < interval.min_value:
        return interval.min_value - value
    elif value > interval.max_value:
        return value - interval.max_value
    else:
        return 0
