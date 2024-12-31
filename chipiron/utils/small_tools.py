"""
This module contains utility functions and classes for small tools.
"""

import copy
import os
import sys
import typing
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Any

import dacite
import numpy as np
import numpy.typing as nptyping
import yaml

import chipiron
import chipiron as ch
from chipiron.utils.dataclass import IsDataclass

path = typing.Annotated[str | os.PathLike[str], 'path']
seed = typing.Annotated[int, "seed"]


def mkdir(
        folder_path: path
) -> None:
    """
    Create a directory at the specified path.

    Args:
        folder_path: The path to the directory.

    Raises:
        FileNotFoundError: If the parent directory does not exist.
        FileExistsError: If the directory already exists.
    """
    try:
        os.mkdir(folder_path)
    except FileNotFoundError as error:
        sys.exit(
            f"Creation of the directory {folder_path} failed with error {error} in file {__name__}\n with pwd {os.getcwd()}")
    except FileExistsError as error:
        print(f'the file already exists so no creation needed for {folder_path}, with error {error}  ')
    else:
        print(f"Successfully created the directory {folder_path} ")


def yaml_fetch_args_in_file(path_file: path) -> dict[Any, Any]:
    """
    Fetch arguments from a YAML file.

    Args:
        path_file: The path to the YAML file.

    Returns:
        A dictionary containing the arguments.

    """
    with open(path_file, 'r', encoding="utf-8") as file:
        args: dict[Any, Any] = yaml.load(file, Loader=yaml.FullLoader)
    return args


def dict_alphabetic_str(
        dic: dict[Any, Any]
) -> str:
    """
    Convert a dictionary to a string with keys sorted alphabetically.

    Args:
        dic: The dictionary to convert.

    Returns:
        A string representation of the dictionary with keys sorted alphabetically.

    """
    string: str = ''
    for key, value in sorted(dic.items()):
        string += ' {:>30} : {}\n'.format(key, value)
    return string


def unique_int_from_list(a_list: list[int | None]) -> int | None:
    """
    Generate a unique integer from a list of two integers.

    Args:
        a_list: A list of two integers.

    Returns:
        The unique integer generated using the Cantor pairing function.

    Raises:
        AssertionError: If the list does not contain exactly two elements.

    """
    assert (len(a_list) == 2)
    x = a_list[0]
    y = a_list[1]
    if x is None or y is None:
        return None
    else:
        return int(.5 * (x + y) * (x + y + 1) + y)  # Cantor pairing function


def rec_merge_dic(
        a: dict[Any, Any],
        b: dict[Any, Any]
) -> dict[Any, Any]:
    """
    Recursively merge two dictionaries.

    Args:
        a: The first dictionary.
        b: The second dictionary.

    Returns:
        The merged dictionary.

    """
    merged = copy.deepcopy(b)
    for key in a:
        if key in merged:
            if isinstance(a[key], dict) and isinstance(merged[key], dict):
                merged[key] = rec_merge_dic(a[key], merged[key])
        else:
            merged[key] = a[key]

    return merged


def nth_key[_T](
        dct: dict[_T, Any],
        n: int
) -> _T:
    """
    Get the nth key from a dictionary.

    Args:
        dct: The dictionary.
        n: The index of the key to retrieve.

    Returns:
        The nth key from the dictionary.

    """
    it = iter(dct)
    # Consume n elements.
    next(islice(it, n, n), None)
    # Return the value at the current position.
    # This raises StopIteration if n is beyond the limits.
    # Use next(it, None) to suppress that exception.
    return next(it)


def softmax(
        x: list[float],
        temperature: float
) -> nptyping.NDArray[np.float64]:
    """
    Compute softmax values for each set of scores in x.

    Args:
        x: The list of scores.
        temperature: The temperature parameter.

    Returns:
        The softmax values.

    """
    e_x: nptyping.NDArray[np.float64] = np.exp((x - np.max(x)) * temperature)
    res: nptyping.NDArray[np.float64] = e_x / e_x.sum(axis=0)  # only difference
    return res


@typing.dataclass_transform()
def fetch_args_modify_and_convert[_T_co:IsDataclass](
        path_to_file: path,  # path to a yaml file
        dataclass_name: type[_T_co],  # the dataclass into which the dictionary will be converted
        modification: dict[Any, Any] | None = None,  # modification to the dict extracted from the yaml file
) -> _T_co:
    """
    Fetch, modify, and convert arguments from a YAML file to a dataclass.

    Args:
        path_to_file: The path to the YAML file.
        dataclass_name: The dataclass into which the dictionary will be converted.
        modification: The modification to the dictionary extracted from the YAML file.

    Returns:
        An instance of the dataclass with the modified arguments.

    """
    if modification is None:
        modification = {}
    file_args: dict[Any, Any] = chipiron.utils.yaml_fetch_args_in_file(path_to_file)
    merged_args_dict: dict[Any, Any] = ch.tool.rec_merge_dic(file_args, modification)

    print('merged_args_dict', merged_args_dict)
    # formatting the dictionary into the corresponding dataclass
    dataclass_args: _T_co = dacite.from_dict(
        data_class=dataclass_name,
        data=merged_args_dict,
        config=dacite.Config(cast=[Enum])
    )

    return dataclass_args


@dataclass
class Interval:
    """
    Represents an interval with a minimum and maximum value.
    """
    min_value: float | None = None
    max_value: float | None = None


def intersect_intervals(
        interval_1: Interval,
        interval_2: Interval
) -> Interval | None:
    """
    Find the intersection of two intervals.

    Args:
        interval_1: The first interval.
        interval_2: The second interval.

    Returns:
        The intersection of the two intervals, or None if there is no intersection.

    Raises:
        AssertionError: If any of the intervals have missing values.

    """
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
    """
    Calculate the distance between a number and an interval.

    Args:
        value: The number.
        interval: The interval.

    Returns:
        The distance between the number and the interval.

    Raises:
        AssertionError: If the interval has missing values.

    """
    assert (interval.max_value is not None and interval.min_value is not None)
    if value < interval.min_value:
        return interval.min_value - value
    elif value > interval.max_value:
        return value - interval.max_value
    else:
        return 0


def unflatten(dictionary: dict[Any, Any]) -> dict[Any, Any]:
    result_dict: dict[Any, Any] = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict
