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


def mkdir(folder_path):
    a=os.getcwd()
    print('os.getcwd()',os.getcwd())
    try:
        os.mkdir(folder_path)
    except FileNotFoundError as error:
        sys.exit(f"Creation of the directory {folder_path} failed with error {error} in file {__name__}")
    except FileExistsError as error:
        print(f'the file already exists so no creation needed for {folder_path} ')
    else:
        print(f"Successfully created the directory {folder_path} ")


def yaml_fetch_args_in_file(path_file: str) -> dict:
    with open(path_file, 'r', encoding="utf-8") as file:
        args: dict = yaml.load(file, Loader=yaml.FullLoader)
    return args


def dict_alphabetic_str(dic):
    string = ''
    for key, value in sorted(dic.items()):
        string += ' {:>30} : {}\n'.format(key, value)
    return string


def unique_int_from_list(a_list) -> int | None:
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
    print(a, b)
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


def fetch_args_modify_and_convert[ADataclass: IsDataclass](
        path_to_file: str | bytes | os.PathLike,  # path to a yaml file
        dataclass_name: Type[ADataclass],  # the dataclass into which the dictionary will be converted
        modification: dict | None = None,  # modification to the dict extracted from the yaml file
) -> ADataclass:
    if modification is None:
        modification = {}
    file_args: dict = ch.tool.yaml_fetch_args_in_file(path_to_file)
    merged_args_dict: dict = ch.tool.rec_merge_dic(file_args, modification)

    print('merged_args_dict', merged_args_dict)
    # formatting the dictionary into the corresponding dataclass
    dataclass_args: ADataclass = dacite.from_dict(data_class=dataclass_name,
                                                  data=merged_args_dict,
                                                  config=dacite.Config(cast=[Enum]))

    return dataclass_args
