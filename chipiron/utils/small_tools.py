"""Document the module contains utility functions and classes for small tools."""

import copy
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import typing
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from chipiron.utils.logger import chipiron_logger

path = typing.Annotated[str | os.PathLike[str], "path"]


def mkdir_if_not_existing(folder_path: path) -> None:
    """Create a directory at the specified path.

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
            f"Creation of the directory {folder_path} failed with error {error} in file {__name__}\n with pwd {os.getcwd()}"
        )
    except FileExistsError as error:
        chipiron_logger.error(
            "the file already exists so no creation needed for %s, with error %s",
            folder_path,
            error,
        )
    else:
        chipiron_logger.info("Successfully created the directory %s ", folder_path)


def yaml_fetch_args_in_file(path_file: path) -> dict[Any, Any]:
    """Fetch arguments from a YAML file.

    Args:
        path_file: The path to the YAML file.

    Returns:
        A dictionary containing the arguments.

    """
    with open(path_file, encoding="utf-8") as file:
        args: dict[Any, Any] = yaml.load(file, Loader=yaml.FullLoader)
    return args


def dict_alphabetic_str(dic: dict[Any, Any]) -> str:
    """Convert a dictionary to a string with keys sorted alphabetically.

    Args:
        dic: The dictionary to convert.

    Returns:
        A string representation of the dictionary with keys sorted alphabetically.

    """
    string: str = ""
    for key, value in sorted(dic.items()):
        string += f" {key:>30} : {value}\n"
    return string


def unique_int_from_list(a_list: list[int | None]) -> int | None:
    """Generate a unique integer from a list of two integers.

    Args:
        a_list: A list of two integers.

    Returns:
        The unique integer generated using the Cantor pairing function.

    Raises:
        AssertionError: If the list does not contain exactly two elements.

    """
    assert len(a_list) == 2
    x = a_list[0]
    y = a_list[1]
    if x is None or y is None:
        return None
    return int(0.5 * (x + y) * (x + y + 1) + y)  # Cantor pairing function


def rec_merge_dic(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively merge two dictionaries.

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


def resolve_package_path(path_to_file: str | Path) -> str:
    """Replace 'package://' at the start of the path with the chipiron package root.

    Args:
        path_to_file (str or Path): Input path, possibly starting with 'package://'.

    Returns:
        str: Resolved absolute path.

    """
    if isinstance(path_to_file, Path):
        path_to_file = str(path_to_file)

    if path_to_file.startswith("package://"):
        relative_path = path_to_file[len("package://") :]
        resource = files("chipiron").joinpath(relative_path)

        if not resource.is_file() and not resource.is_dir():
            raise FileNotFoundError

        return str(resource)  # You can also use `.as_posix()` if you need POSIX format
    return str(path_to_file)


def get_package_root_path(package_name: str) -> str:
    """Get the root path of a package.".

    Args:
        package_name (str): The name of the package.

    Raises:
        ImportError: If the package cannot be found.

    Returns:
        str: The root path of the package.

    """
    spec: importlib.machinery.ModuleSpec | None = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        raise ImportError

    # Get the package directory, not just the __init__.py file
    return os.path.dirname(spec.origin)
