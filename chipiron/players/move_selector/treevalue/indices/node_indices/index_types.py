"""
This module defines the enumeration for index computation types.
"""

from enum import Enum


class IndexComputationType(str, Enum):
    """
    Enumeration for index computation types.

    Attributes:
        MinGlobalChange (str): Represents the minimum global change computation type.
        MinLocalChange (str): Represents the minimum local change computation type.
        RecurZipf (str): Represents the recurzipf computation type.
    """

    MIN_GLOBAL_CHANGE = "min_global_change"
    MIN_LOCAL_CHANGE = "min_local_change"
    RECUR_ZIPF = "recurzipf"
