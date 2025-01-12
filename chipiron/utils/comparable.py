"""
This module defines the Comparable class, which is an abstract base class for objects that can be compared.

Classes that inherit from Comparable must implement the __lt__ method, which defines the less-than comparison operation.

"""

from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar


class Comparable(metaclass=ABCMeta):
    """
    An abstract base class for objects that can be compared.

    Subclasses of `Comparable` must implement the `__lt__` method to define the less-than comparison.

    Attributes:
        None

    Methods:
        __lt__(self, other: Any) -> bool: Abstract method that compares the object with another object and
         returns True if the object is less than the other object, False otherwise.
    """

    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...


CT = TypeVar("CT", bound=Comparable)
