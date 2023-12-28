from enum import Enum


class NodeSelectorType(str, Enum):
    RecurZipfBase: str = 'RecurZipfBase'
    Sequool: str = 'Sequool'
    Uniform: str = 'Uniform'
