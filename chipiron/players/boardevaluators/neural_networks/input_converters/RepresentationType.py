from enum import Enum


class RepresentationType(str, Enum):
    NOBUG364 = '364_no_bug'
    BUG364 = '364_bug'
    NO = 'no'
