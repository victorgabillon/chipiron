from enum import Enum


class IndexComputationType(str, Enum):
    MinGlobalChange = 'min_global_change'
    MinLocalChange = 'min_local_change'
    RecurZipf = 'recurzipf'
