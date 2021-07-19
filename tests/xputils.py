from functools import lru_cache
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "COMPLETE_DTYPE_MAP",
    "create_array_module",
]

COMPLETE_DTYPE_MAP = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
}


@lru_cache()
def create_array_module(attrvals: Optional[Tuple[Tuple[str, Any], ...]] = None):
    class ArrayModule:
        __name__ = "mockpy"
        iinfo = np.iinfo
        finfo = np.finfo
        asarray = np.asarray

    array_module = ArrayModule()

    if attrvals is None:
        attrvals = tuple(COMPLETE_DTYPE_MAP.items())

    for attr, value in attrvals:
        setattr(array_module, attr, value)

    return array_module
