from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np

# TODO use numpy._array_api when the Array API support PR goes through:
#      github.com/numpy/numpy/pull/18585 yet as I'd

__all__ = [
    "DTYPE_NAMES",
    "COMPLETE_DTYPE_MAP",
    "create_array_module",
]

DTYPE_NAMES = {
    "all": [
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float32", "float64",
    ],
    "ints": ["int8", "int16", "int32", "int64"],
    "uints": ["uint8", "uint16", "uint32", "uint64"],
    "floats": ["float32", "float64"],
}


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
def create_array_module(
    *,
    assign_attrs: Tuple[Tuple[str, Any], ...] = (),
    attrs_to_del: Tuple[str, ...] = (),
):
    attributes = {
        "__name__": "mockpy",
        "iinfo": np.iinfo,
        "finfo": np.finfo,
        "asarray": np.asarray,
        "reshape": np.reshape,
        "empty": np.empty,
        "zeros": np.zeros,
        "ones": np.ones,
        "zeros_like": np.zeros_like,
        "ones_like": np.ones_like,
        "full": np.full,
        "any": np.any,
        "all": np.all,
        "nonzero": np.nonzero,
        "unique": np.unique,
        "logical_or": np.logical_or,
    }
    attributes.update(COMPLETE_DTYPE_MAP)

    for attr in attrs_to_del:
        del attributes[attr]

    for attr, value in assign_attrs:
        attributes[attr] = value

    return SimpleNamespace(**attributes)
