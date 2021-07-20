from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np

# I don't use _array_api from github.com/numpy/numpy/pull/18585 yet as I'd
# rather work in the mode of not building NumPy from source to use this test
# suite

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
    }
    attributes.update(COMPLETE_DTYPE_MAP)

    for attr in attrs_to_del:
        del attributes[attr]

    for attr, value in assign_attrs:
        attributes[attr] = value

    return SimpleNamespace(**attributes)
