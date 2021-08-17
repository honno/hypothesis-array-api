from copy import copy
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np
import pytest

__all__ = ["xp", "create_array_module", "XP_IS_COMPLIANT"]

METHODS_MAP = {
    "iinfo": np.iinfo,
    "finfo": np.finfo,
    "asarray": np.asarray,
    "reshape": np.reshape,
    "empty": np.empty,
    "zeros": np.zeros,
    "ones": np.ones,
    "arange": np.arange,
    "full": np.full,
    "any": np.any,
    "all": np.all,
    "isfinite": np.isfinite,
    "nonzero": np.nonzero,
    "unique": np.unique,
    "sum": np.sum,
    "isnan": np.isnan,
    "broadcast_arrays": np.broadcast_arrays,
    "logical_or": np.logical_or,
}

DTYPES_MAP = {
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

CONSTANTS_MAP = {
    "nan": np.nan,
}

ATTRS_MAP = {
    "__name__": "mockpy",
    **METHODS_MAP,
    **DTYPES_MAP,
    **CONSTANTS_MAP,
}


@lru_cache()
def create_array_module(
    *,
    assign: Tuple[Tuple[str, Any], ...] = (),
    exclude: Tuple[str, ...] = (),
):
    attributes = copy(ATTRS_MAP)
    for attr in exclude:
        del attributes[attr]
    for attr, val in assign:
        attributes[attr] = val
    return SimpleNamespace(**attributes)


# We try importing the Array API namespace from NumPy first, which modern
# versions should include. If not available we default to our own mocked module,
# which should allow our test suite to still work. A constant is set accordingly
# to inform our test suite of whether the array module here is a mock or not.
try:
    with pytest.warns(UserWarning):
        from numpy import array_api as xp
    XP_IS_COMPLIANT = True
except ImportError:
    xp = create_array_module()
    XP_IS_COMPLIANT = False
