from functools import lru_cache
from typing import Union

import numpy as np
import torch
from hypothesis import given
from pytest import mark

import hypothesis_array as xpst

_am_supported_dtypes = {
    np: [
        # bool namespace is not the NumPy scalar np.bool_
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
    torch: [
        # asarray() not supported
        # also following dtypes not supported:
        # - torch.uint16
        # - torch.uint32
        # - torch.uint64
    ],
}
am_supported_dtypes = []
for am, dtypes in _am_supported_dtypes.items():
    for dtype in dtypes:
        am_supported_dtypes.append((am, dtype))


@lru_cache()
def builtin_from_dtype_name(name: str) -> Union[bool, int, float]:
    if name == "bool":
        return bool
    elif name in xpst.DTYPE_NAMES["ints"] or name in xpst.DTYPE_NAMES["uints"]:
        return int
    elif name in xpst.DTYPE_NAMES["floats"]:
        return float
    raise ValueError()


@mark.parametrize("am, dtype_name", am_supported_dtypes)
def test_strategy_inference(am, dtype_name):
    builtin = builtin_from_dtype_name(dtype_name)
    dtype = getattr(am, dtype_name)
    xpst.array_module = am
    strategy = xpst.from_dtype(dtype)

    @given(strategy)
    def test(value):
        assert isinstance(value, builtin)

    test()
