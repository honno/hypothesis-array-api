from functools import lru_cache
from typing import Type, Union

import numpy as np
import torch
from hypothesis import given
from pytest import mark

import hypothesis_array as xpst

_xp_supported_dtypes = {
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
xp_supported_dtypes = []
for xp, dtypes in _xp_supported_dtypes.items():
    for dtype in dtypes:
        xp_supported_dtypes.append((xp, dtype))


@lru_cache()
def builtin_from_dtype_name(name: str) -> Type[Union[bool, int, float]]:
    if name == "bool":
        return bool
    elif name in xpst.INT_NAMES or name in xpst.UINT_NAMES:
        return int
    elif name in xpst.FLOAT_NAMES:
        return float
    raise ValueError()


@mark.parametrize("xp, dtype_name", xp_supported_dtypes)
def test_strategy_inference(xp, dtype_name):
    builtin = builtin_from_dtype_name(dtype_name)
    dtype = getattr(xp, dtype_name)
    strategy = xpst.from_dtype(xp, dtype)

    @given(strategy)
    def test(value):
        assert isinstance(value, builtin)

    test()
