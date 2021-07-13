import numpy as np
import torch
from hypothesis import given
from pytest import mark

import hypothesis_array as amst

_module_dtypes = {
    np: [
        # bool namespace is not the NumPy scalar np.bool_
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ],
    torch: [
        # asarray() not supported
        # also following dtypes not supported:
        # - torch.uint16
        # - torch.uint32
        # - torch.uint64
    ],
}
module_dtypes = []
for array_module, dtypes in _module_dtypes.items():
    for dtype in dtypes:
        module_dtypes.append((array_module, dtype))


@mark.parametrize("array_module, dtype", module_dtypes)
def test_strategy_inference(array_module, dtype):
    amst.array_module = array_module
    strategy = amst.from_dtype(dtype)

    @given(strategy)
    def test(value):
        assert isinstance(value, dtype)

    test()
