import numpy as np
import torch
from pytest import mark

import hypothesis_array as aast

# array api dtypes:
# - int8"
# - int16"
# - int32"
# - int64"
# - uint8"
# - uint16"
# - uint32"
# - uint64"
# - float32"
# - float64"
# - bool"


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
        # not supported:
        # - torch.uint16
        # - torch.uint32
        # - torch.uint64
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float32,
        torch.float64,
        torch.bool,
    ],
}
module_dtypes = []
for aa, dtypes in _module_dtypes.items():
    for dtype in dtypes:
        module_dtypes.append((aa, dtype))

@mark.parametrize("aa, dtype", module_dtypes)
def test_strategy_inference(aa, dtype):
    aast.aa = aa
    strategy = aast.from_dtype(dtype)

    assert isinstance(strategy.example(), dtype)
    # TODO check all draws of a typical strategy run
