from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

import numpy as np
from pytest import mark

import hypothesis_array as aast

aast.aa = np

np_dtypes = {
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

@mark.parametrize("dtype", np_dtypes.values(), ids=np_dtypes.keys())
def test_strategy_inference(dtype):
    strategy = aast.from_dtype(dtype)

    assert isinstance(strategy.example(), dtype)
    # TODO check all draws of a typical strategy run
