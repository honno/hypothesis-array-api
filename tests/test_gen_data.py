from hypothesis import given

import hypothesis_array as amst

from .array_module_utils import create_array_module

am = create_array_module()
amst.array_module = am


@given(amst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@given(amst.scalar_dtypes())
def test_can_generate_dtypes(dtype):
    assert dtype in [
        am.bool,
        am.int8, am.int16, am.int32, am.int64,
        am.uint8, am.uint16, am.uint32, am.uint64,
        am.float32, am.float64,
    ]
