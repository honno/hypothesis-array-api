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
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(am, name) for name in amst.DTYPE_NAMES["all"])


@given(amst.boolean_dtypes())
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == am.bool


@given(amst.integer_dtypes())
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(am, name) for name in amst.DTYPE_NAMES["ints"])


@given(amst.unsigned_integer_dtypes())
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(am, name) for name in amst.DTYPE_NAMES["uints"])


@given(amst.floating_dtypes())
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(am, name) for name in amst.DTYPE_NAMES["floats"])
