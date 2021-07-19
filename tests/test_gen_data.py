from hypothesis import given

import hypothesis_array as xpst

from .array_module_utils import create_array_module

am = create_array_module()
xpst.array_module = am


@given(xpst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@given(xpst.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(am, name) for name in xpst.DTYPE_NAMES["all"])


@given(xpst.boolean_dtypes())
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == am.bool


@given(xpst.integer_dtypes())
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(am, name) for name in xpst.DTYPE_NAMES["ints"])


@given(xpst.unsigned_integer_dtypes())
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(am, name) for name in xpst.DTYPE_NAMES["uints"])


@given(xpst.floating_dtypes())
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(am, name) for name in xpst.DTYPE_NAMES["floats"])


@given(xpst.arrays(dtype=am.bool, shape=(42,)))
def test_can_generate_1d_arrays(array):
    assert array.dtype == am.bool
    assert array.ndim == 1
    assert array.shape == (42,)
    assert array.size == 42
