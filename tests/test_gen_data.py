from hypothesis import given
from hypothesis import strategies as st

import hypothesis_array as xpst

from .xputils import create_array_module

xp = create_array_module()
xpst.array_module = xp


@given(xpst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@given(xpst.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in xpst.DTYPE_NAMES["all"])


@given(xpst.boolean_dtypes())
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == xp.bool


@given(xpst.integer_dtypes())
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in xpst.DTYPE_NAMES["ints"])


@given(xpst.unsigned_integer_dtypes())
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in xpst.DTYPE_NAMES["uints"])


@given(xpst.floating_dtypes())
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in xpst.DTYPE_NAMES["floats"])


@given(st.data())
def test_can_draw_arrays_from_scalars(data):
    dtype = data.draw(xpst.scalar_dtypes())
    array = data.draw(xpst.arrays(dtype=dtype, shape=()))

    # TODO use array.__array_namespace__() once NumPy releases _array_api
    assert array.dtype == dtype


@given(st.data())
def test_can_draw_arrays_from_scalar_strategies(data):
    dtype_st_func = data.draw(
        st.sampled_from(
            [
                xpst.scalar_dtypes,
                xpst.boolean_dtypes,
                xpst.integer_dtypes,
                xpst.unsigned_integer_dtypes,
                xpst.floating_dtypes,
            ]
        )
    )
    data.draw(xpst.arrays(dtype=dtype_st_func(), shape=()))

    # TODO use array.__array_namespace__() once NumPy releases _array_api
    # TODO assert array.dtype in [<possible dtypes...>]


@given(xpst.arrays(dtype=xp.bool, shape=(42,)))
def test_can_generate_1d_arrays(array):
    assert array.dtype == xp.bool
    assert array.ndim == 1
    assert array.shape == (42,)
    assert array.size == 42
