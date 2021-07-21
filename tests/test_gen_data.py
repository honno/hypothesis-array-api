from math import prod

from hypothesis import given
from hypothesis import strategies as st
from pytest import mark, param

import hypothesis_array as xpst

from .xputils import create_array_module

xp = create_array_module()

# Currently tests here will fail when running the whole test suite due to the
# monkey patching method of specifying xp.


@given(xpst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@given(xpst.scalar_dtypes(xp))
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in xpst.DTYPE_NAMES)


@given(xpst.boolean_dtypes(xp))
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == xp.bool


@given(xpst.integer_dtypes(xp))
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in xpst.INT_NAMES)


@given(xpst.unsigned_integer_dtypes(xp))
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in xpst.UINT_NAMES)


@given(xpst.floating_dtypes(xp))
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in xpst.FLOAT_NAMES)


@given(st.data())
def test_can_generate_arrays(data):
    dtype = data.draw(xpst.scalar_dtypes(xp))
    shape = data.draw(xpst.array_shapes())
    array = data.draw(xpst.arrays(xp, dtype, shape))

    assert array.dtype == dtype
    assert array.ndim == len(shape)
    assert array.shape == shape
    assert array.size == prod(shape)
    # TODO check array.__array_namespace__() exists once xputils is compliant


# TODO assert stuff in tests below

@mark.parametrize(
    "strategy",
    [
        param(xpst.scalar_dtypes(xp), id="scalar"),
        param(xpst.boolean_dtypes(xp), id="boolean"),
        param(xpst.integer_dtypes(xp), id="signed_integer"),
        param(xpst.unsigned_integer_dtypes(xp), id="unsigned_integer"),
        param(xpst.floating_dtypes(xp), id="floating"),
    ],
)
def test_can_draw_arrays_from_scalar_strategies(strategy):
    @given(xpst.arrays(xp, strategy, ()))
    def test(_):
        pass

    test()


def test_can_draw_arrays_from_array_shapes():
    @given(xpst.arrays(xp, xp.bool, xpst.array_shapes()))
    def test(_):
        pass

    test()
