from math import prod

from hypothesis import given
from hypothesis import strategies as st
from pytest import mark, param

import hypothesis_array as xpst

from .xputils import create_array_module

np = create_array_module()
npst = xpst.get_strategies_namespace(np)

# Currently tests here will fail when running the whole test suite due to the
# monkey patching method of specifying np.


@given(npst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@given(npst.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(np, name) for name in xpst.DTYPE_NAMES)


@given(npst.boolean_dtypes())
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == np.bool


@given(npst.integer_dtypes())
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(np, name) for name in xpst.INT_NAMES)


@given(npst.unsigned_integer_dtypes())
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(np, name) for name in xpst.UINT_NAMES)


@given(npst.floating_dtypes())
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(np, name) for name in xpst.FLOAT_NAMES)


@given(st.data())
def test_can_generate_arrays(data):
    dtype = data.draw(npst.scalar_dtypes())
    shape = data.draw(npst.array_shapes())
    array = data.draw(npst.arrays(dtype, shape))

    assert array.dtype == dtype
    assert array.ndim == len(shape)
    assert array.shape == shape
    assert array.size == prod(shape)
    # TODO check array.__array_namespace__() exists once xputils is compliant


# TODO assert stuff in tests below

@mark.parametrize(
    "strategy",
    [
        param(npst.scalar_dtypes(), id="scalar"),
        param(npst.boolean_dtypes(), id="boolean"),
        param(npst.integer_dtypes(), id="signed_integer"),
        param(npst.unsigned_integer_dtypes(), id="unsigned_integer"),
        param(npst.floating_dtypes(), id="floating"),
    ],
)
def test_can_draw_arrays_from_scalar_strategies(strategy):
    @given(npst.arrays(strategy, ()))
    def test(_):
        pass

    test()


def test_can_draw_arrays_from_array_shapes():
    @given(npst.arrays(np.bool, npst.array_shapes()))
    def test(_):
        pass

    test()
