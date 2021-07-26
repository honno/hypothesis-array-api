from math import prod

from hypothesis import given
from hypothesis import strategies as st

import hypothesis_array as _xpst

from .common.debug import minimal
from .xputils import DTYPE_NAMES, create_array_module

xp = create_array_module()
xpst = _xpst.get_strategies_namespace(xp)


@given(xpst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@given(xpst.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["all"])


@given(xpst.boolean_dtypes())
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == xp.bool


@given(xpst.integer_dtypes())
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["ints"])


@given(xpst.unsigned_integer_dtypes())
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["uints"])


@given(xpst.floating_dtypes())
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["floats"])


@given(st.data())
def test_can_generate_arrays(data):
    dtype = data.draw(xpst.scalar_dtypes())
    shape = data.draw(xpst.array_shapes())
    array = data.draw(xpst.arrays(dtype, shape))

    assert array.dtype == dtype
    assert array.ndim == len(shape)
    assert array.shape == shape
    assert array.size == prod(shape)
    # TODO check array.__array_namespace__() exists once xputils is compliant


@given(st.data())
def test_can_draw_arrays_from_scalar_strategies(data):
    strategy = data.draw(st.sampled_from([
        xpst.scalar_dtypes(),
        xpst.boolean_dtypes(),
        xpst.integer_dtypes(),
        xpst.unsigned_integer_dtypes(),
        xpst.floating_dtypes(),
    ]))
    array = data.draw(xpst.arrays(strategy, ()))  # noqa
    # TODO check array.__array_namespace__()


@given(xpst.arrays(xp.bool, xpst.array_shapes()))
def test_can_draw_arrays_from_shapes_strategy(array):
    assert array.dtype == xp.bool
    # TODO check array.__array_namespace__()


@given(xpst.arrays(xp.bool, ()))
def test_empty_dimensions_are_arrays(array):
    # TODO check array.__array_namespace__()
    assert array.dtype == xp.bool
    assert array.ndim == 0
    assert array.shape == ()


@given(xpst.arrays(xp.bool, (1, 0, 1)))
def test_can_handle_zero_dimensions(array):
    assert array.dtype == xp.bool
    assert array.shape == (1, 0, 1)


@given(xpst.arrays(xp.uint32, (5, 5)), st.just(xp.zeros((5, 5), dtype=xp.int8)))
def test_generates_unsigned_ints(array, zeros):
    assert xp.all(array >= zeros)


def test_generates_and_minimizes():
    strategy = xpst.arrays(xp.float32, (2, 2))
    zeros = xp.zeros(shape=(2, 2))
    assert xp.all(minimal(strategy) == zeros)


def test_can_minimize_large_arrays():
    array = minimal(
        xpst.arrays(xp.uint32, 100),
        lambda x: xp.any(x) and not xp.all(x),
        timeout_after=60,
    )

    zeros = xp.zeros_like(array)
    ones = xp.ones_like(array)
    assert xp.all(xp.logical_or(array == zeros, array == ones))

    # nonzero() is optional for Array API libraries as of 2021-07-26
    if hasattr(xp, "nonzero"):
        nonzero_count = 0
        for nonzero_indices in xp.nonzero(array):
            nonzero_count += nonzero_indices.size
        assert nonzero_count in (1, array.size - 1)
