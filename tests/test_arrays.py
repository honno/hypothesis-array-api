# This file was part of and modifed from Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# ./CONTRIBUTING.rst for a full list of people who may hold copyright,
# and consult the git log of ./hypothesis-python/tests/numpy/test_gen_data.py
# if you need to determine who owns an individual contribution.
# ('.' represents the root of the Hypothesis git repository)
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from math import prod

from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument, Unsatisfiable
from pytest import mark, raises

from hypothesis_array import DTYPE_NAMES, get_strategies_namespace

from .common.debug import find_any, minimal
from .common.utils import fails_with
from .xputils import create_array_module

xp = create_array_module()
xps = get_strategies_namespace(xp)


@given(st.data())
def test_can_generate_arrays_from_scalars(data):
    dtype = data.draw(xps.scalar_dtypes())
    array = data.draw(xps.arrays(dtype, ()))

    assert array.dtype == dtype
    # TODO check array.__array_namespace__()


@given(st.sampled_from(DTYPE_NAMES), st.data())
def test_can_generate_arrays_from_scalar_names(name, data):
    array = data.draw(xps.arrays(name, ()))
    assert array.dtype == getattr(xp, name)
    # TODO check array.__array_namespace__()


@given(st.data())
def test_can_generate_arrays_from_shapes(data):
    shape = data.draw(xps.array_shapes())
    array = data.draw(xps.arrays(xp.bool, shape))

    assert array.ndim == len(shape)
    assert array.shape == shape
    assert array.size == prod(shape)
    # TODO check array.__array_namespace__()


@given(st.integers(0, 10), st.data())
def test_can_generate_arrays_from_int_shapes(size, data):
    array = data.draw(xps.arrays(xp.bool, size))

    assert array.ndim == 1
    assert array.shape == (size,)
    assert array.size == size
    # TODO check array.__array_namespace__()


@given(st.data())
def test_can_draw_arrays_from_scalar_strategies(data):
    strat = data.draw(st.sampled_from([
        xps.scalar_dtypes(),
        xps.boolean_dtypes(),
        xps.integer_dtypes(),
        xps.unsigned_integer_dtypes(),
        xps.floating_dtypes(),
    ]))
    array = data.draw(xps.arrays(strat, ()))  # noqa
    # TODO check array.__array_namespace__()


@given(
    st.lists(st.sampled_from(DTYPE_NAMES), min_size=1, unique=True), st.data()
)
def test_can_draw_arrays_from_scalar_name_strategies(names, data):
    array = data.draw(xps.arrays(st.sampled_from(names), ()))  # noqa
    # TODO check array.__array_namespace__()


@given(xps.arrays(xp.bool, xps.array_shapes()))
def test_can_draw_arrays_from_shapes_strategy(array):
    assert array.dtype == xp.bool
    # TODO check array.__array_namespace__()


@given(xps.arrays(xp.bool, st.integers(0, 100)))
def test_can_draw_arrays_from_integers_strategy_as_shape(array):
    assert array.dtype == xp.bool
    # TODO check array.__array_namespace__()


@given(xps.arrays(xp.bool, ()))
def test_empty_dimensions_are_arrays(array):
    # TODO check array.__array_namespace__()
    assert array.dtype == xp.bool
    assert array.ndim == 0
    assert array.shape == ()


@given(xps.arrays(xp.bool, (1, 0, 1)))
def test_can_handle_zero_dimensions(array):
    assert array.dtype == xp.bool
    assert array.shape == (1, 0, 1)


@given(xps.arrays(xp.uint32, (5, 5)))
def test_generates_unsigned_ints(array):
    assert xp.all(array >= 0)


def test_generates_and_minimizes():
    strat = xps.arrays(xp.float32, (2, 2))
    assert xp.all(minimal(strat) == 0)


def test_minimise_array_strategy():
    smallest = minimal(
        xps.arrays(xps.scalar_dtypes(), xps.array_shapes(max_dims=3, max_side=3)),
    )
    assert smallest.dtype == xp.bool
    assert not xp.any(smallest)


def test_can_minimize_large_arrays():
    array = minimal(
        xps.arrays(xp.uint32, 100),
        lambda x: xp.any(x) and not xp.all(x),
        timeout_after=60,
    )

    assert xp.all(xp.logical_or(array == 0, array == 1))

    # xp.nonzero() is optional for Array API libraries
    if hasattr(xp, "nonzero"):
        nonzero_count = 0
        for nonzero_indices in xp.nonzero(array):
            nonzero_count += nonzero_indices.size
        assert nonzero_count in (1, array.size - 1)


@given(xps.arrays(xp.int8, st.integers(0, 20), unique=True))
def test_array_values_are_unique(array):
    # xp.unique() is optional for Array API libraries
    if hasattr(xp, "unique"):
        unique_values = xp.unique(array)
        assert unique_values.size == array.size


def test_cannot_generate_unique_array_of_too_many_elements():
    strat = xps.arrays(xp.int8, 10, elements=st.integers(0, 5), unique=True)
    with raises(Unsatisfiable):
        strat.example()


def test_cannot_fill_with_non_castable_value():
    strat = xps.arrays(xp.int8, 10, fill=st.just("not a castable value"))
    with raises(InvalidArgument):
        strat.example()


@given(
    xps.arrays(
        dtype=xp.float32,
        shape=st.integers(0, 20),
        elements=st.just(0.0),
        fill=st.just(xp.nan),
        unique=True,
    )
)
def test_array_values_are_unique_high_collision(array):
    hits = (array == 0.0)
    nzeros = 0
    for i in range(array.size):
        if hits[i]:
            nzeros += 1
    assert nzeros <= 1


@given(xps.arrays(xp.int8, (4,), elements=st.integers(0, 3), unique=True))
def test_generates_all_values_for_unique_array(array):
    # xp.unique() is optional for Array API libraries
    if hasattr(xp, "unique"):
        unique_values = xp.unique(array)
        assert unique_values.size == array.size


def test_may_fill_with_nan_when_unique_is_set():
    find_any(
        xps.arrays(
            dtype=xp.float32,
            shape=10,
            elements=st.floats(allow_nan=False),
            unique=True,
            fill=st.just(xp.nan),
        ),
        lambda x: xp.any(xp.isnan(x)),
    )


@fails_with(InvalidArgument)
@given(
    xps.arrays(
        dtype=xp.float32,
        shape=10,
        elements=st.floats(allow_nan=False),
        unique=True,
        fill=st.just(0.0),
    )
)
def test_may_not_fill_with_non_nan_when_unique_is_set(_):
    pass


@mark.parametrize(
    "kwargs",
    [
        {"elements": st.just(300)},
        {"elements": st.nothing(), "fill": st.just(300)},
    ],
)
@fails_with(InvalidArgument)
@given(st.data())
def test_may_not_use_overflowing_integers(kwargs, data):
    strat = xps.arrays(dtype=xp.int8, shape=1, **kwargs)
    data.draw(strat)


@mark.parametrize("fill", [False, True])
@mark.parametrize(
    "dtype, strat",
    [
        (xp.float32, st.floats(min_value=10 ** 40, allow_infinity=False)),
        (xp.float64, st.floats(min_value=10 ** 40, allow_infinity=False)),
    ]
)
@fails_with(InvalidArgument)
@given(st.data())
def test_may_not_use_unrepresentable_elements(fill, dtype, strat, data):
    if fill:
        kw = {"elements": st.nothing(), "fill": strat}
    else:
        kw = {"elements": strat}
    strat = xps.arrays(dtype=dtype, shape=1, **kw)
    data.draw(strat)


@given(
    xps.arrays(dtype=xp.float32, shape=10, elements={"min_value": 0, "max_value": 1})
)
def test_floats_can_be_constrained_at_low_width(array):
    assert xp.all(array >= 0)
    assert xp.all(array <= 1)


@given(
    xps.arrays(
        dtype=xp.float32,
        shape=10,
        elements={
            "min_value": 0,
            "max_value": 1,
            "exclude_min": True,
            "exclude_max": True,
        },
    )
)
def test_floats_can_be_constrained_at_low_width_excluding_endpoints(array):
    assert xp.all(array > 0)
    assert xp.all(array < 1)


def count_unique(array):
    """Returns the number of unique elements.
    NaN values are treated as unique to eachother.

    The Array API doesn't specify how ``unique()`` should behave for Nan values,
    so this method provides consistent behaviour.
    """
    n_unique = 0

    nan_index = xp.isnan(array)
    for isnan, count in zip(*xp.unique(nan_index, return_counts=True)):
        if isnan:
            n_unique += count
            break

    filtered_array = array[~nan_index]  # TODO Array API makes boolean indexing optinal
    unique_array = xp.unique(filtered_array)
    n_unique += unique_array.size

    return n_unique


@given(
    xps.arrays(
        dtype=xp.float32,
        elements=st.just(xp.nan),
        shape=xps.array_shapes(),
    )
)
def test_count_unique(array):
    assert count_unique(array) == array.size


@given(
    xps.arrays(
        dtype=xp.float32,
        elements=st.floats(allow_nan=False, width=32),
        shape=10,
        unique=True,
        fill=st.just(xp.nan),
    )
)
def test_is_still_unique_with_nan_fill(array):
    if hasattr(xp, "unique"):
        assert count_unique(array) == array.size


@given(
    xps.arrays(
        dtype=xp.float32,
        shape=10,
        unique=True,
        elements=st.integers(1, 9),
        fill=st.just(xp.nan),
    )
)
def test_unique_array_with_fill_can_use_all_elements(array):
    if hasattr(xp, "unique"):
        assume(count_unique(array) == array.size)


@given(xps.arrays(dtype=xp.uint8, shape=25, unique=True, fill=st.nothing()))
def test_unique_array_without_fill(array):
    # This test covers the collision-related branches for fully dense unique arrayays.
    # Choosing 25 of 256 possible elements means we're almost certain to see colisions
    # thanks to the 'birthday paradox', but finding unique elemennts is still easy.
    if hasattr(xp, "unique"):
        assume(count_unique(array) == array.size)
