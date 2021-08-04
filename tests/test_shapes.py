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


from functools import reduce
from itertools import zip_longest

from hypothesis import HealthCheck, assume, given, note, settings
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from pytest import mark, raises

from hypothesis_array import Shape, get_strategies_namespace

from .common.debug import find_any, minimal
from .xputils import create_array_module

xp = create_array_module()
xpst = get_strategies_namespace(xp)

ANY_SHAPE = xpst.array_shapes(min_dims=0, max_dims=32, min_side=0, max_side=32)
ANY_NONZERO_SHAPE = xpst.array_shapes(min_dims=0, max_dims=32, min_side=1, max_side=32)


@settings(deadline=None, max_examples=10)
@given(st.integers(0, 10), st.integers(0, 9), st.integers(0), st.integers(0))
def test_minimise_array_shapes(min_dims, dim_range, min_side, side_range):
    smallest = minimal(
        xpst.array_shapes(
            min_dims=min_dims,
            max_dims=min_dims + dim_range,
            min_side=min_side,
            max_side=min_side + side_range,
        )
    )
    assert len(smallest) == min_dims and all(k == min_side for k in smallest)


@mark.parametrize(
    "kwargs", [{"min_side": 100}, {"min_dims": 15}, {"min_dims": 32}]
)
def test_interesting_array_shapes_argument(kwargs):
    xpst.array_shapes(**kwargs).example()


@given(xpst.broadcastable_shapes((), min_side=0, max_side=0, min_dims=0, max_dims=0))
def test_broadcastable_empty_shape(shape):
    assert shape == ()


@given(shape=ANY_SHAPE, data=st.data())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_broadcastable_shape_bounds_are_satisfied(shape, data):
    min_dims = data.draw(st.integers(0, 32), label="min_dims")
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    min_side = data.draw(st.integers(0, 3), label="min_side")
    max_side = data.draw(st.none() | st.integers(min_side, 6), label="max_side")
    try:
        bshape = data.draw(
            xpst.broadcastable_shapes(
                shape,
                min_side=min_side,
                max_side=max_side,
                min_dims=min_dims,
                max_dims=max_dims,
            ),
            label="bshape",
        )
    except InvalidArgument:
        assume(False)
        assert False, "unreachable"

    if max_dims is None:
        max_dims = max(len(shape), min_dims) + 2

    if max_side is None:
        max_side = max(tuple(shape[::-1][:max_dims]) + (min_side,)) + 2

    assert isinstance(bshape, tuple) and all(isinstance(s, int) for s in bshape)
    assert min_dims <= len(bshape) <= max_dims
    assert all(min_side <= s <= max_side for s in bshape)


def _draw_valid_bounds(data, shape, max_dims, permit_none=True):
    if max_dims == 0 or not shape:
        return 0, None

    smallest_side = min(shape[::-1][:max_dims])
    if smallest_side > 1:
        min_strat = st.sampled_from([1, smallest_side])
    else:
        min_strat = st.just(smallest_side)
    min_side = data.draw(min_strat, label="min_side")

    largest_side = max(max(shape[::-1][:max_dims]), min_side)
    if permit_none:
        max_strat = st.one_of(st.none(), st.integers(largest_side, largest_side + 2))
    else:
        max_strat = st.integers(largest_side, largest_side + 2)
    max_side = data.draw(max_strat, label="max_side")

    return min_side, max_side


def _broadcast_two_shapes(shape1: Shape, shape2: Shape) -> Shape:
    result = []
    for a, b in zip_longest(shape1[::-1], shape2[::-1], fillvalue=1):
        if a != b and (a != 1) and (b != 1):
            raise ValueError(
                f"shapes {shape1} and {shape2} are not broadcast-compatible"
            )
        result.append(a if a != 1 else b)
    return tuple(result[::-1])


def _broadcast_shapes(*shapes):
    """Returns the shape resulting from broadcasting the input shapes together.

    Raises `ValueError` if the shapes are not broadcast-compatible"""
    assert shapes, "Must pass >=1 shapes to broadcast"
    return reduce(_broadcast_two_shapes, shapes, ())


@settings(deadline=None, max_examples=500)
@given(
    shapes=st.lists(
        xpst.array_shapes(min_dims=0, min_side=0, max_dims=4, max_side=4), min_size=1
    )
)
def test_broadcastable_shape_util(shapes):
    # Ensures that _broadcast_shapes() raises when fed incompatible shapes,
    # and ensures that it produces the true broadcasted shape
    if len(shapes) == 1:
        assert _broadcast_shapes(*shapes) == shapes[0]
        return

    arrays = [xp.zeros(s, dtype=xp.uint8) for s in shapes]

    try:
        broadcast_out = xp.broadcast_arrays(*arrays)
    except ValueError:
        with raises(ValueError):
            _broadcast_shapes(*shapes)
        return
    broadcasted_shape = _broadcast_shapes(*shapes)

    assert broadcast_out[0].shape == broadcasted_shape


@given(shape=ANY_NONZERO_SHAPE, data=st.data())
@settings(deadline=None, max_examples=200)
def test_broadcastable_shape_has_good_default_values(shape, data):
    # Ensures that default parameters can always produce broadcast-compatible shapes
    broadcastable_shape = data.draw(
        xpst.broadcastable_shapes(shape), label="broadcastable_shapes"
    )
    # error if drawn shape for b is not broadcast-compatible
    _broadcast_shapes(shape, broadcastable_shape)


@given(min_dims=st.integers(0, 32), shape=ANY_SHAPE, data=st.data())
@settings(deadline=None)
def test_broadcastable_shape_can_broadcast(min_dims, shape, data):
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    min_side, max_side = _draw_valid_bounds(data, shape, max_dims)
    broadcastable_shape = data.draw(
        xpst.broadcastable_shapes(
            shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        ),
        label="broadcastable_shapes",
    )
    # error if drawn shape for b is not broadcast-compatible
    _broadcast_shapes(shape, broadcastable_shape)


@given(min_dims=st.integers(0, 32), shape=ANY_SHAPE, data=st.data())
@settings(deadline=None, max_examples=10)
def test_minimize_broadcastable_shape(min_dims, shape, data):
    # Ensure aligned dimensions of broadcastable shape minimizes to `(1,) * min_dims`
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    min_side, max_side = _draw_valid_bounds(data, shape, max_dims, permit_none=False)
    smallest = minimal(
        xpst.broadcastable_shapes(
            shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        )
    )
    note(f"(smallest): {smallest}")
    n_leading = max(len(smallest) - len(shape), 0)
    n_aligned = max(len(smallest) - n_leading, 0)
    expected = [min_side] * n_leading + [
        1 if min_side <= 1 <= max_side else i for i in shape[len(shape) - n_aligned:]
    ]
    assert tuple(expected) == smallest


@given(max_dims=st.integers(4, 6), data=st.data())
@settings(deadline=None)
def test_broadcastable_shape_adjusts_max_dim_with_explicit_bounds(max_dims, data):
    # Ensures that broadcastable_shapes() limits itself to satisfiable dimensions
    # Broadcastable values can only be drawn for dims 0-3 for these shapes
    shape = data.draw(st.sampled_from([(5, 3, 2, 1), (0, 3, 2, 1)]), label="shape")
    broadcastable_shape = data.draw(
        xpst.broadcastable_shapes(
            shape, min_side=2, max_side=3, min_dims=3, max_dims=max_dims
        ),
        label="broadcastable_shapes",
    )
    assert len(broadcastable_shape) == 3
    # error if drawn shape for b is not broadcast-compatible
    _broadcast_shapes(shape, broadcastable_shape)


@given(min_dims=st.integers(0, 32), min_side=st.integers(2, 3), data=st.data())
@settings(deadline=None, max_examples=10)
def test_broadcastable_shape_shrinking_with_singleton_out_of_bounds(
    min_dims, min_side, data
):
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    max_side = data.draw(st.none() | st.integers(min_side, 6), label="max_side")
    shape = data.draw(st.integers(1, 4).map(lambda n: n * (1,)), label="shape")
    smallest = minimal(
        xpst.broadcastable_shapes(
            shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        )
    )
    assert smallest == (min_side,) * min_dims


@given(
    shape=xpst.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5),
    max_dims=st.integers(0, 6),
    data=st.data(),
)
@settings(deadline=None)
def test_broadcastable_shape_can_generate_arbitrary_ndims(shape, max_dims, data):
    # ensures that generates shapes can possess any length in [min_dims, max_dims]
    desired_ndim = data.draw(st.integers(0, max_dims), label="desired_ndim")
    min_dims = data.draw(
        st.one_of(st.none(), st.integers(0, desired_ndim)), label="min_dims"
    )
    # check default arg behavior too
    kwargs = {"min_dims": min_dims} if min_dims is not None else {}
    find_any(
        xpst.broadcastable_shapes(shape, min_side=0, max_dims=max_dims, **kwargs),
        lambda x: len(x) == desired_ndim,
        settings(max_examples=10 ** 6),
    )
