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

from hypothesis import given, settings
from hypothesis import strategies as st

from hypothesis_array import get_strategies_namespace

from .common.debug import minimal
from .xputils import create_array_module

xp = create_array_module()
xps = get_strategies_namespace(xp)


@given(xps.valid_tuple_axes(3))
def test_can_generate_valid_tuple_axes(axis):
    assert isinstance(axis, tuple)
    assert all(isinstance(i, int) for i in axis)


@given(ndim=st.integers(0, 5), data=st.data())
def test_mapped_positive_axes_are_unique(ndim, data):
    min_size = data.draw(st.integers(0, ndim), label="min_size")
    max_size = data.draw(st.integers(min_size, ndim), label="max_size")
    axes = data.draw(
        xps.valid_tuple_axes(ndim, min_size=min_size, max_size=max_size), label="axes"
    )
    assert len(set(axes)) == len({i if 0 < i else ndim + i for i in axes})


@given(ndim=st.integers(0, 5), data=st.data())
def test_length_bounds_are_satisfied(ndim, data):
    min_size = data.draw(st.integers(0, ndim), label="min_size")
    max_size = data.draw(st.integers(min_size, ndim), label="max_size")
    axes = data.draw(
        xps.valid_tuple_axes(ndim, min_size=min_size, max_size=max_size), label="axes"
    )
    assert min_size <= len(axes) <= max_size


@given(shape=xps.array_shapes(), data=st.data())
def test_axes_are_valid_inputs_to_sum(shape, data):
    array = xp.zeros(shape, dtype="uint8")
    axes = data.draw(xps.valid_tuple_axes(ndim=len(shape)), label="axes")
    xp.sum(array, axes)


@settings(deadline=None, max_examples=10)
@given(ndim=st.integers(0, 3), data=st.data())
def test_minimize_tuple_axes(ndim, data):
    min_size = data.draw(st.integers(0, ndim), label="min_size")
    max_size = data.draw(st.integers(min_size, ndim), label="max_size")
    smallest = minimal(xps.valid_tuple_axes(
        ndim, min_size=min_size, max_size=max_size))
    assert len(smallest) == min_size and all(k > -1 for k in smallest)


@settings(deadline=None, max_examples=10)
@given(ndim=st.integers(0, 3), data=st.data())
def test_minimize_negative_tuple_axes(ndim, data):
    min_size = data.draw(st.integers(0, ndim), label="min_size")
    max_size = data.draw(st.integers(min_size, ndim), label="max_size")
    smallest = minimal(
        xps.valid_tuple_axes(ndim, min_size=min_size, max_size=max_size),
        lambda x: all(i < 0 for i in x),
    )
    assert len(smallest) == min_size
