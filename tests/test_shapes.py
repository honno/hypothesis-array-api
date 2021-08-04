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
from pytest import mark

from hypothesis_array import get_strategies_namespace

from .common.debug import minimal
from .xputils import create_array_module

xp = create_array_module()
xpst = get_strategies_namespace(xp)


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
