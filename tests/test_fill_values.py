# This file was part of and modifed from Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# ./CONTRIBUTING.rst for a full list of people who may hold copyright,
# and consult the git log of ./hypothesis-python/tests/numpy/test_fill_values.py
# if you need to determine who owns an individual contribution.
# ('.' represents the root of the Hypothesis git repository)
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis import given
from hypothesis import strategies as st
from pytest import skip

from hypothesis_array import get_strategies_namespace

from .common.debug import find_any, minimal
from .xputils import create_array_module

xp = create_array_module()
xpst = get_strategies_namespace(xp)


@given(xpst.arrays(xp.bool, (), fill=st.nothing()))
def test_can_generate_0d_arrays_with_no_fill(array):
    assert array.dtype == xp.bool
    assert array.ndim == 0
    assert array.shape == ()


@st.composite
def distinct_integers(draw):
    used = draw(st.shared(st.builds(set), key="distinct_integers.used"))
    i = draw(st.integers(0, 2 ** 64 - 1).filter(lambda x: x not in used))
    used.add(i)
    return i


@given(xpst.arrays(xp.uint64, 10, elements=distinct_integers()))
def test_does_not_reuse_distinct_integers(array):
    # xp.unique() is optional for Array API libraries
    if hasattr(xp, "unique"):
        unique_values = xp.unique(array)
        assert unique_values.size == array.size


def test_may_reuse_distinct_integers_if_asked():
    if hasattr(xp, "unique"):
        def nunique(array) -> int:
            unique_values = xp.unique(array)
            return unique_values.size
        find_any(
            xpst.arrays(
                xp.uint64, 10, elements=distinct_integers(), fill=distinct_integers()
            ),
            lambda x: nunique(x) < len(x),
        )
    else:
        skip()


def test_minimizes_to_fill():
    smallest = minimal(xpst.arrays(xp.float32, 10, fill=st.just(3.0)))
    assert xp.all(smallest == 3.0)


@given(
    xpst.arrays(
        dtype=xp.float32,
        elements=st.floats(width=32).filter(bool),
        shape=(3, 3, 3),
        fill=st.just(1.0),
    )
)
def test_fills_everything(array):
    assert xp.all(array)
