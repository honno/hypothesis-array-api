# This file was part of and modifed from Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# ./CONTRIBUTING.rst for a full list of people who may hold copyright,
# and consult the git log of ./hypothesis-python/tests/numpy/test_narrow_floats.py
# if you need to determine who owns an individual contribution.
# ('.' represents the root of the Hypothesis git repository)
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis import given
from hypothesis import strategies as st
from pytest import mark

from hypothesis_array import get_strategies_namespace

from .xputils import create_array_module

xp = create_array_module()
xpst = get_strategies_namespace(xp)


@mark.parametrize("dtype", [xp.float32, xp.float64])
@mark.parametrize("low", [-2.0, -1.0, 0.0, 1.0])
@given(st.data())
def test_bad_float_exclude_min_in_array(dtype, low, data):
    strat = xpst.arrays(
        dtype=dtype,
        shape=(),
        elements={
            "min_value": low,
            "max_value": low + 1,
            "exclude_min": True,
        },
    )
    array = data.draw(strat, label="array")
    assert array > low
