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

import hypothesis_array as _xpst

from .xputils import create_array_module

xp = create_array_module()
xpst = _xpst.get_strategies_namespace(xp)


@given(xpst.arrays(xp.bool, (), fill=st.nothing()))
def test_can_generate_0d_arrays_with_no_fill(array):
    assert array.dtype == xp.bool
    assert array.ndim == 0
    assert array.shape == ()
