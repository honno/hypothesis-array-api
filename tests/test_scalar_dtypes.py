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


from hypothesis import given
from pytest import mark

from hypothesis_array import get_strategies_namespace

from .common.debug import minimal
from .xputils import DTYPE_NAMES, create_array_module

xp = create_array_module()
xps = get_strategies_namespace(xp)


@given(xps.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["all"])


@given(xps.boolean_dtypes())
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == xp.bool


@given(xps.integer_dtypes())
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["ints"])


@given(xps.unsigned_integer_dtypes())
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["uints"])


@given(xps.floating_dtypes())
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["floats"])


def test_minimise_scalar_dtypes():
    assert minimal(xps.scalar_dtypes()) == xp.bool


@mark.parametrize(
    "strat_func, sizes",
    [
        (xps.integer_dtypes, 8),
        (xps.unsigned_integer_dtypes, 8),
        (xps.floating_dtypes, 32),
    ]
)
def test_can_specify_sizes_as_an_int(strat_func, sizes):
    strat_func(sizes=sizes)


@given(xps.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)
