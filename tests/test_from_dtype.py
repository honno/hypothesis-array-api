# This file was part of and modifed from Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# ./CONTRIBUTING.rst for a full list of people who may hold copyright,
# and consult the git log of ./hypothesis-python/tests/numpy/test_argument_validation.py
# if you need to determine who owns an individual contribution.
# ('.' represents the root of the Hypothesis git repository)
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
from functools import lru_cache
from typing import Type, Union

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pytest import mark

from hypothesis_array import get_strategies_namespace

from .xputils import DTYPE_NAMES, create_array_module

xp = create_array_module()
xps = get_strategies_namespace(xp)


@given(xps.scalar_dtypes())
def test_strategies_for_standard_dtypes_have_reusable_values(dtype):
    assert xps.from_dtype(dtype).has_reusable_values


@lru_cache()
def builtin_from_dtype_name(name: str) -> Type[Union[bool, int, float]]:
    if name == "bool":
        return bool
    elif name in DTYPE_NAMES["ints"] or name in DTYPE_NAMES["uints"]:
        return int
    elif name in DTYPE_NAMES["floats"]:
        return float
    raise ValueError()


@mark.parametrize("name", DTYPE_NAMES["all"])
def test_produces_instances_from_dtype(name):
    builtin = builtin_from_dtype_name(name)
    dtype = getattr(xp, name)

    @given(xps.from_dtype(dtype))
    def test_is_builtin(value):
        assert isinstance(value, builtin)

    test_is_builtin()


@mark.parametrize("name", DTYPE_NAMES["all"])
def test_produces_instances_from_name(name):
    builtin = builtin_from_dtype_name(name)

    @given(xps.from_dtype(name))
    def test_is_builtin(value):
        assert isinstance(value, builtin)

    test_is_builtin()


DTYPES = [getattr(xp, name) for name in DTYPE_NAMES["all"]]


@given(xps.scalar_dtypes(), st.data())
@settings(max_examples=100)
def test_infer_strategy_from_dtype(dtype, data):
    # Given a dtype
    assert dtype in DTYPES
    # We can infer a strategy
    strat = xps.from_dtype(dtype)
    assert isinstance(strat, st.SearchStrategy)
    # And use it to fill an array of that dtype
    strat = xps.arrays(dtype, 10, elements=strat)
    data.draw(strat)


@given(st.data(), xps.scalar_dtypes())
def test_all_inferred_scalar_strategies_roundtrip(data, dtype):
    array = xp.zeros(shape=1, dtype=dtype)
    ex = data.draw(xps.from_dtype(array.dtype))
    assume(xp.all(ex == ex))  # the roundtrip test *should* fail!  (eg NaN)
    array[0] = ex
    assert array[0] == ex


@mark.parametrize(
    "dtype, kwargs, predicate",
    [
        # Floating point: bounds, exclusive bounds, and excluding nonfinites
        (xp.float32, {"min_value": 1, "max_value": 2}, lambda x: 1 <= x <= 2),
        (
            xp.float32,
            {"min_value": 1, "max_value": 2, "exclude_min": True, "exclude_max": True},
            lambda x: 1 < x < 2,
        ),
        (xp.float32, {"allow_nan": False}, lambda x: not math.isnan(x)),
        (xp.float32, {"allow_infinity": False}, lambda x: not math.isinf(x)),
        (
            xp.float32,
            {"allow_nan": False, "allow_infinity": False},
            lambda x: not math.isnan(x) and not math.isinf(x),
        ),
        # Integer bounds, limited to the representable range
        (xp.int8, {"min_value": -1, "max_value": 1}, lambda x: -1 <= x <= 1),
        (xp.uint8, {"min_value": 1, "max_value": 2}, lambda x: 1 <= x <= 2),
    ]
)
@given(data=st.data())
def test_from_dtype_with_kwargs(data, dtype, kwargs, predicate):
    strat = xps.from_dtype(dtype, **kwargs)
    value = data.draw(strat)
    assert predicate(value)
