import math
from functools import lru_cache
from typing import Type, Union

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from hypothesis_array import (DTYPE_NAMES, INT_NAMES, UINT_NAMES,
                              get_strategies_namespace)

from .common.debug import minimal
from .xputils import xp

xps = get_strategies_namespace(xp)


@given(xps.scalar_dtypes())
def test_strategies_for_standard_dtypes_have_reusable_values(dtype):
    assert xps.from_dtype(dtype).has_reusable_values


@lru_cache()
def builtin_from_dtype_name(name: str) -> Type[Union[bool, int, float]]:
    if name == "bool":
        return bool
    elif name in INT_NAMES or name in UINT_NAMES:
        return int
    elif name in DTYPE_NAMES:
        return float
    raise ValueError()


@pytest.mark.parametrize("name", DTYPE_NAMES)
def test_produces_instances_from_dtype(name):
    builtin = builtin_from_dtype_name(name)
    dtype = getattr(xp, name)

    @given(xps.from_dtype(dtype))
    def test_is_builtin(value):
        assert isinstance(value, builtin)

    test_is_builtin()


@pytest.mark.parametrize("name", DTYPE_NAMES)
def test_produces_instances_from_name(name):
    builtin = builtin_from_dtype_name(name)

    @given(xps.from_dtype(name))
    def test_is_builtin(value):
        assert isinstance(value, builtin)

    test_is_builtin()


DTYPES = [getattr(xp, name) for name in DTYPE_NAMES]


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
    assume(ex == ex)  # the roundtrip test *should* fail!  (eg NaN)
    array[0] = ex
    assert array[0] == ex


@pytest.mark.parametrize(
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


def test_can_minimize_floats():
    smallest = minimal(xps.from_dtype(xp.float32), lambda n: n >= 1.0)
    assert smallest in (1, 50)
