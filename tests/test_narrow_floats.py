from hypothesis import given
from hypothesis import strategies as st
from pytest import mark

from hypothesis_array import get_strategies_namespace

from .xputils import xp

xps = get_strategies_namespace(xp)


@mark.parametrize("dtype", [xp.float32, xp.float64])
@mark.parametrize("low", [-2.0, -1.0, 0.0, 1.0])
@given(st.data())
def test_bad_float_exclude_min_in_array(dtype, low, data):
    strat = xps.arrays(
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
