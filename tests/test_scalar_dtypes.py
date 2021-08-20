import pytest
from hypothesis import given

from hypothesis_array import DTYPE_NAMES, INT_NAMES, NUMERIC_NAMES, UINT_NAMES

from .common.debug import minimal
from .xputils import xp, xps

pytestmark = [pytest.mark.mockable_xp]


@given(xps.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES)


@given(xps.boolean_dtypes())
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == xp.bool


@given(xps.numeric_dtypes())
def test_can_generate_numeric_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in NUMERIC_NAMES)


@given(xps.integer_dtypes())
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in INT_NAMES)


@given(xps.unsigned_integer_dtypes())
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in UINT_NAMES)


@given(xps.floating_dtypes())
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES)


def test_minimise_scalar_dtypes():
    assert minimal(xps.scalar_dtypes()) == xp.bool


@pytest.mark.parametrize(
    "strat_func, sizes",
    [
        (xps.integer_dtypes, 8),
        (xps.unsigned_integer_dtypes, 8),
        (xps.floating_dtypes, 32),
    ]
)
def test_can_specify_sizes_as_an_int(strat_func, sizes):
    strat_func(sizes=sizes)
