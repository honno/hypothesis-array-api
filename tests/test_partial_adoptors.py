import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument

from hypothesis_array import *
from hypothesis_array import (DTYPE_NAMES, FLOAT_NAMES, INT_NAMES,
                              NUMERIC_NAMES, UINT_NAMES, infer_xp_is_compliant)

from .xputils import create_array_module


def test_warning_on_noncompliant_xp():
    xp = create_array_module()
    with pytest.warns(HypothesisWarning):
        infer_xp_is_compliant(xp)


@pytest.mark.parametrize(
    "func, args, attr",
    [(from_dtype, ["int8"], "iinfo"), (arrays, ["int8", 5], "full")],
)
def test_error_on_missing_attr(func, args, attr):
    xp = create_array_module(exclude=(attr,))
    with pytest.raises(InvalidArgument, match="does not have required attributes"):
        func(xp, *args).example()


dtypeless_xp = create_array_module(exclude=tuple(DTYPE_NAMES))


@pytest.mark.parametrize(
    "func",
    [
        scalar_dtypes,
        boolean_dtypes,
        numeric_dtypes,
        integer_dtypes,
        unsigned_integer_dtypes,
        floating_dtypes,
    ]
)
def test_error_on_missing_dtypes(func):
    with pytest.raises(InvalidArgument):
        func(dtypeless_xp).example()


@pytest.mark.parametrize(
    "func, keep_any",
    [
        (scalar_dtypes, DTYPE_NAMES),
        (numeric_dtypes, NUMERIC_NAMES),
        (integer_dtypes, INT_NAMES),
        (unsigned_integer_dtypes, UINT_NAMES),
        (floating_dtypes, FLOAT_NAMES),
    ]
)
@given(st.data())
def test_warning_on_partial_dtypes(func, keep_any, data):
    exclude = data.draw(
        st.lists(
            st.sampled_from(keep_any),
            min_size=1,
            max_size=len(keep_any) - 1,
            unique=True,
        )
    )
    xp = create_array_module(exclude=tuple(exclude))
    with pytest.warns(HypothesisWarning):
        data.draw(func(xp))
