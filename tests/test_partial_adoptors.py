from typing import Dict, TypeVar

from hypothesis import given
from hypothesis import strategies as st
from pytest import raises

import hypothesis_array as xpst

from .array_module_utils import complete_dtype_map, create_array_module

T = TypeVar("T")


@st.composite
def dtype_maps(draw) -> st.SearchStrategy[Dict[str, T]]:
    booleans = st.booleans()
    dtype_map = {}

    for dtype_name, dtype in complete_dtype_map.items():
        if draw(booleans):
            dtype_map[dtype_name] = dtype

    return dtype_map


@given(dtype_maps())
def test_from_dtype(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(name_dtype_pairs)

    for dtype_name, dtype in name_dtype_pairs:
        xpst.from_dtype(dtype)


def test_error_on_missing_attr():
    class ArrayModule:
        __name__ = "foo"
        int8 = None
    am = ArrayModule()
    xpst.array_module = am
    with raises(
            AttributeError,
            match="'foo' does not have required attribute 'iinfo'"
    ):
        xpst.from_dtype(am.int8)


@given(dtype_maps())
def test_scalar_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(name_dtype_pairs)

    if len(dtype_map) == 0:
        with raises(xpst.MissingDtypesError):
            xpst.scalar_dtypes()

    else:
        @given(xpst.scalar_dtypes())
        def test(dtype):
            pass

        test()


@given(dtype_maps())
def test_boolean_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(name_dtype_pairs)

    if "bool" in dtype_map.keys():
        @given(xpst.boolean_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(xpst.MissingDtypesError):
            xpst.boolean_dtypes()


@given(dtype_maps())
def test_integer_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(name_dtype_pairs)

    if any(name in dtype_map.keys() for name in xpst.DTYPE_NAMES["ints"]):
        @given(xpst.integer_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(xpst.MissingDtypesError):
            xpst.integer_dtypes()


@given(dtype_maps())
def test_unsigned_integer_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(name_dtype_pairs)

    if any(name in dtype_map.keys() for name in xpst.DTYPE_NAMES["uints"]):
        @given(xpst.unsigned_integer_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(xpst.MissingDtypesError):
            xpst.unsigned_integer_dtypes()


@given(dtype_maps())
def test_floating_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(name_dtype_pairs)

    if any(name in dtype_map.keys() for name in xpst.DTYPE_NAMES["floats"]):
        @given(xpst.floating_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(xpst.MissingDtypesError):
            xpst.floating_dtypes()
