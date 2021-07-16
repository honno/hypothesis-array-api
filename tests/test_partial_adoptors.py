from typing import Dict, TypeVar

from hypothesis import given
from hypothesis import strategies as st
from pytest import raises

import hypothesis_array as amst

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
    amst.array_module = create_array_module(name_dtype_pairs)

    for dtype_name, dtype in name_dtype_pairs:
        amst.from_dtype(dtype)


def test_error_on_missing_attr():
    class ArrayModule:
        __name__ = "foo"
        int8 = None
    am = ArrayModule()
    amst.array_module = am
    with raises(
            AttributeError,
            match="'foo' does not have required attribute 'iinfo'"
    ):
        amst.from_dtype(am.int8)


@given(dtype_maps())
def test_scalar_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    amst.array_module = create_array_module(name_dtype_pairs)

    if len(dtype_map) == 0:
        with raises(amst.MissingDtypesError):
            amst.scalar_dtypes()

    else:
        @given(amst.scalar_dtypes())
        def test(dtype):
            pass

        test()


@given(dtype_maps())
def test_boolean_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    amst.array_module = create_array_module(name_dtype_pairs)

    if "bool" in dtype_map.keys():
        @given(amst.boolean_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(amst.MissingDtypesError):
            amst.boolean_dtypes()


@given(dtype_maps())
def test_integer_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    amst.array_module = create_array_module(name_dtype_pairs)

    if any(name in dtype_map.keys() for name in amst.DTYPE_NAMES["ints"]):
        @given(amst.integer_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(amst.MissingDtypesError):
            amst.integer_dtypes()


@given(dtype_maps())
def test_unsigned_integer_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    amst.array_module = create_array_module(name_dtype_pairs)

    if any(name in dtype_map.keys() for name in amst.DTYPE_NAMES["uints"]):
        @given(amst.unsigned_integer_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(amst.MissingDtypesError):
            amst.unsigned_integer_dtypes()


@given(dtype_maps())
def test_floating_dtypes(dtype_map):
    name_dtype_pairs = tuple(dtype_map.items())
    amst.array_module = create_array_module(name_dtype_pairs)

    if any(name in dtype_map.keys() for name in amst.DTYPE_NAMES["floats"]):
        @given(amst.floating_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(amst.MissingDtypesError):
            amst.floating_dtypes()
