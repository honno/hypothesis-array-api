from types import SimpleNamespace
from typing import Dict, List, Tuple, TypeVar

from hypothesis import given
from hypothesis import strategies as st
from pytest import raises

import hypothesis_array as xpst

from .xputils import COMPLETE_DTYPE_MAP, create_array_module

T = TypeVar("T")


@st.composite
def dtype_maps(draw) -> st.SearchStrategy[Tuple[Dict[str, T], List[str]]]:
    booleans = st.booleans()
    dtype_map = {}
    missing_dtypes = []

    for dtype_name, dtype in COMPLETE_DTYPE_MAP.items():
        if draw(booleans):
            dtype_map[dtype_name] = dtype
        else:
            missing_dtypes.append(dtype_name)

    return dtype_map, missing_dtypes


@given(dtype_maps())
def test_from_dtype(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    for dtype_name, dtype in attr_dtype_pairs:
        xpst.from_dtype(dtype)


def test_error_on_missing_attr():
    xp = SimpleNamespace(**{"__name__": "foo", "int8": None})
    xpst.array_module = xp
    with raises(
        AttributeError,
        match="'foo' does not have required attribute 'iinfo'"
    ):
        xpst.from_dtype(xp.int8)


@given(dtype_maps())
def test_scalar_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if len(dtype_map) == 0:
        with raises(xpst.MissingDtypesError):
            xpst.scalar_dtypes()

    else:
        @given(xpst.scalar_dtypes())
        def test(dtype):
            pass

        test()


@given(dtype_maps())
def test_boolean_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if "bool" in dtype_map.keys():
        @given(xpst.boolean_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(xpst.MissingDtypesError):
            xpst.boolean_dtypes()


@given(dtype_maps())
def test_integer_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if any(name in dtype_map.keys() for name in xpst.DTYPE_NAMES["ints"]):
        @given(xpst.integer_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(xpst.MissingDtypesError):
            xpst.integer_dtypes()


@given(dtype_maps())
def test_unsigned_integer_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if any(name in dtype_map.keys() for name in xpst.DTYPE_NAMES["uints"]):
        @given(xpst.unsigned_integer_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(xpst.MissingDtypesError):
            xpst.unsigned_integer_dtypes()


@given(dtype_maps())
def test_floating_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xpst.array_module = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if any(name in dtype_map.keys() for name in xpst.DTYPE_NAMES["floats"]):
        @given(xpst.floating_dtypes())
        def test(dtype):
            pass

        test()

    else:
        with raises(xpst.MissingDtypesError):
            xpst.floating_dtypes()
