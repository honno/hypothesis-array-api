from types import SimpleNamespace
from typing import Dict, List, Tuple, TypeVar

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from pytest import raises

import hypothesis_array as xpst

from .xputils import DTYPE_NAMES, DTYPES_MAP, create_array_module

T = TypeVar("T")


@st.composite
def dtype_maps(draw) -> st.SearchStrategy[Tuple[Dict[str, T], List[str]]]:
    booleans = st.booleans()
    dtype_map = {}
    missing_dtypes = []

    for dtype_name, dtype in DTYPES_MAP.items():
        if draw(booleans):
            dtype_map[dtype_name] = dtype
        else:
            missing_dtypes.append(dtype_name)

    return dtype_map, missing_dtypes


@given(dtype_maps())
def test_from_dtype(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xp = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    for dtype_name, dtype in attr_dtype_pairs:
        xpst.from_dtype(xp, dtype)


def test_error_on_missing_attr():
    xp = SimpleNamespace(__name__="foo", int8=None)
    with raises(
        AttributeError,
        match="'foo' does not have required attribute 'iinfo'"
    ):
        xpst.from_dtype(xp, xp.int8)


@given(dtype_maps())
def test_scalar_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xp = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if len(dtype_map) == 0:
        with raises(InvalidArgument):
            xpst.scalar_dtypes(xp)

    else:
        @given(xpst.scalar_dtypes(xp))
        def test(dtype):
            pass

        test()


@given(dtype_maps())
def test_boolean_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xp = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if "bool" in dtype_map.keys():
        @given(xpst.boolean_dtypes(xp))
        def test(dtype):
            pass

        test()

    else:
        with raises(InvalidArgument):
            xpst.boolean_dtypes(xp)


@given(dtype_maps())
def test_integer_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xp = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if any(name in dtype_map.keys() for name in DTYPE_NAMES["ints"]):
        @given(xpst.integer_dtypes(xp))
        def test(dtype):
            pass

        test()

    else:
        with raises(InvalidArgument):
            xpst.integer_dtypes(xp)


@given(dtype_maps())
def test_unsigned_integer_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xp = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if any(name in dtype_map.keys() for name in DTYPE_NAMES["uints"]):
        @given(xpst.unsigned_integer_dtypes(xp))
        def test(dtype):
            pass

        test()

    else:
        with raises(InvalidArgument):
            xpst.unsigned_integer_dtypes(xp)


@given(dtype_maps())
def test_floating_dtypes(dtype_map_and_missing_dtypes):
    dtype_map, missing_dtypes = dtype_map_and_missing_dtypes
    attr_dtype_pairs = tuple(dtype_map.items())
    xp = create_array_module(
        assign_attrs=attr_dtype_pairs,
        attrs_to_del=tuple(missing_dtypes),
    )

    if any(name in dtype_map.keys() for name in DTYPE_NAMES["floats"]):
        @given(xpst.floating_dtypes(xp))
        def test(dtype):
            pass

        test()

    else:
        with raises(InvalidArgument):
            xpst.floating_dtypes(xp)
