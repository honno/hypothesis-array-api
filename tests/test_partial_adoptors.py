from typing import List, Dict, Tuple, TypeVar
from functools import lru_cache

from hypothesis import strategies as st
from hypothesis import given, assume
import numpy as np

import hypothesis_array as aast

T = TypeVar("T")

complete_dtype_map = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
}

@st.composite
def dtype_maps(draw) -> st.SearchStrategy[Dict[str, T]]:
    booleans = st.booleans()
    dtype_map = {}
    for dtype_name, dtype in complete_dtype_map.items():
        if draw(booleans):
            dtype_map[dtype_name] = dtype

    return dtype_map

@lru_cache()
def create_array_module(name_dtype_pairs: Tuple[Tuple[str, T], ...]):
    class ArrayModule:
        iinfo=np.iinfo
        finfo=np.finfo
        asarray=np.asarray

    aa = ArrayModule()

    for dtype_name, dtype in name_dtype_pairs:
        setattr(aa, dtype_name, dtype)

    return aa

@given(dtype_maps(), st.data())
def test_inferred_dtype_strategies(dtype_map, data):
    name_dtype_pairs = tuple(dtype_map.items())
    aast.aa = create_array_module(name_dtype_pairs)

    for dtype_name, dtype in name_dtype_pairs:
        strategy = aast.from_dtype(dtype)
        assert isinstance(data.draw(strategy), dtype)
