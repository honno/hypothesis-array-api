from typing import TypeVar

from hypothesis import strategies as st

aa = None  # monkey patch this as the array module for now

T = TypeVar("T")

def from_dtype(dtype: T, **kwargs) -> st.SearchStrategy[T]:
    if dtype == aa.bool:
        base_strategy = st.booleans(**kwargs)
    elif dtype in (aa.int8, aa.int16, aa.int32, aa.int64):
        iinfo = aa.iinfo(dtype)
        base_strategy = st.integers(min_value=iinfo.min, max_value=iinfo.max, **kwargs)
    elif dtype in (aa.float32, aa.float64):
        finfo = aa.finfo(dtype)
        base_strategy = st.floats(min_value=finfo.min, max_value=finfo.max, **kwargs)
    else:
        raise NotImplementedError()

    return base_strategy.map(dtype)
