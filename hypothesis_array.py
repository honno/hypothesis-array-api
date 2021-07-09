from typing import Optional, Union, TypeVar

from hypothesis import strategies as st

aa = None  # monkey patch this as the array module for now

T = TypeVar("T")

def from_dtype(dtype: T, **kwargs) -> st.SearchStrategy[T]:
    if dtype in (aa.int8, aa.int16, aa.int32, aa.int64):
        base_strategy = st.integers(**kwargs)
    else:
        raise NotImplementedError()

    return base_strategy.map(dtype)
