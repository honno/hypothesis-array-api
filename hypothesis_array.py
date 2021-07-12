from typing import TypeVar, Callable

from hypothesis import strategies as st

aa = None  # monkey patch this as the array module for now

T = TypeVar("T")

def from_dtype(dtype: T) -> st.SearchStrategy[T]:
    if dtype in (aa.int8, aa.int16, aa.int32, aa.int64):
        iinfo = aa.iinfo(dtype)
        base_strategy = st.integers(min_value=iinfo.min, max_value=iinfo.max)
        dtype_name = f"int{iinfo.bits}"
    elif dtype in (aa.uint8, aa.uint16, aa.uint32, aa.uint64):
        iinfo = aa.iinfo(dtype)
        base_strategy = st.integers(min_value=iinfo.min, max_value=iinfo.max)
        dtype_name = f"uint{iinfo.bits}"
    elif dtype in (aa.float32, aa.float64):
        finfo = aa.finfo(dtype)
        base_strategy = st.floats(min_value=finfo.min, max_value=finfo.max)
        dtype_name = f"float{finfo.bits}"
    elif dtype == aa.bool:
        raise NotImplementedError("'asarray(x, dtype=\"bool\")' outputs Python's bool")
    else:
        raise NotImplementedError()

    def dtype_mapper(x):
        array = aa.asarray([x], dtype=dtype_name)
        return array[0]

    return base_strategy.map(dtype_mapper)
