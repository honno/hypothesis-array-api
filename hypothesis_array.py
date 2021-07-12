from functools import wraps
from typing import TypeVar, Callable, List, Any
from dataclasses import dataclass, field

from hypothesis import strategies as st

array_module = None  # monkey patch this as the array module for now

T = TypeVar("T")

@dataclass
class ArrayModuleWrapper:
    am: Any
    attr_misses: List[str] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.am, name)
        except AttributeError:
            self.attr_misses.append(name)
            return None

def stub_array_module(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if array_module is None:
            raise Exception("'array_module' needs to be monkey patched")

        amw = ArrayModuleWrapper(array_module)

        return func(amw, *args, **kwargs)

    return wrapper

def check_am_attr(amw: ArrayModuleWrapper, attr: str):
    if not hasattr(amw.am, attr):
        raise AttributeError(
            f"array module '{awm.am}' does not have required attribute '{attr}'"
        )

@stub_array_module
def from_dtype(amw: ArrayModuleWrapper, dtype: T) -> st.SearchStrategy[T]:
    check_am_attr(amw, "asarray")

    if dtype == amw.bool:
        base_strategy = st.booleans()
        dtype_namwe = "bool"
    elif dtype in (amw.int8, amw.int16, amw.int32, amw.int64):
        iinfo = amw.iinfo(dtype)
        base_strategy = st.integers(min_value=iinfo.min, max_value=iinfo.max)
        dtype_namwe = f"int{iinfo.bits}"
    elif dtype in (amw.uint8, amw.uint16, amw.uint32, amw.uint64):
        iinfo = amw.iinfo(dtype)
        base_strategy = st.integers(min_value=iinfo.min, max_value=iinfo.max)
        dtype_namwe = f"uint{iinfo.bits}"
    elif dtype in (amw.float32, amw.float64):
        finfo = amw.finfo(dtype)
        base_strategy = st.floats(min_value=finfo.min, max_value=finfo.max)
        dtype_namwe = f"float{finfo.bits}"
    else:
        raise NotImplementedError()

    def dtype_mapper(x):
        array = amw.asarray([x], dtype=dtype_namwe)
        return array[0]

    return base_strategy.map(dtype_mapper)
