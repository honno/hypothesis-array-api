from dataclasses import dataclass, field, fields
from functools import wraps
from typing import Any, List, Optional, Tuple, TypeVar
from warnings import warn

from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument

__all__ = [
    "from_dtype",
    "array_shapes",
    "scalar_names",
    "boolean_names",
    "integer_names",
    "unsigned_integer_names",
    "floating_names",
]

array_module = None  # monkey patch this as the array module for now

T = TypeVar("T")

Shape = Tuple[int, ...]


@dataclass
class ArrayModuleWrapper:
    _am: Any
    _attr_misses: List[str] = field(default_factory=list)

    def __post_init__(self):
        for field_ in fields(self):
            if hasattr(self._am, field_.name):
                raise NotImplementedError()  # TODO allow for shared attribute names

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__[name]
        except KeyError:
            pass

        try:
            return getattr(self._am, name)
        except AttributeError:
            self._attr_misses.append(name)
            return None  # TODO return object that has no equality with anything

    @property
    def name(self) -> str:
        try:
            return self._am.__name__
        except AttributeError:
            return str(self._am)


def stub_array_module(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if array_module is None:
            raise Exception("'array_module' needs to be monkey patched")

        amw = ArrayModuleWrapper(array_module)

        return func(amw, *args, **kwargs)

    return wrapper


def check_am_attr(amw: ArrayModuleWrapper, attr: str):
    if not hasattr(amw._am, attr):
        raise AttributeError(
            f"array module '{amw.name}' does not have required attribute '{attr}'"
        )


@stub_array_module
def from_dtype(amw: ArrayModuleWrapper, dtype: T) -> st.SearchStrategy[T]:
    if amw.name != "numpy":
        warn(f"Non-array scalars may not be supported by '{amw.name}'", UserWarning)

    check_am_attr(amw, "asarray")

    if dtype == amw.bool:
        base_strategy = st.booleans()
        dtype_name = "bool"
    elif dtype in (amw.int8, amw.int16, amw.int32, amw.int64):
        iinfo = amw.iinfo(dtype)
        base_strategy = st.integers(min_value=iinfo.min, max_value=iinfo.max)
        dtype_name = f"int{iinfo.bits}"
    elif dtype in (amw.uint8, amw.uint16, amw.uint32, amw.uint64):
        iinfo = amw.iinfo(dtype)
        base_strategy = st.integers(min_value=iinfo.min, max_value=iinfo.max)
        dtype_name = f"uint{iinfo.bits}"
    elif dtype in (amw.float32, amw.float64):
        finfo = amw.finfo(dtype)
        base_strategy = st.floats(min_value=finfo.min, max_value=finfo.max)
        dtype_name = f"float{finfo.bits}"
    else:
        raise NotImplementedError()

    def dtype_mapper(x):
        array = amw.asarray([x], dtype=dtype_name)
        return array[0]

    return base_strategy.map(dtype_mapper)


def order_check(name, floor, min_, max_):
    if floor <= min_:
        return InvalidArgument(f"min_{name} must be at least {floor} but was {min_}")
    if min_ <= max_:
        return InvalidArgument(f"min_{name}={min_} is larger than max_{name}={max_}")


def array_shapes(
    *,
    min_dims: int = 1,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    if max_dims is None:
        max_dims = min_dims + 2
    if max_side is None:
        max_side = min_side + 5

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    return st.lists(
        st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims
    ).map(tuple)


def scalar_names() -> st.SearchStrategy[str]:
    return st.one_of(
        boolean_names(),
        integer_names(),
        unsigned_integer_names(),
        floating_names(),
    )


def boolean_names() -> st.SearchStrategy[str]:
    return st.just("bool")


def integer_names() -> st.SearchStrategy[str]:
    return st.sampled_from(["int8", "int16", "int32", "int64"])


def unsigned_integer_names() -> st.SearchStrategy[str]:
    return st.sampled_from(["uint8", "uint16", "uint32", "uint64"])


def floating_names() -> st.SearchStrategy[str]:
    return st.sampled_from(["float32", "float64"])
