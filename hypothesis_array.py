from dataclasses import dataclass
from functools import wraps
from itertools import tee
from typing import Any, Iterable, List, Optional, Tuple, TypeVar, Union
from warnings import warn

from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument

__all__ = [
    "array_shapes",
    "scalar_dtypes",
    "boolean_dtypes",
    "integer_dtypes",
    "unsigned_integer_dtypes",
    "floating_dtypes",
    "from_dtype",
]

array_module = None  # monkey patch this as the array module for now

DTYPE_NAMES = {
    "all": [
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float32", "float64",
    ],
    "ints": ["int8", "int16", "int32", "int64"],
    "uints": ["uint8", "uint16", "uint32", "uint64"],
    "floats": ["float32", "float64"],
}

Boolean = TypeVar("Boolean")
SignedInteger = TypeVar("SignedInteger")
UnsignedInteger = TypeVar("UnsignedInteger")
Float = TypeVar("Float")
DataType = Union[Boolean, SignedInteger, UnsignedInteger, Float]

Shape = Tuple[int, ...]


class Stub(str):
    pass


class ArrayModuleWrapper:
    def __init__(self, array_module: Any):
        self.am = array_module
        if hasattr(self.am, "am"):
            warn(f"array module '{self}' has attribute 'am' which will be inaccessible")

    def __getattr__(self, name: str) -> Any:
        if name == "am":
            return self.am

        try:
            return getattr(self.am, name)
        except AttributeError:
            return Stub(name)

    def __str__(self):
        try:
            return self.am.__name__
        except AttributeError:
            return str(self.am)


def partition_stubs(
    iterable: Iterable[Union[Any, Stub]]
) -> Tuple[List[Any], List[Stub]]:
    it1, it2 = tee(iterable)
    non_stubs = [x for x in it1 if not isinstance(x, Stub)]
    stubs = [x for x in it2 if isinstance(x, Stub)]

    return non_stubs, stubs


def wrap_array_module(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if array_module is None:
            raise Exception("'array_module' needs to be monkey patched.")

        amw = ArrayModuleWrapper(array_module)

        return func(amw, *args, **kwargs)

    return wrapper


def check_am_attr(amw: ArrayModuleWrapper, attr: str):
    if not hasattr(amw.am, attr):
        raise AttributeError(
            f"Array module '{amw}' does not have required attribute '{attr}'"
        )


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


# we assume there are dtype objects part of the array module namespace
# note there is a current discussion about whether this is correct
# github.com/data-apis/array-api/issues/152


@dataclass
class MissingDtypesError(AttributeError):
    amw: ArrayModuleWrapper
    stubs: List[Stub]

    def __str__(self):
        f_stubs = ", ".join(f"'{stub}'" for stub in self.stubs)
        if len(self.stubs) == 1:
            return (
                f"Array module '{self.amw}' does not have"
                f" the required dtype {f_stubs} in its namespace."
            )
        else:
            return (
                f"Array module '{self.amw}' does not have"
                f" the following required dtypes in its namespace: {f_stubs}"
            )


def warn_on_missing_dtypes(amw: ArrayModuleWrapper, stubs: List[Stub]):
    f_stubs = ", ".join(f"'{stub}'" for stub in stubs)
    if len(stubs) == 1:
        warn(
            f"Array module '{amw}' does not have"
            f" the dtype {f_stubs} in its namespace."
        )
    else:
        warn(
            f"Array module '{amw}' does not have"
            f" the following dtypes in its namespace: {f_stubs}."
        )


@wrap_array_module
def scalar_dtypes(amw) -> st.SearchStrategy[DataType]:
    dtypes, stubs = partition_stubs(getattr(amw, name) for name in DTYPE_NAMES["all"])
    if len(dtypes) == 0:
        raise MissingDtypesError(amw, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(amw, stubs)

    return st.sampled_from(dtypes)


@wrap_array_module
def boolean_dtypes(amw: ArrayModuleWrapper) -> st.SearchStrategy[Boolean]:
    dtype = amw.bool
    if isinstance(dtype, Stub):
        raise MissingDtypesError(amw, [dtype])

    return st.just(dtype)


@wrap_array_module
def integer_dtypes(amw: ArrayModuleWrapper) -> st.SearchStrategy[SignedInteger]:
    dtypes, stubs = partition_stubs(getattr(amw, name) for name in DTYPE_NAMES["ints"])
    if len(dtypes) == 0:
        raise MissingDtypesError(amw, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(amw, stubs)

    return st.sampled_from(dtypes)


@wrap_array_module
def unsigned_integer_dtypes(
    amw: ArrayModuleWrapper
) -> st.SearchStrategy[UnsignedInteger]:
    dtypes, stubs = partition_stubs(getattr(amw, name) for name in DTYPE_NAMES["uints"])
    if len(dtypes) == 0:
        raise MissingDtypesError(amw, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(amw, stubs)

    return st.sampled_from(dtypes)


@wrap_array_module
def floating_dtypes(amw: ArrayModuleWrapper) -> st.SearchStrategy[Float]:
    dtypes, stubs = partition_stubs(getattr(amw, name)
                                    for name in DTYPE_NAMES["floats"])
    if len(dtypes) == 0:
        raise MissingDtypesError(amw, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(amw, stubs)

    return st.sampled_from(dtypes)


@wrap_array_module
def from_dtype(amw: ArrayModuleWrapper, dtype: DataType) -> st.SearchStrategy[DataType]:
    if amw.name != "numpy":
        warn(f"Non-array scalars may not be supported by '{amw}'", UserWarning)

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
