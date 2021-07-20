from dataclasses import dataclass
from functools import wraps
from itertools import tee
from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from warnings import warn

from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.validation import check_type

__all__ = [
    "from_dtype",
    "arrays",
    "array_shapes",
    "scalar_dtypes",
    "boolean_dtypes",
    "integer_dtypes",
    "unsigned_integer_dtypes",
    "floating_dtypes",
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
Array = TypeVar("Array")  # TODO make this a generic or something

T = TypeVar("T")
Shape = Tuple[int, ...]


class Stub(str):
    pass


class ArrayModuleWrapper:
    def __init__(self, array_module: Any):
        try:
            array = array_module.asarray([True], dtype=bool)
            array.__array_namespace__()
            self.xp = array_module
        except AttributeError:
            self.xp = array_module
            warn(f"Could not determine whether module '{self}' is an Array API library")

        if hasattr(self.xp, "xp"):
            warn(f"Array module '{self}' has attribute 'xp' which will be inaccessible")

    def __getattr__(self, name: str) -> Any:
        if name == "xp":
            return self.xp

        try:
            return getattr(self.xp, name)
        except AttributeError:
            return Stub(name)

    def __str__(self):
        try:
            return self.xp.__name__
        except AttributeError:
            return str(self.xp)


def partition_stubs(
    iterable: Iterable[Union[T, Stub]]
) -> Tuple[List[T], List[Stub]]:
    it1, it2 = tee(iterable)
    non_stubs = [x for x in it1 if not isinstance(x, Stub)]
    stubs = [x for x in it2 if isinstance(x, Stub)]

    return non_stubs, stubs


def wrap_array_module(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if array_module is None:
            raise Exception("'array_module' needs to be monkey patched.")

        xpw = ArrayModuleWrapper(array_module)

        return func(xpw, *args, **kwargs)

    return wrapper


def check_attr(xpw: ArrayModuleWrapper, attr: str):
    if not hasattr(xpw.xp, attr):
        raise AttributeError(
            f"Array module '{xpw}' does not have required attribute '{attr}'"
        )


def order_check(name, floor, min_, max_):
    if floor > min_:
        raise InvalidArgument(f"min_{name} must be at least {floor} but was {min_}")
    if min_ >= max_:
        raise InvalidArgument(f"min_{name}={min_} is larger than max_{name}={max_}")


# Note NumPy supports non-array scalars which hypothesis.extra.numpy.from_dtype
# utilises, but this from_dtype() method returns just base strategies.

@wrap_array_module
def from_dtype(
        xpw: ArrayModuleWrapper,
        dtype: DataType,
) -> st.SearchStrategy[Union[bool, int, float]]:

    if dtype == xpw.bool:
        return st.booleans()

    elif dtype in (xpw.int8, xpw.int16, xpw.int32, xpw.int64):
        check_attr(xpw, "iinfo")
        iinfo = xpw.iinfo(dtype)
        return st.integers(min_value=iinfo.min, max_value=iinfo.max)

    elif dtype in (xpw.uint8, xpw.uint16, xpw.uint32, xpw.uint64):
        check_attr(xpw, "iinfo")
        iinfo = xpw.iinfo(dtype)
        return st.integers(min_value=iinfo.min, max_value=iinfo.max)

    elif dtype in (xpw.float32, xpw.float64):
        check_attr(xpw, "finfo")
        finfo = xpw.finfo(dtype)
        return st.floats(min_value=finfo.min, max_value=finfo.max)

    raise NotImplementedError()


@wrap_array_module
def arrays(
        xpw: ArrayModuleWrapper,
        dtype: Union[DataType, st.SearchStrategy[DataType]],
        shape: Shape,
) -> st.SearchStrategy[Array]:
    if len(shape) not in [0, 1]:
        raise NotImplementedError()

    check_attr(xpw, "asarray")

    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(lambda dtype: arrays(dtype, shape))

    element_strategy = from_dtype(dtype)

    if len(shape) == 0:
        @st.composite
        def strategy(draw) -> st.SearchStrategy[Array]:
            element = draw(element_strategy)
            array = xpw.asarray(element, dtype=dtype)

            return array

    else:
        @st.composite
        def strategy(draw) -> st.SearchStrategy[Array]:
            elements = draw(
                st.lists(element_strategy, min_size=shape[0], max_size=shape[0])
            )
            array = xpw.asarray(elements, dtype=dtype)

            return array

    return strategy()


def array_shapes(
    *,
    min_dims: int = 1,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    check_type(int, min_dims, "min_dims")
    check_type(int, min_side, "min_side")
    if max_dims is None:
        max_dims = min_dims + 2
    check_type(int, max_dims, "max_dims")
    if max_side is None:
        max_side = min_side + 5
    check_type(int, max_side, "max_side")
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
    xpw: ArrayModuleWrapper
    stubs: List[Stub]

    def __str__(self):
        f_stubs = ", ".join(f"'{stub}'" for stub in self.stubs)
        if len(self.stubs) == 1:
            return (
                f"Array module '{self.xpw}' does not have"
                f" the required dtype {f_stubs} in its namespace."
            )
        else:
            return (
                f"Array module '{self.xpw}' does not have"
                f" the following required dtypes in its namespace: {f_stubs}"
            )


def warn_on_missing_dtypes(xpw: ArrayModuleWrapper, stubs: List[Stub]):
    f_stubs = ", ".join(f"'{stub}'" for stub in stubs)
    if len(stubs) == 1:
        warn(
            f"Array module '{xpw}' does not have"
            f" the dtype {f_stubs} in its namespace."
        )
    else:
        warn(
            f"Array module '{xpw}' does not have"
            f" the following dtypes in its namespace: {f_stubs}."
        )


@wrap_array_module
def scalar_dtypes(xpw) -> st.SearchStrategy[Type[DataType]]:
    dtypes, stubs = partition_stubs(getattr(xpw, name) for name in DTYPE_NAMES["all"])
    if len(dtypes) == 0:
        raise MissingDtypesError(xpw, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xpw, stubs)

    return st.sampled_from(dtypes)


@wrap_array_module
def boolean_dtypes(xpw: ArrayModuleWrapper) -> st.SearchStrategy[Type[Boolean]]:
    dtype = xpw.bool
    if isinstance(dtype, Stub):
        raise MissingDtypesError(xpw, [dtype])

    return st.just(dtype)


@wrap_array_module
def integer_dtypes(xpw: ArrayModuleWrapper) -> st.SearchStrategy[Type[SignedInteger]]:
    dtypes, stubs = partition_stubs(getattr(xpw, name) for name in DTYPE_NAMES["ints"])
    if len(dtypes) == 0:
        raise MissingDtypesError(xpw, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xpw, stubs)

    return st.sampled_from(dtypes)


@wrap_array_module
def unsigned_integer_dtypes(
    xpw: ArrayModuleWrapper
) -> st.SearchStrategy[UnsignedInteger]:
    dtypes, stubs = partition_stubs(getattr(xpw, name) for name in DTYPE_NAMES["uints"])
    if len(dtypes) == 0:
        raise MissingDtypesError(xpw, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xpw, stubs)

    return st.sampled_from(dtypes)


@wrap_array_module
def floating_dtypes(xpw: ArrayModuleWrapper) -> st.SearchStrategy[Type[Float]]:
    dtypes, stubs = partition_stubs(getattr(xpw, name)
                                    for name in DTYPE_NAMES["floats"])
    if len(dtypes) == 0:
        raise MissingDtypesError(xpw, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xpw, stubs)

    return st.sampled_from(dtypes)
