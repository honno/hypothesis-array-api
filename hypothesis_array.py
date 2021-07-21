from dataclasses import dataclass
from types import ModuleType
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union
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

INT_NAMES = ["int8", "int16", "int32", "int64"]
UINT_NAMES = ["uint8", "uint16", "uint32", "uint64"]
FLOAT_NAMES = ["float32", "float64"]
DTYPE_NAMES = INT_NAMES + UINT_NAMES + FLOAT_NAMES
DTYPE_NAMES.append("bool")

Boolean = TypeVar("Boolean")
SignedInteger = TypeVar("SignedInteger")
UnsignedInteger = TypeVar("UnsignedInteger")
Float = TypeVar("Float")
DataType = Union[Boolean, SignedInteger, UnsignedInteger, Float]
Array = TypeVar("Array")  # TODO make this a generic or something

T = TypeVar("T")
Shape = Tuple[int, ...]


def get_xp_name(xp: ModuleType):
    try:
        return xp.__name__
    except AttributeError:
        return str(xp)


def partition_xp_attrs_and_stubs(
    xp: ModuleType,
    attributes: List[str]
) -> Tuple[List[Any], List[str]]:
    non_stubs = []
    stubs = []
    for attr in attributes:
        try:
            non_stubs.append(getattr(xp, attr))
        except AttributeError:
            stubs.append(attr)

    return non_stubs, stubs


def infer_xp_is_valid(xp: ModuleType):
    try:
        array = xp.asarray(True, dtype=xp.bool)
        array.__array_namespace__()
    except AttributeError:
        xp_name = get_xp_name(xp)
        warn(f"Could not determine whether module '{xp_name}' is an Array API library")


def check_xp_attr(xp: ModuleType, attr: str):
    if not hasattr(xp, attr):
        xp_name = get_xp_name(xp)
        raise AttributeError(
            f"Array module '{xp_name}' does not have required attribute '{attr}'"
        )


def order_check(name, floor, min_, max_):
    if floor > min_:
        raise InvalidArgument(f"min_{name} must be at least {floor} but was {min_}")
    if min_ > max_:
        raise InvalidArgument(f"min_{name}={min_} is larger than max_{name}={max_}")


# Note NumPy supports non-array scalars which hypothesis.extra.numpy.from_dtype
# utilises, but this from_dtype() method returns just base strategies.

def from_dtype(
    xp: ModuleType,
    dtype: DataType,
) -> st.SearchStrategy[Union[bool, int, float]]:
    stubs = []

    try:
        bool_dtype = xp.bool
        if dtype == bool_dtype:
            return st.booleans()
    except AttributeError:
        stubs.append("bool")

    int_dtypes, int_stubs = partition_xp_attrs_and_stubs(xp, INT_NAMES)
    if dtype in int_dtypes:
        check_xp_attr(xp, "iinfo")
        iinfo = xp.iinfo(dtype)

        return st.integers(min_value=iinfo.min, max_value=iinfo.max)

    uint_dtypes, uint_stubs = partition_xp_attrs_and_stubs(xp, UINT_NAMES)
    if dtype in uint_dtypes:
        check_xp_attr(xp, "iinfo")
        iinfo = xp.iinfo(dtype)

        return st.integers(min_value=iinfo.min, max_value=iinfo.max)

    float_dtypes, float_stubs = partition_xp_attrs_and_stubs(xp, FLOAT_NAMES)
    if dtype in float_dtypes:
        check_xp_attr(xp, "finfo")
        finfo = xp.finfo(dtype)

        return st.floats(min_value=finfo.min, max_value=finfo.max)

    stubs.extend(int_stubs)
    stubs.extend(uint_stubs)
    stubs.extend(float_stubs)
    if len(stubs) > 0:
        warn_on_missing_dtypes(xp, stubs)

    raise InvalidArgument(f"No strategy inference for {dtype}")


def arrays(
    xp: ModuleType,
    dtype: Union[DataType, st.SearchStrategy[DataType]],
    shape: Union[Shape, st.SearchStrategy[Shape]],
) -> st.SearchStrategy[Array]:
    check_xp_attr(xp, "asarray")

    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(lambda d: arrays(xp, d, shape))
    if isinstance(shape, st.SearchStrategy):
        return shape.flatmap(lambda s: arrays(xp, dtype, s))

    elements = from_dtype(xp, dtype)

    if len(shape) == 0:
        return elements.map(lambda e: xp.asarray(e, dtype=dtype))

    strategy = elements
    for dimension_size in reversed(shape):
        strategy = st.lists(strategy, min_size=dimension_size, max_size=dimension_size)

    return strategy.map(lambda array: xp.asarray(array, dtype=dtype))


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


# We assume there are dtype objects part of the array module namespace.
# Note there is a current discussion about whether this is expected behaviour:
# github.com/data-apis/array-api/issues/152


@dataclass
class MissingDtypesError(AttributeError):
    xp: ModuleType
    missing_dtype_names: List[str]

    def __str__(self):
        xp_name = get_xp_name(self.xp)
        f_stubs = ", ".join(f"'{stub}'" for stub in self.missing_dtype_names)
        return (
            f"Array module '{xp_name}' does not have"
            f" the following required dtypes in its namespace: {f_stubs}"
        )


def warn_on_missing_dtypes(xp: ModuleType, missing_dtype_names: List[str]):
    xp_name = get_xp_name(xp)
    f_stubs = ", ".join(f"'{stub}'" for stub in missing_dtype_names)
    warn(
        f"Array module '{xp_name}' does not have"
        f" the following dtypes in its namespace: {f_stubs}."
    )


def scalar_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[DataType]]:
    dtypes, stubs = partition_xp_attrs_and_stubs(xp, DTYPE_NAMES)
    if len(dtypes) == 0:
        raise MissingDtypesError(xp, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xp, stubs)

    return st.sampled_from(dtypes)


def boolean_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[Boolean]]:
    try:
        return st.just(xp.bool)
    except AttributeError:
        raise MissingDtypesError(xp, ["bool"])


def integer_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[SignedInteger]]:
    dtypes, stubs = partition_xp_attrs_and_stubs(xp, INT_NAMES)
    if len(dtypes) == 0:
        raise MissingDtypesError(xp, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xp, stubs)

    return st.sampled_from(dtypes)


def unsigned_integer_dtypes(xp: ModuleType) -> st.SearchStrategy[UnsignedInteger]:
    dtypes, stubs = partition_xp_attrs_and_stubs(xp, UINT_NAMES)
    if len(dtypes) == 0:
        raise MissingDtypesError(xp, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xp, stubs)

    return st.sampled_from(dtypes)


def floating_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[Float]]:
    dtypes, stubs = partition_xp_attrs_and_stubs(xp, FLOAT_NAMES)
    if len(dtypes) == 0:
        raise MissingDtypesError(xp, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xp, stubs)

    return st.sampled_from(dtypes)
