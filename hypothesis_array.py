from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union
from warnings import warn

from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.validation import check_type

__all__ = [
    "get_strategies_namespace",
    "arrays",
    "array_shapes",
    "from_dtype",
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


def check_xp_is_compliant(xp: ModuleType):
    # TODO cache module name and ignore below if already hit
    try:
        array = xp.asarray(True, dtype=xp.bool)
        array.__array_namespace__()
    except AttributeError:
        warn(
            f"Could not determine whether module '{xp.__name__}'"
            " is an Array API library",
            HypothesisWarning,
        )


def check_xp_attr(xp: ModuleType, attr: str):
    if not hasattr(xp, attr):
        raise AttributeError(
            f"Array module '{xp.__name__}' does not have required attribute '{attr}'"
        )


@dataclass
class MissingDtypesError(InvalidArgument, AttributeError):
    xp: ModuleType
    missing_dtypes: List[str]

    def __str__(self):
        f_stubs = ", ".join(f"'{stub}'" for stub in self.missing_dtypes)
        return (
            f"Array module '{self.xp.__name__}' does not have"
            f" the following required dtypes in its namespace: {f_stubs}"
        )


def warn_on_missing_dtypes(xp: ModuleType, missing_dtypes: List[str]):
    f_stubs = ", ".join(f"'{stub}'" for stub in missing_dtypes)
    warn(
        f"Array module '{xp.__name__}' does not have"
        f" the following dtypes in its namespace: {f_stubs}.",
        HypothesisWarning,
    )


def order_check(name, floor, min_, max_):
    if floor > min_:
        raise InvalidArgument(f"min_{name} must be at least {floor} but was {min_}")
    if min_ > max_:
        raise InvalidArgument(f"min_{name}={min_} is larger than max_{name}={max_}")


def get_strategies_namespace(xp: ModuleType) -> SimpleNamespace:
    check_xp_is_compliant(xp)

    return SimpleNamespace(
        from_dtype=lambda *a, **kw: from_dtype(xp, *a, *kw),
        arrays=lambda *a, **kw: arrays(xp, *a, *kw),
        array_shapes=lambda *a, **kw: array_shapes(*a, *kw),
        scalar_dtypes=lambda *a, **kw: scalar_dtypes(xp, *a, *kw),
        boolean_dtypes=lambda *a, **kw: boolean_dtypes(xp, *a, *kw),
        integer_dtypes=lambda *a, **kw: integer_dtypes(xp, *a, *kw),
        unsigned_integer_dtypes=lambda *a, **kw: unsigned_integer_dtypes(xp, *a, *kw),
        floating_dtypes=lambda *a, **kw: floating_dtypes(xp, *a, *kw),
    )


# Note NumPy supports non-array scalars which hypothesis.extra.numpy.from_dtype
# utilises, but this from_dtype() method returns just base strategies.

def from_dtype(
    xp: ModuleType,
    dtype: DataType,
) -> st.SearchStrategy[Union[bool, int, float]]:
    check_xp_is_compliant(xp)

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
    # TODO do these only once... maybe have _arrays() which is recursive instead
    check_xp_is_compliant(xp)
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


def scalar_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[DataType]]:
    check_xp_is_compliant(xp)

    dtypes, stubs = partition_xp_attrs_and_stubs(xp, DTYPE_NAMES)
    if len(dtypes) == 0:
        raise MissingDtypesError(xp, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xp, stubs)

    return st.sampled_from(dtypes)


def boolean_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[Boolean]]:
    check_xp_is_compliant(xp)

    try:
        return st.just(xp.bool)
    except AttributeError:
        raise MissingDtypesError(xp, ["bool"])


def integer_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[SignedInteger]]:
    check_xp_is_compliant(xp)

    dtypes, stubs = partition_xp_attrs_and_stubs(xp, INT_NAMES)
    if len(dtypes) == 0:
        raise MissingDtypesError(xp, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xp, stubs)

    return st.sampled_from(dtypes)


def unsigned_integer_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[UnsignedInteger]]:
    check_xp_is_compliant(xp)

    dtypes, stubs = partition_xp_attrs_and_stubs(xp, UINT_NAMES)
    if len(dtypes) == 0:
        raise MissingDtypesError(xp, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xp, stubs)

    return st.sampled_from(dtypes)


def floating_dtypes(xp: ModuleType) -> st.SearchStrategy[Type[Float]]:
    check_xp_is_compliant(xp)

    dtypes, stubs = partition_xp_attrs_and_stubs(xp, FLOAT_NAMES)
    if len(dtypes) == 0:
        raise MissingDtypesError(xp, stubs)
    elif len(stubs) != 0:
        warn_on_missing_dtypes(xp, stubs)

    return st.sampled_from(dtypes)
