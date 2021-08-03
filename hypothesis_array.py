# This file was part of and modifed from Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# ./CONTRIBUTING.rst for a full list of people who may hold copyright,
# and consult the git log of ./hypothesis-python/src/hypothesis/extra/numpy.py
# if you need to determine who owns an individual contribution.
# ('.' represents the root of the Hypothesis git repository)
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
from types import SimpleNamespace
from typing import (Any, Iterable, List, Mapping, Optional, Sequence, Tuple,
                    Type, TypeVar, Union)
from warnings import warn

from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.validation import check_type, check_valid_interval

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
    "valid_tuple_axes",
]


Boolean = TypeVar("Boolean")
SignedInteger = TypeVar("SignedInteger")
UnsignedInteger = TypeVar("UnsignedInteger")
Float = TypeVar("Float")
DataType = Union[Boolean, SignedInteger, UnsignedInteger, Float]
Array = TypeVar("Array")  # TODO make this a generic or something
Shape = Tuple[int, ...]
T = TypeVar("T")

DTYPE_NAMES = [
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
]


def partition_attributes_and_stubs(
    xp,
    attributes: Iterable[str]
) -> Tuple[List[Any], List[str]]:
    non_stubs = []
    stubs = []
    for attr in attributes:
        try:
            non_stubs.append(getattr(xp, attr))
        except AttributeError:
            stubs.append(attr)

    return non_stubs, stubs


def infer_xp_is_compliant(xp):
    try:
        array = xp.asarray(0, dtype=xp.int8)
        array.__array_namespace__()
    except AttributeError:
        warn(
            f"Could not determine whether module {xp.__name__}"
            " is an Array API library",
            HypothesisWarning,
        )


def check_xp_attributes(xp, attributes: List[str]):
    missing_attrs = []
    for attr in attributes:
        if not hasattr(xp, attr):
            missing_attrs.append(attr)

    if len(missing_attrs) > 0:
        f_attrs = ", ".join(missing_attrs)
        raise AttributeError(
            f"Array module {xp.__name__} does not have required attributes: {f_attrs}"
        )


def warn_on_missing_dtypes(xp, stubs: List[str]):
    f_stubs = ", ".join(stubs)
    warn(
        f"Array module {xp.__name__} does not have"
        f" the following dtypes in its namespace: {f_stubs}.",
        HypothesisWarning,
    )


def order_check(name, floor, min_, max_):
    if floor > min_:
        raise InvalidArgument(f"min_{name} must be at least {floor} but was {min_}")
    if min_ > max_:
        raise InvalidArgument(f"min_{name}={min_} is larger than max_{name}={max_}")


def get_strategies_namespace(xp) -> SimpleNamespace:
    infer_xp_is_compliant(xp)

    return SimpleNamespace(
        from_dtype=lambda *a, **kw: from_dtype(xp, *a, **kw),
        arrays=lambda *a, **kw: arrays(xp, *a, **kw),
        array_shapes=array_shapes,
        scalar_dtypes=lambda *a, **kw: scalar_dtypes(xp, *a, **kw),
        boolean_dtypes=lambda *a, **kw: boolean_dtypes(xp, *a, **kw),
        integer_dtypes=lambda *a, **kw: integer_dtypes(xp, *a, **kw),
        unsigned_integer_dtypes=lambda *a, **kw: unsigned_integer_dtypes(xp, *a, **kw),
        floating_dtypes=lambda *a, **kw: floating_dtypes(xp, *a, **kw),
        valid_tuple_axes=valid_tuple_axes,
    )


# Note NumPy supports non-array scalars which hypothesis.extra.numpy.from_dtype
# utilises, but this from_dtype() method returns just base strategies.

def from_dtype(
    xp,
    dtype: Union[Type[DataType], str],
    *,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    exclude_min: Optional[bool] = None,
    exclude_max: Optional[bool] = None,
) -> st.SearchStrategy[Union[bool, int, float]]:
    infer_xp_is_compliant(xp)

    if isinstance(dtype, str):
        if dtype in DTYPE_NAMES:
            try:
                dtype = getattr(xp, dtype)
                return from_dtype(xp, dtype)
            except AttributeError as e:
                raise InvalidArgument(
                    f"Array module {xp.__name__} does not have"
                    f" dtype {dtype} in its namespace"
                ) from e
        else:
            f_valid_dtypes = ", ".join(DTYPE_NAMES)
            raise InvalidArgument(
                f"{dtype} is not a valid Array API data type"
                f" (pick from: {f_valid_dtypes})"
            )

    stubs = []

    try:
        bool_dtype = xp.bool
        if dtype == bool_dtype:
            return st.booleans()
    except AttributeError:
        stubs.append("bool")

    def minmax_values_kw(info):
        kw = {}

        if min_value is None:
            kw["min_value"] = info.min
        else:
            if min_value < info.min:
                raise InvalidArgument(
                    f"dtype {dtype} requires min_value={min_value}"
                    f" to be at least {info.min}"
                )
            kw["min_value"] = min_value

        if max_value is None:
            kw["max_value"] = info.max
        else:
            if max_value > info.max:
                raise InvalidArgument(
                    f"dtype {dtype} requires max_value={max_value}"
                    f" to be at most {info.max}"
                )
            kw["max_value"] = max_value

        return kw

    int_dtypes, int_stubs = partition_attributes_and_stubs(
        xp, ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
    )
    if dtype in int_dtypes:
        check_xp_attributes(xp, ["iinfo"])
        iinfo = xp.iinfo(dtype)
        kw = minmax_values_kw(iinfo)
        return st.integers(**kw)

    float_dtypes, float_stubs = partition_attributes_and_stubs(
        xp, ["float32", "float64"]
    )
    if dtype in float_dtypes:
        check_xp_attributes(xp, ["finfo"])
        finfo = xp.finfo(dtype)

        kw = minmax_values_kw(finfo)
        if allow_nan is not None:
            kw["allow_nan"] = allow_nan
        if allow_infinity is not None:
            kw["allow_infinity"] = allow_infinity
        if exclude_min is not None:
            kw["exclude_min"] = exclude_min
        if exclude_max is not None:
            kw["exclude_max"] = exclude_max

        return st.floats(width=finfo.bits, **kw)

    stubs.extend(int_stubs)
    stubs.extend(float_stubs)
    if len(stubs) > 0:
        warn_on_missing_dtypes(xp, stubs)

    raise InvalidArgument(f"No strategy inference for {dtype}")


class ArrayStrategy(st.SearchStrategy):
    def __init__(self, xp, element_strategy, dtype, shape, fill, unique):
        self.xp = xp
        self.element_strategy = element_strategy
        self.dtype = dtype
        self.shape = shape
        self.fill = fill
        self.unique = unique
        self.array_size = math.prod(shape)

    def set_element(self, data, result, idx, strategy=None):
        strategy = strategy or self.element_strategy
        val = data.draw(strategy)

        try:
            result[idx] = val
        except TypeError as e:
            # TODO check type promotion rules first
            raise InvalidArgument(
                f"Could not add generated array element '{repr(val)}'"
                f"of dtype {type(val)} to array of dtype {result.dtype}."
            ) from e

        if val == val and self.xp.all(result[idx] != val):
            raise InvalidArgument(
                f"Generated array element '{repr(val)}' from strategy {strategy}"
                f" cannot be represented as dtype {self.dtype}."
                f" Array module {self.xp.__name__} instead"
                f" represents the element as '{result[idx]}'."
                "Consider using a more precise elements strategy,"
                " for example passing the width argument to floats()."
            )

        # TODO explicitly check overflow errors

    def do_draw(self, data):
        if 0 in self.shape:
            return self.xp.empty(self.shape, dtype=self.dtype)

        result = self.xp.empty(self.array_size, dtype=self.dtype)

        if self.fill.is_empty:
            if self.unique:
                seen = set()
                elements = cu.many(
                    data,
                    min_size=self.array_size,
                    max_size=self.array_size,
                    average_size=self.array_size,
                )
                i = 0
                while elements.more():
                    self.set_element(data, result, i)
                    if result[i] not in seen:
                        seen.add(result[i])
                        i += 1
                    else:
                        elements.reject()
            else:
                for i in range(self.array_size):
                    self.set_element(data, result, i)
        else:
            # TODO use xp.full() instead for optimisation(?)
            #      and put set_element() checks into a seperate method
            self.set_element(
                data, result, slice(None, None), strategy=self.fill
            )
            if self.unique and not self.xp.all(self.xp.isnan(result)):
                raise InvalidArgument(
                    f"Array module {self.xp.__name__} did not recognise"
                    f" fill value as NaN - instead got '{repr(result[0])}'."
                    " Cannot fill unique array with non-NaN values."
                )

            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                average_size=math.sqrt(self.array_size),
            )

            index_set = self.xp.full(self.array_size, False, dtype=self.xp.bool)
            seen = set()

            while elements.more():
                i = cu.integer_range(data, 0, self.array_size - 1)

                if index_set[i]:
                    elements.reject()
                    continue

                self.set_element(data, result, i)

                if self.unique:
                    if result[i] in seen:
                        elements.reject()
                        continue
                    else:
                        seen.add(result[i])

                index_set[i] = True

        result = self.xp.reshape(result, self.shape)

        return result


def arrays(
    xp,
    dtype: Union[
        Type[DataType], str, st.SearchStrategy[Type[DataType]], st.SearchStrategy[str]
    ],
    shape: Union[int, Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[st.SearchStrategy] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> st.SearchStrategy[Array]:
    infer_xp_is_compliant(xp)
    check_xp_attributes(xp, ["empty", "full", "all", "isnan", "bool", "reshape"])

    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(
            lambda d: arrays(xp, d, shape, elements=elements, fill=fill, unique=unique)
        )
    if isinstance(shape, st.SearchStrategy):
        return shape.flatmap(
            lambda s: arrays(xp, dtype, s, elements=elements, fill=fill, unique=unique)
        )

    if isinstance(shape, int):
        shape = (shape,)

    if elements is None:
        elements = from_dtype(xp, dtype)
    elif isinstance(elements, Mapping):
        elements = from_dtype(xp, dtype, **elements)

    if fill is None:
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements

    return ArrayStrategy(xp, elements, dtype, shape, fill, unique)


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


# We assume there are dtypes objects part of the array module namespace.
# Note there is a current discussion about whether this is expected behaviour:
# github.com/data-apis/array-api/issues/152


def check_dtypes(xp, dtypes: List[Type[DataType]], stubs: List[str]):
    if len(dtypes) == 0:
        f_stubs = ", ".join(stubs)
        raise InvalidArgument(
            f"Array module {xp.__name__} does not have"
            f" the following required dtypes in its namespace: {f_stubs}"
        )
    elif len(stubs) > 0:
        warn_on_missing_dtypes(xp, stubs)


def scalar_dtypes(xp) -> st.SearchStrategy[Type[DataType]]:
    infer_xp_is_compliant(xp)

    dtypes, stubs = partition_attributes_and_stubs(xp, DTYPE_NAMES)
    check_dtypes(xp, dtypes, stubs)

    return st.sampled_from(dtypes)


def boolean_dtypes(xp) -> st.SearchStrategy[Type[Boolean]]:
    infer_xp_is_compliant(xp)

    try:
        return st.just(xp.bool)
    except AttributeError:
        raise InvalidArgument(
            f"Array module {xp.__name__} does not have"
            f" a bool dtype in its namespace"
        ) from None


def check_valid_sizes(category: str, sizes: Sequence[int], valid_sizes: Sequence[int]):
    invalid_sizes = []
    for size in sizes:
        if size not in valid_sizes:
            invalid_sizes.append(size)

    if len(invalid_sizes) > 0:
        f_valid_sizes = ", ".join(str(s) for s in valid_sizes)
        f_invalid_sizes = ", ".join(str(s) for s in invalid_sizes)
        raise InvalidArgument(
            f"The following sizes are not valid for {category} dtypes:"
            f" {f_invalid_sizes} (valid sizes: {f_valid_sizes})"
        )


def numeric_dtype_names(base_name: str, sizes: Sequence[int]):
    for size in sizes:
        yield f"{base_name}{size}"


def integer_dtypes(
    xp, sizes: Union[int, Sequence[int]] = (8, 16, 32, 64)
) -> st.SearchStrategy[Type[SignedInteger]]:
    infer_xp_is_compliant(xp)

    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (8, 16, 32, 64))

    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("int", sizes)
    )
    check_dtypes(xp, dtypes, stubs)

    return st.sampled_from(dtypes)


def unsigned_integer_dtypes(
    xp, sizes: Union[int, Sequence[int]] = (8, 16, 32, 64)
) -> st.SearchStrategy[Type[UnsignedInteger]]:
    infer_xp_is_compliant(xp)

    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (8, 16, 32, 64))

    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("uint", sizes)
    )
    check_dtypes(xp, dtypes, stubs)

    return st.sampled_from(dtypes)


def floating_dtypes(
    xp, sizes: Union[int, Sequence[int]] = (32, 64)
) -> st.SearchStrategy[Type[Float]]:
    infer_xp_is_compliant(xp)

    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (32, 64))

    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("float", sizes)
    )
    check_dtypes(xp, dtypes, stubs)

    return st.sampled_from(dtypes)


def valid_tuple_axes(
    ndim: int,
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    if max_size is None:
        max_size = ndim

    check_type(int, ndim, "ndim")
    check_type(int, min_size, "min_size")
    check_type(int, max_size, "max_size")
    order_check("size", 0, min_size, max_size)
    check_valid_interval(max_size, ndim, "max_size", "ndim")

    axes = st.integers(0, max(0, 2 * ndim - 1)).map(
        lambda x: x if x < ndim else x - 2 * ndim
    )

    return st.lists(
        axes, min_size=min_size, max_size=max_size, unique_by=lambda x: x % ndim
    ).map(tuple)
