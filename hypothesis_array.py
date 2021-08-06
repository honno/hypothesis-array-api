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

"""Note this reference page is very much WIP.
See `Hypothesis for the scientific stack
<https://hypothesis.readthedocs.io/en/latest/numpy.html#numpy>`_
for a general idea on what we're doing here.
"""

import math
from collections import defaultdict
from functools import update_wrapper
from types import SimpleNamespace
from typing import (Any, Iterable, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)
from warnings import warn

from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.strategies import check_strategy

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
    "broadcastable_shapes",
    "mutually_broadcastable_shapes",
]


Boolean = TypeVar("Boolean")
SignedInteger = TypeVar("SignedInteger")
UnsignedInteger = TypeVar("UnsignedInteger")
Float = TypeVar("Float")
DataType = Union[Boolean, SignedInteger, UnsignedInteger, Float]
Array = TypeVar("Array")
Shape = Tuple[int, ...]


class BroadcastableShapes(NamedTuple):
    input_shapes: Tuple[Shape, ...]
    result_shape: Shape


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
    """Creates a strategy which can generate any castable value of the given
    Array API dtype."""
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
    def __init__(self, xp, elements_strategy, dtype, shape, fill, unique):
        self.xp = xp
        self.elements_strategy = elements_strategy
        self.dtype = dtype
        self.shape = shape
        self.fill = fill
        self.unique = unique
        self.array_size = math.prod(shape)

    def check_results_set(self, result, idx, val):
        if val == val and self.xp.all(result[idx] != val):
            raise InvalidArgument(
                f"Generated array element {repr(val)} from"
                f" strategy {self.elements_strategy}"
                f" cannot be represented as dtype {self.dtype}."
                f" Array module {self.xp.__name__} instead"
                f" represents the element as {result[idx]}.\n"
                "Consider using a more precise elements strategy,"
                " for example passing the width argument to floats()."
            )

    def set_value(self, result, i, val):
        try:
            result[i] = val
        except TypeError as e:
            # TODO check type promotion rules first
            raise InvalidArgument(
                f"Could not add generated array element {repr(val)}"
                f"of dtype {type(val)} to array of dtype {result.dtype}."
            ) from e

        self.check_results_set(result, i, val)

    def do_draw(self, data):
        if 0 in self.shape:
            return self.xp.empty(self.shape, dtype=self.dtype)

        if self.fill.is_empty:
            result = self.xp.empty(self.array_size, dtype=self.dtype)
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
                    val = data.draw(self.elements_strategy)
                    self.set_value(result, i, val)
                    if result[i] not in seen:
                        seen.add(val)
                        i += 1
                    else:
                        elements.reject()
            else:
                for i in range(self.array_size):
                    val = data.draw(self.elements_strategy)
                    self.set_value(result, i, val)
        else:
            fill_val = data.draw(self.fill)
            try:
                result = self.xp.full(self.array_size, fill_val, dtype=self.dtype)
            except ValueError as e:
                raise InvalidArgument(
                    f"Could not create full array of dtype {self.dtype}"
                    f" with fill value {repr(fill_val)}"
                ) from e
            self.check_results_set(result, slice(None, None), fill_val)
            if self.unique and not self.xp.all(self.xp.isnan(result)):
                raise InvalidArgument(
                    f"Array module {self.xp.__name__} did not recognise"
                    f" fill value as NaN - instead got {repr(result[0])}."
                    " Cannot fill unique array with non-NaN values."
                )

            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                average_size=math.sqrt(self.array_size),
            )

            index_set = defaultdict(bool)
            seen = set()

            while elements.more():
                i = cu.integer_range(data, 0, self.array_size - 1)
                if index_set[i]:
                    elements.reject()
                    continue
                val = data.draw(self.elements_strategy)
                if self.unique:
                    if val in seen:
                        elements.reject()
                        continue
                    else:
                        seen.add(val)
                self.set_value(result, i, val)
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
    """Returns a strategy for generating Array API arrays."""

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

    if not all(isinstance(s, int) for s in shape):
        raise InvalidArgument(
            f"Array shape must be integer in each dimension, provided shape was {shape}"
        )

    if elements is None:
        elements = from_dtype(xp, dtype)
    elif isinstance(elements, Mapping):
        elements = from_dtype(xp, dtype, **elements)
    check_strategy(elements, "elements")

    if fill is None:
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    check_strategy(fill, "fill")

    return ArrayStrategy(xp, elements, dtype, shape, fill, unique)


def array_shapes(
    *,
    min_dims: int = 1,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    """Return a strategy for array shapes (tuples of int >= 1)."""
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
    """Return a strategy for all Array API dtypes."""
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
    xp, *, sizes: Union[int, Sequence[int]] = (8, 16, 32, 64)
) -> st.SearchStrategy[Type[SignedInteger]]:
    """Return a strategy for signed integer dtypes in the Array API."""
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
    xp, *, sizes: Union[int, Sequence[int]] = (8, 16, 32, 64)
) -> st.SearchStrategy[Type[UnsignedInteger]]:
    """Return a strategy for unsigned integer dtypes in the Array API."""
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
    xp, *, sizes: Union[int, Sequence[int]] = (32, 64)
) -> st.SearchStrategy[Type[Float]]:
    """Return a strategy for floating dtypes in the Array API."""
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
    """Return a strategy for something I don't quite understand yet."""
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


class MutuallyBroadcastableShapesStrategy(st.SearchStrategy):
    def __init__(
        self,
        num_shapes,
        base_shape=(),
        min_dims=0,
        max_dims=None,
        min_side=1,
        max_side=None,
    ):
        assert 0 <= min_side <= max_side
        assert 0 <= min_dims <= max_dims

        self.base_shape = base_shape
        self.num_shapes = num_shapes
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.min_side = min_side
        self.max_side = max_side

        self.side_strat = st.integers(min_side, max_side)
        self.size_one_allowed = self.min_side <= 1 <= self.max_side

    def do_draw(self, data):
        # All shapes are handled in column-major order; i.e. they are reversed
        base_shape = self.base_shape[::-1]
        result_shape = list(base_shape)
        shapes = [[] for _ in range(self.num_shapes)]
        use = [True for _ in range(self.num_shapes)]

        for dim_count in range(1, self.max_dims + 1):
            dim = dim_count - 1

            # We begin by drawing a valid dimension-size for the given
            # dimension. This restricts the variability across the shapes
            # at this dimension such that they can only choose between
            # this size and a singleton dimension.
            if len(base_shape) < dim_count or base_shape[dim] == 1:
                # dim is unrestricted by the base-shape: shrink to min_side
                dim_side = data.draw(self.side_strat)
            elif base_shape[dim] <= self.max_side:
                # dim is aligned with non-singleton base-dim
                dim_side = base_shape[dim]
            else:
                # only a singleton is valid in alignment with the base-dim
                dim_side = 1

            for shape_id, shape in enumerate(shapes):
                # Populating this dimension-size for each shape, either
                # the drawn size is used or, if permitted, a singleton
                # dimension.
                if dim_count <= len(base_shape) and self.size_one_allowed:
                    # aligned: shrink towards size 1
                    side = data.draw(st.sampled_from([1, dim_side]))
                else:
                    side = dim_side

                # Use a trick where where a biased coin is queried to see
                # if the given shape-tuple will continue to be grown. All
                # of the relevant draws will still be made for the given
                # shape-tuple even if it is no longer being added to.
                # This helps to ensure more stable shrinking behavior.
                if self.min_dims < dim_count:
                    use[shape_id] &= cu.biased_coin(
                        data, 1 - 1 / (1 + self.max_dims - dim)
                    )

                if use[shape_id]:
                    shape.append(side)
                    if len(result_shape) < len(shape):
                        result_shape.append(shape[-1])
                    elif shape[-1] != 1 and result_shape[dim] == 1:
                        result_shape[dim] = shape[-1]
            if not any(use):
                break

        result_shape = result_shape[: max(map(len, [self.base_shape] + shapes))]

        assert len(shapes) == self.num_shapes
        assert all(self.min_dims <= len(s) <= self.max_dims for s in shapes)
        assert all(self.min_side <= s <= self.max_side for side in shapes for s in side)

        return BroadcastableShapes(
            input_shapes=tuple(tuple(reversed(shape)) for shape in shapes),
            result_shape=tuple(reversed(result_shape)),
        )


def broadcastable_shapes(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    """Return a strategy for generating shapes that are broadcast-compatible
    with the provided shape."""
    check_type(tuple, shape, "shape")
    strict_check = max_side is None or max_dims is None
    check_type(int, min_side, "min_side")
    check_type(int, min_dims, "min_dims")

    if max_dims is None:
        max_dims = min(32, max(len(shape), min_dims) + 2)
    else:
        check_type(int, max_dims, "max_dims")

    if max_side is None:
        max_side = max(tuple(shape[-max_dims:]) + (min_side,)) + 2
    else:
        check_type(int, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    if 32 < max_dims:
        raise InvalidArgument("max_dims cannot exceed 32")

    dims, bound_name = (max_dims, "max_dims") if strict_check else (
        min_dims, "min_dims")

    # check for unsatisfiable min_side
    if not all(min_side <= s for s in shape[::-1][:dims] if s != 1):
        raise InvalidArgument(
            "Given shape=%r, there are no broadcast-compatible "
            "shapes that satisfy: %s=%s and min_side=%s"
            % (shape, bound_name, dims, min_side)
        )

    # check for unsatisfiable [min_side, max_side]
    if not (
        min_side <= 1 <= max_side or all(s <= max_side for s in shape[::-1][:dims])
    ):
        raise InvalidArgument(
            "Given shape=%r, there are no broadcast-compatible shapes "
            "that satisfy: %s=%s and [min_side=%s, max_side=%s]"
            % (shape, bound_name, dims, min_side, max_side)
        )

    if not strict_check:
        # reduce max_dims to exclude unsatisfiable dimensions
        for n, s in zip(range(max_dims), reversed(shape)):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break

    return MutuallyBroadcastableShapesStrategy(
        num_shapes=1,
        base_shape=shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    ).map(lambda x: x.input_shapes[0])


def mutually_broadcastable_shapes(
    num_shapes: int,
    *,
    base_shape: Shape = (),
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[BroadcastableShapes]:
    """Return a strategy for generating a specified number of shapes that are
    mutually-broadcastable with one another and with the provided base shape."""

    check_type(int, num_shapes, "num_shapes")
    if num_shapes < 1:
        raise InvalidArgument(f"num_shapes={num_shapes} must be at least 1")

    check_type(tuple, base_shape, "base_shape")
    check_type(int, min_side, "min_side")
    check_type(int, min_dims, "min_dims")

    strict_check = max_dims is not None

    if max_dims is None:
        max_dims = min(32, max(len(base_shape), min_dims) + 2)
    check_type(int, max_dims, "max_dims")

    if max_side is None:
        max_side = max(base_shape[-max_dims:] + (min_side,)) + 2
    check_type(int, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    if strict_check:
        dims = max_dims
        bound_name = "max_dims"
    else:
        dims = min_dims
        bound_name = "min_dims"

    # check for unsatisfiable min_side
    if not all(min_side <= s for s in base_shape[::-1][:dims] if s != 1):
        raise InvalidArgument(
            f"Given base_shape={base_shape}, there are no broadcast-compatible"
            f" shapes that satisfy: {bound_name}={dims} and min_side={min_side}"
        )

    # check for unsatisfiable [min_side, max_side]
    if not (
        min_side <= 1 <= max_side or all(s <= max_side for s in base_shape[::-1][:dims])
    ):
        raise InvalidArgument(
            f"Given base_shape={base_shape}, there are no broadcast-compatible"
            f" shapes that satisfy all of {bound_name}={dims},"
            f" min_side={min_side}, and max_side={max_side}"
        )

    if not strict_check:
        # reduce max_dims to exclude unsatisfiable dimensions
        for n, s in zip(range(max_dims), base_shape[::-1]):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break

    return MutuallyBroadcastableShapesStrategy(
        num_shapes=num_shapes,
        base_shape=base_shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    )


def get_strategies_namespace(xp) -> SimpleNamespace:
    """Creates a strategies namespace."""
    infer_xp_is_compliant(xp)

    return SimpleNamespace(
        from_dtype=update_wrapper(
            lambda *a, **kw: from_dtype(xp, *a, **kw), from_dtype
        ),
        arrays=update_wrapper(
            lambda *a, **kw: arrays(xp, *a, **kw), arrays
        ),
        array_shapes=array_shapes,
        scalar_dtypes=update_wrapper(
            lambda *a, **kw: scalar_dtypes(xp, *a, **kw), scalar_dtypes
        ),
        boolean_dtypes=update_wrapper(
            lambda *a, **kw: boolean_dtypes(xp, *a, **kw), boolean_dtypes
        ),
        integer_dtypes=update_wrapper(
            lambda *a, **kw: integer_dtypes(xp, *a, **kw), integer_dtypes
        ),
        unsigned_integer_dtypes=update_wrapper(
            lambda *a, **kw: unsigned_integer_dtypes(xp, *a, **kw),
            unsigned_integer_dtypes,
        ),
        floating_dtypes=update_wrapper(
            lambda *a, **kw: floating_dtypes(xp, *a, **kw), floating_dtypes
        ),
        valid_tuple_axes=valid_tuple_axes,
        broadcastable_shapes=broadcastable_shapes,
        mutually_broadcastable_shapes=mutually_broadcastable_shapes,
    )
