# This file was part of and modifed from Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# ./CONTRIBUTING.rst for a full list of people who may hold copyright,
# and consult the git log of ./hypothesis-python/tests/numpy/test_argument_validation.py
# if you need to determine who owns an individual contribution.
# ('.' represents the root of the Hypothesis git repository)
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.errors import InvalidArgument
from pytest import mark, param, raises

from hypothesis_array import get_strategies_namespace

from .xputils import create_array_module

xp = create_array_module()
xpst = get_strategies_namespace(xp)


def e(a, **kwargs):
    kw = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    return param(a, kwargs, id=f"{a.__name__}({kw})")


@mark.parametrize(
    ("function", "kwargs"),
    [
        e(xpst.array_shapes, min_side=2, max_side=1),
        e(xpst.array_shapes, min_dims=3, max_dims=2),
        e(xpst.array_shapes, min_dims=-1),
        e(xpst.array_shapes, min_side=-1),
        e(xpst.array_shapes, min_side="not an int"),
        e(xpst.array_shapes, max_side="not an int"),
        e(xpst.array_shapes, min_dims="not an int"),
        e(xpst.array_shapes, max_dims="not an int"),

        e(xpst.arrays, dtype=xp.int8, shape=(0.5,)),
        e(xpst.arrays, dtype=xp.int8, shape=1, fill=3),
        e(xpst.arrays, dtype=xp.int8, shape=1, elements="not a strategy"),

        e(xpst.from_dtype, dtype=1),
        e(xpst.from_dtype, dtype=xp.int8, min_value=-999),
        e(xpst.from_dtype, dtype=xp.int8, max_value=999),
        e(xpst.from_dtype, dtype=xp.uint8, min_value=-999),
        e(xpst.from_dtype, dtype=xp.uint8, max_value=999),

        e(xpst.integer_dtypes, sizes=()),
        e(xpst.integer_dtypes, sizes=(3,)),
        e(xpst.unsigned_integer_dtypes, sizes=()),
        e(xpst.unsigned_integer_dtypes, sizes=(3,)),
        e(xpst.floating_dtypes, sizes=()),
        e(xpst.floating_dtypes, sizes=(3,)),

        e(xpst.valid_tuple_axes, ndim=-1),
        e(xpst.valid_tuple_axes, ndim=2, min_size=-1),
        e(xpst.valid_tuple_axes, ndim=2, min_size=3, max_size=10),
        e(xpst.valid_tuple_axes, ndim=2, min_size=2, max_size=1),
        e(xpst.valid_tuple_axes, ndim=2.0, min_size=2, max_size=1),
        e(xpst.valid_tuple_axes, ndim=2, min_size=1.0, max_size=2),
        e(xpst.valid_tuple_axes, ndim=2, min_size=1, max_size=2.0),
        e(xpst.valid_tuple_axes, ndim=2, min_size=1, max_size=3),
        e(xpst.broadcastable_shapes, shape="a"),
        e(xpst.broadcastable_shapes, shape=(2, 2), min_side="a"),
        e(xpst.broadcastable_shapes, shape=(2, 2), min_dims="a"),
        e(xpst.broadcastable_shapes, shape=(2, 2), max_side="a"),
        e(xpst.broadcastable_shapes, shape=(2, 2), max_dims="a"),
        e(xpst.broadcastable_shapes, shape=(2, 2), min_side=-1),
        e(xpst.broadcastable_shapes, shape=(2, 2), min_dims=-1),
        e(xpst.broadcastable_shapes, shape=(2, 2), min_side=1, max_side=0),
        e(xpst.broadcastable_shapes, shape=(2, 2), min_dims=1, max_dims=0),
        e(
            xpst.broadcastable_shapes,  # max_side too small
            shape=(5, 1),
            min_dims=2,
            max_dims=4,
            min_side=2,
            max_side=3,
        ),
        e(
            xpst.broadcastable_shapes,  # min_side too large
            shape=(0, 1),
            min_dims=2,
            max_dims=4,
            min_side=2,
            max_side=3,
        ),
        e(
            xpst.broadcastable_shapes,  # default max_dims unsatisfiable
            shape=(5, 3, 2, 1),
            min_dims=3,
            max_dims=None,
            min_side=2,
            max_side=3,
        ),
        e(
            xpst.broadcastable_shapes,  # default max_dims unsatisfiable
            shape=(0, 3, 2, 1),
            min_dims=3,
            max_dims=None,
            min_side=2,
            max_side=3,
        ),

        e(xpst.mutually_broadcastable_shapes, num_shapes=0),
        e(xpst.mutually_broadcastable_shapes, num_shapes="a"),
        e(xpst.mutually_broadcastable_shapes, num_shapes=2, base_shape="a"),
        e(
            xpst.mutually_broadcastable_shapes,  # min_side is invalid type
            num_shapes=2,
            min_side="a",
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # min_dims is invalid type
            num_shapes=2,
            min_dims="a",
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # max_side is invalid type
            num_shapes=2,
            max_side="a",
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # max_side is invalid type
            num_shapes=2,
            max_dims="a",
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # min_side is out of domain
            num_shapes=2,
            min_side=-1,
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # min_dims is out of domain
            num_shapes=2,
            min_dims=-1,
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # max_side < min_side
            num_shapes=2,
            min_side=1,
            max_side=0,
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # max_dims < min_dims
            num_shapes=2,
            min_dims=1,
            max_dims=0,
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # max_side too small
            num_shapes=2,
            base_shape=(5, 1),
            min_dims=2,
            max_dims=4,
            min_side=2,
            max_side=3,
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # min_side too large
            num_shapes=2,
            base_shape=(0, 1),
            min_dims=2,
            max_dims=4,
            min_side=2,
            max_side=3,
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # user-specified max_dims unsatisfiable
            num_shapes=1,
            base_shape=(5, 3, 2, 1),
            min_dims=3,
            max_dims=4,
            min_side=2,
            max_side=3,
        ),
        e(
            xpst.mutually_broadcastable_shapes,  # user-specified max_dims unsatisfiable
            num_shapes=2,
            base_shape=(0, 3, 2, 1),
            min_dims=3,
            max_dims=4,
            min_side=2,
            max_side=3,
        ),

        e(xpst.basic_indices, shape=0),
        e(xpst.basic_indices, shape=("1", "2")),
        e(xpst.basic_indices, shape=(0, -1)),
        e(xpst.basic_indices, shape=(0, 0), allow_newaxis=None),
        e(xpst.basic_indices, shape=(0, 0), allow_ellipsis=None),
        e(xpst.basic_indices, shape=(0, 0), min_dims=-1),
        e(xpst.basic_indices, shape=(0, 0), min_dims=1.0),
        e(xpst.basic_indices, shape=(0, 0), max_dims=-1),
        e(xpst.basic_indices, shape=(0, 0), max_dims=1.0),
        e(xpst.basic_indices, shape=(0, 0), min_dims=2, max_dims=1),
    ],
)
def test_raise_invalid_argument(function, kwargs):
    with raises(InvalidArgument):
        function(**kwargs).example()
