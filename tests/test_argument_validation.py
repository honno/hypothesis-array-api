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

        e(xpst.integer_dtypes, sizes=()),
        e(xpst.integer_dtypes, sizes=(3,)),
        e(xpst.unsigned_integer_dtypes, sizes=()),
        e(xpst.unsigned_integer_dtypes, sizes=(3,)),
        e(xpst.floating_dtypes, sizes=()),
        e(xpst.floating_dtypes, sizes=(3,)),
    ],
)
def test_raise_invalid_argument(function, kwargs):
    with raises(InvalidArgument):
        function(**kwargs).example()
