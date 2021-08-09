# This file was part of and modifed from Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# ./CONTRIBUTING.rst for a full list of people who may hold copyright,
# and consult the git log of ./hypothesis-python/tests/numpy/test_gen_data.py
# if you need to determine who owns an individual contribution.
# ('.' represents the root of the Hypothesis git repository)
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math

from hypothesis import assume, given
from hypothesis import strategies as st
from pytest import mark

from hypothesis_array import get_strategies_namespace

from .common.debug import find_any
from .xputils import create_array_module

xp = create_array_module()
xps = get_strategies_namespace(xp)


@mark.parametrize(
    "condition",
    [
        lambda ix: Ellipsis in ix,
        lambda ix: Ellipsis not in ix,
    ],
    ids=["Ellipsis in ix", "Ellipsis not in ix"]
)
def test_indices_options(condition):
    indexers = xps.array_shapes(min_dims=0, max_dims=32).flatmap(
        lambda shape: xps.indices(shape, allow_none=True)
    )
    find_any(indexers, condition)


def test_indices_can_generate_empty_tuple():
    find_any(xps.indices(shape=(0, 0), allow_ellipsis=True), lambda ix: ix == ())


def test_indices_can_generate_non_tuples():
    find_any(
        xps.indices(shape=(0, 0), allow_ellipsis=True),
        lambda ix: not isinstance(ix, tuple),
    )


def test_indices_can_generate_long_ellipsis():
    # Runs of slice(None) - such as [0,:,:,:,0] - can be replaced by e.g. [0,...,0]
    find_any(
        xps.indices(shape=(1, 0, 0, 0, 1), allow_ellipsis=True),
        lambda ix: len(ix) == 3 and ix[1] == Ellipsis,
    )


@given(
    xps.indices(shape=(0, 0, 0, 0, 0)).filter(
        lambda idx: isinstance(idx, tuple) and Ellipsis in idx
    )
)
def test_indices_replaces_whole_axis_slices_with_ellipsis(idx):
    # If ... is in the slice, it replaces all ,:, entries for this shape.
    assert slice(None) not in idx


@given(
    shape=xps.array_shapes(min_dims=0, max_side=4)
    | xps.array_shapes(min_dims=0, min_side=0, max_side=10),
    min_dims=st.integers(0, 5),
    allow_ellipsis=st.booleans(),
    allow_none=st.booleans(),
    data=st.data(),
)
def test_indices_generate_valid_indexers(
    shape, min_dims, allow_ellipsis, allow_none, data
):
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    indexer = data.draw(
        xps.indices(
            shape,
            min_dims=min_dims,
            max_dims=max_dims,
            allow_ellipsis=allow_ellipsis,
            allow_none=allow_none,
        ),
        label="indexer",
    )
    # Check that disallowed things are indeed absent
    if not allow_none:
        if isinstance(indexer, tuple):
            assert 0 <= len(indexer) <= len(shape) + int(allow_ellipsis)
        else:
            assert 1 <= len(shape) + int(allow_ellipsis)
        assert None not in shape
    if not allow_ellipsis:
        assert Ellipsis not in shape

    if 0 in shape:
        # If there's a zero in the shape, the array will have no elements.
        array = xp.zeros(shape)
        assert array.size == 0
    elif math.prod(shape) <= 10 ** 5:
        # If it's small enough to instantiate, do so with distinct elements.
        array = xp.arange(math.prod(shape)).reshape(shape)
    else:
        # We can't cheat on this one, so just try another.
        assume(False)