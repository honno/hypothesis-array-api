from math import prod

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument, Unsatisfiable
from pytest import mark, raises

from hypothesis_array import get_strategies_namespace

from .common.debug import find_any, minimal
from .common.utils import fails_with
from .xputils import DTYPE_NAMES, create_array_module

xp = create_array_module()
xpst = get_strategies_namespace(xp)


@given(xpst.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["all"])


@given(xpst.boolean_dtypes())
def test_can_generate_boolean_dtypes(dtype):
    assert dtype == xp.bool


@given(xpst.integer_dtypes())
def test_can_generate_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["ints"])


@given(xpst.unsigned_integer_dtypes())
def test_can_generate_unsigned_integer_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["uints"])


@given(xpst.floating_dtypes())
def test_can_generate_floating_dtypes(dtype):
    assert dtype in (getattr(xp, name) for name in DTYPE_NAMES["floats"])


def test_minimise_scalar_dtypes():
    assert minimal(xpst.scalar_dtypes()) == xp.bool


@mark.parametrize(
    "strat_func, sizes",
    [
        (xpst.integer_dtypes, 8),
        (xpst.unsigned_integer_dtypes, 8),
        (xpst.floating_dtypes, 32),
    ]
)
def test_can_specify_size_as_an_int(strat_func, sizes):
    strat_func(sizes)


@given(xpst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@settings(deadline=None, max_examples=10)
@given(st.integers(0, 10), st.integers(0, 9), st.integers(0), st.integers(0))
def test_minimise_array_shapes(min_dims, dim_range, min_side, side_range):
    strat = xpst.array_shapes(
        min_dims=min_dims,
        max_dims=min_dims + dim_range,
        min_side=min_side,
        max_side=min_side + side_range,
    )
    smallest = minimal(strat)
    assert len(smallest) == min_dims and all(k == min_side for k in smallest)


@mark.parametrize(
    "kwargs", [{"min_side": 100}, {"min_dims": 15}, {"min_dims": 32}]
)
def test_interesting_array_shapes_argument(kwargs):
    xpst.array_shapes(**kwargs).example()


@given(st.data())
def test_can_generate_arrays_from_scalars(data):
    dtype = data.draw(xpst.scalar_dtypes())
    array = data.draw(xpst.arrays(dtype, ()))

    assert array.dtype == dtype
    # TODO check array.__array_namespace__()


@given(st.data())
def test_can_generate_arrays_from_shapes(data):
    shape = data.draw(xpst.array_shapes())
    array = data.draw(xpst.arrays(xp.bool, shape))

    assert array.ndim == len(shape)
    assert array.shape == shape
    assert array.size == prod(shape)
    # TODO check array.__array_namespace__()


@given(st.data())
def test_can_draw_arrays_from_scalar_strategies(data):
    strat = data.draw(st.sampled_from([
        xpst.scalar_dtypes(),
        xpst.boolean_dtypes(),
        xpst.integer_dtypes(),
        xpst.unsigned_integer_dtypes(),
        xpst.floating_dtypes(),
    ]))
    array = data.draw(xpst.arrays(strat, ()))  # noqa
    # TODO check array.__array_namespace__()


@given(xpst.arrays(xp.bool, xpst.array_shapes()))
def test_can_draw_arrays_from_shapes_strategy(array):
    assert array.dtype == xp.bool
    # TODO check array.__array_namespace__()


@given(xpst.arrays(xp.bool, st.integers(0, 100)))
def test_can_draw_arrays_from_integers_strategy_as_shape(array):
    assert array.dtype == xp.bool
    # TODO check array.__array_namespace__()


@given(xpst.arrays(xp.bool, ()))
def test_empty_dimensions_are_arrays(array):
    # TODO check array.__array_namespace__()
    assert array.dtype == xp.bool
    assert array.ndim == 0
    assert array.shape == ()


@given(xpst.arrays(xp.bool, (1, 0, 1)))
def test_can_handle_zero_dimensions(array):
    assert array.dtype == xp.bool
    assert array.shape == (1, 0, 1)


@given(xpst.arrays(xp.uint32, (5, 5)))
def test_generates_unsigned_ints(array):
    assert xp.all(array >= 0)


def test_generates_and_minimizes():
    strat = xpst.arrays(xp.float32, (2, 2))
    assert xp.all(minimal(strat) == 0)


def test_minimise_array_strategy():
    smallest = minimal(
        xpst.arrays(xpst.scalar_dtypes(), xpst.array_shapes(max_dims=3, max_side=3)),
    )
    assert smallest.dtype == xp.bool
    assert not xp.any(smallest)


def test_can_minimize_large_arrays():
    array = minimal(
        xpst.arrays(xp.uint32, 100),
        lambda x: xp.any(x) and not xp.all(x),
        timeout_after=60,
    )

    assert xp.all(xp.logical_or(array == 0, array == 1))

    # xp.nonzero() is optional for Array API libraries
    if hasattr(xp, "nonzero"):
        nonzero_count = 0
        for nonzero_indices in xp.nonzero(array):
            nonzero_count += nonzero_indices.size
        assert nonzero_count in (1, array.size - 1)


@given(xpst.arrays(xp.int8, st.integers(0, 20), unique=True))
def test_array_values_are_unique(array):
    # xp.unique() is optional for Array API libraries
    if hasattr(xp, "unique"):
        unique_values = xp.unique(array)
        assert unique_values.size == array.size


def test_cannot_generate_unique_array_of_too_many_elements():
    strat = xpst.arrays(xp.int8, 10, elements=st.integers(0, 5), unique=True)
    with raises(Unsatisfiable):
        strat.example()


@given(
    xpst.arrays(
        dtype=xp.float32,
        shape=st.integers(0, 20),
        elements=st.just(0.0),
        fill=st.just(xp.nan),
        unique=True,
    )
)
def test_array_values_are_unique_high_collision(array):
    hits = (array == 0.0)
    nzeros = 0
    for i in range(array.size):
        if hits[i]:
            nzeros += 1
    assert nzeros <= 1


@given(xpst.arrays(xp.int8, (4,), elements=st.integers(0, 3), unique=True))
def test_generates_all_values_for_unique_array(array):
    # xp.unique() is optional for Array API libraries
    if hasattr(xp, "unique"):
        unique_values = xp.unique(array)
        assert unique_values.size == array.size


def test_may_fill_with_nan_when_unique_is_set():
    find_any(
        xpst.arrays(
            dtype=xp.float32,
            shape=10,
            elements=st.floats(allow_nan=False),
            unique=True,
            fill=st.just(xp.nan),
        ),
        lambda x: xp.any(xp.isnan(x)),
    )


@fails_with(InvalidArgument)
@given(
    xpst.arrays(
        dtype=xp.float32,
        shape=10,
        elements=st.floats(allow_nan=False),
        unique=True,
        fill=st.just(0.0),
    )
)
def test_may_not_fill_with_non_nan_when_unique_is_set(_):
    pass


@mark.parametrize(
    "kwargs",
    [
        {"elements": st.just(300)},
        {"elements": st.nothing(), "fill": st.just(300)},
    ],
)
@fails_with(InvalidArgument)
@given(st.data())
def test_may_not_use_overflowing_integers(kwargs, data):
    strat = xpst.arrays(dtype=xp.int8, shape=1, **kwargs)
    data.draw(strat)


@mark.parametrize("fill", [False, True])
@mark.parametrize(
    "dtype, strat",
    [
        (xp.float32, st.floats(min_value=10 ** 40, allow_infinity=False)),
        (xp.float64, st.floats(min_value=10 ** 40, allow_infinity=False)),
    ]
)
@fails_with(InvalidArgument)
@given(st.data())
def test_may_not_use_unrepresentable_elements(fill, dtype, strat, data):
    if fill:
        kw = {"elements": st.nothing(), "fill": strat}
    else:
        kw = {"elements": strat}
    strat = xpst.arrays(dtype=dtype, shape=1, **kw)
    data.draw(strat)


@given(
    xpst.arrays(dtype=xp.float32, shape=10, elements={"min_value": 0, "max_value": 1})
)
def test_floats_can_be_constrained_at_low_width(array):
    assert xp.all(array >= 0)
    assert xp.all(array <= 1)


@given(
    xpst.arrays(
        dtype=xp.float32,
        shape=10,
        elements={
            "min_value": 0,
            "max_value": 1,
            "exclude_min": True,
            "exclude_max": True,
        },
    )
)
def test_floats_can_be_constrained_at_low_width_excluding_endpoints(array):
    assert xp.all(array > 0)
    assert xp.all(array < 1)
