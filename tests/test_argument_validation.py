from hypothesis.errors import InvalidArgument
from pytest import mark, param, raises

import hypothesis_array as _xpst

from .xputils import create_array_module

xp = create_array_module()
xpst = _xpst.get_strategies_namespace(xp)


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
