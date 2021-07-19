from hypothesis.errors import InvalidArgument
from pytest import mark, param, raises

import hypothesis_array as amst


def e(a, **kwargs):
    kw = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    return param(a, kwargs, id=f"{a.__name__}({kw})")


@mark.parametrize(
    ("function", "kwargs"),
    [
        e(amst.array_shapes, min_side=2, max_side=1),
        e(amst.array_shapes, min_dims=3, max_dims=2),
        e(amst.array_shapes, min_dims=-1),
        e(amst.array_shapes, min_side=-1),
        e(amst.array_shapes, min_side="not an int"),
        e(amst.array_shapes, max_side="not an int"),
        e(amst.array_shapes, min_dims="not an int"),
        e(amst.array_shapes, max_dims="not an int"),
    ],
)
def test_raise_invalid_argument(function, kwargs):
    with raises(InvalidArgument):
        function(**kwargs).example()
