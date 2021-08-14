from inspect import signature

from pytest import mark

import hypothesis_array as generic_xps
from hypothesis_array import get_strategies_namespace

from .xputils import create_array_module

xp = create_array_module(assign=(("__name__", "mockpy"),))
xps = get_strategies_namespace(xp)


stratnames = [
    "from_dtype",
    "arrays",
    "array_shapes",
    "scalar_dtypes",
    "boolean_dtypes",
    "numeric_dtypes",
    "integer_dtypes",
    "unsigned_integer_dtypes",
    "floating_dtypes",
    "valid_tuple_axes",
    "broadcastable_shapes",
    "mutually_broadcastable_shapes",
    "indices",
]


@mark.parametrize("name", stratnames)
def test_namespaced_methods_wrapped(name):
    """Namespaced strategies have readable method names, even if they are lambdas."""
    namespaced_func = getattr(xps, name)
    assert namespaced_func.__name__ == name


xp_stratnames = []
for func in [getattr(generic_xps, name) for name in stratnames]:
    sig = signature(func)
    if "xp" in sig.parameters:
        xp_stratnames.append(func.__name__)


@mark.parametrize(
    "name, strat",
    [
        # Generic strategies
        ("from_dtype", generic_xps.from_dtype(xp, xp.int8)),
        ("arrays", generic_xps.arrays(xp, xp.int8, 5)),
        ("scalar_dtypes", generic_xps.scalar_dtypes(xp)),
        ("boolean_dtypes", generic_xps.boolean_dtypes(xp)),
        ("numeric_dtypes", generic_xps.numeric_dtypes(xp)),
        ("integer_dtypes", generic_xps.integer_dtypes(xp)),
        ("unsigned_integer_dtypes", generic_xps.unsigned_integer_dtypes(xp)),
        ("floating_dtypes", generic_xps.floating_dtypes(xp)),
        # Namespaced strategies
        ("from_dtype", xps.from_dtype(xp.int8)),
        ("arrays", xps.arrays(xp.int8, 5)),
        ("scalar_dtypes", xps.scalar_dtypes()),
        ("boolean_dtypes", xps.boolean_dtypes()),
        ("numeric_dtypes", xps.numeric_dtypes()),
        ("integer_dtypes", xps.integer_dtypes()),
        ("unsigned_integer_dtypes", xps.unsigned_integer_dtypes()),
        ("floating_dtypes", xps.floating_dtypes()),
    ]
)
def test_xp_strategies_pretty_repr(name, strat):
    """Strategies that take xp use its __name__ for their own repr."""
    assert repr(strat).startswith(name), f"{name} not in strat repr"
    assert len(repr(strat)) < 50, "strat repr looks too long"
    assert xp.__name__ in repr(strat), f"{xp.__name__} not in strat repr"
