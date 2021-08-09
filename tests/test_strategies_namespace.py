from pytest import mark

import hypothesis_array as xps_

from .xputils import create_array_module

xp = create_array_module()
xps = xps_.get_strategies_namespace(xp)


funcnames = xps_.__all__
funcnames.remove("get_strategies_namespace")


@mark.parametrize("name", funcnames)
def test_names_wrapped(name):
    namespaced_func = getattr(xps, name)
    assert namespaced_func.__name__ == name
