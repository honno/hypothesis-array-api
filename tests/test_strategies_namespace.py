from pytest import mark

import hypothesis_array as _xpst

from .xputils import create_array_module

xp = create_array_module()
xpst = _xpst.get_strategies_namespace(xp)


funcnames = _xpst.__all__
funcnames.remove("get_strategies_namespace")


@mark.parametrize("name", funcnames)
def test_names_wrapped(name):
    namespaced_func = getattr(xpst, name)
    assert namespaced_func.__name__ == name
