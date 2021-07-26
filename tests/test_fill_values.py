from hypothesis import given
from hypothesis import strategies as st

import hypothesis_array as _xpst

from .xputils import create_array_module

xp = create_array_module()
xpst = _xpst.get_strategies_namespace(xp)


@given(xpst.arrays(xp.bool, (), fill=st.nothing()))
def test_can_generate_0d_arrays_with_no_fill(array):
    assert array.dtype == xp.bool
    assert array.ndim == 0
    assert array.shape == ()
