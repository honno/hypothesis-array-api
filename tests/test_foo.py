from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

from hypothesis_array import arrays

def test_foo():
    assert isinstance(arrays(nps.unsigned_integer_dtypes(), (1,)), st.SearchStrategy)
