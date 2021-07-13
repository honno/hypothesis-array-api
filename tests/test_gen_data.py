from hypothesis import given

import hypothesis_array as amst


@given(amst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)
