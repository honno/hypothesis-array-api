from hypothesis import given

import hypothesis_array as amst


@given(amst.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@given(amst.scalar_names())
def test_can_generate_dtype_names(name):
    assert isinstance(name, str)
    assert name in [
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float32", "float64",
    ]
