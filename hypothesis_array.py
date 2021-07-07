from hypothesis import strategies as st

def arrays(dtype, shape, *, elements=None, fill=None, unique=False):
    return st.booleans()
