These strategies now have first-party support in [`hypothesis.extra.array_api`](https://hypothesis.readthedocs.io/en/latest/numpy.html#array-api),
so please don't use this package!

---

# Hypothesis strategies for Array API libraries

**Note:** `hypothesis-array-api` uses private APIs from Hypothesis
and so should be considered unstable.

## Install

You can get the strategies from PyPI.

```bash
pip install hypothesis-array-api
```

To install from source,
[get Poetry](https://python-poetry.org/docs/#installation)
and then `poetry install` inside the repository.
Using `poetry shell` is a good idea for development,
where you can use `pytest` to run the full test suite.

## Quickstart

```python
from numpy import array_api as xp

from hypothesis import given
from hypothesis_array import get_strategies_namespace

xps = get_strategies_namespace(xp)

@given(xps.arrays(dtype=xps.scalar_strategies(), shape=xps.array_shapes()))
def your_test(array):
    ...
```

## Contributors

[@honno](https://github.com/honno/) created these strategies
with input from
[@mattip](https://github.com/mattip),
[@asmeurer](https://github.com/asmeurer),
[@rgommers](https://github.com/rgommers)
and other great folk from
[@Quansight-Labs](https://github.com/Quansight-Labs).

Inspiration was taken from the
[NumPy strategies](https://hypothesis.readthedocs.io/en/latest/numpy.html#numpy)
that Hypothesis ships with at `hypothesis.extra.numpy`.
Thanks to the Hypothesis contributors who helped shape it, including:
[@Zac-HD](https://github.com/Zac-HD),
[@rsokl](https://github.com/rsokl),
[@DRMacIver](https://github.com/DRMacIver),
[@takluyver](https://github.com/takluyver),
[@rdturnermtl](https://github.com/rdturnermtl),
[@kprzybyla](https://github.com/kprzybyla),
[@sobolevn](https://github.com/sobolevn),
[@kir0ul](https://github.com/kir0ul),
[@lmount](https://github.com/lmount),
[@jdufresne](https://github.com/jdufresne),
[@gsnsw-felixs](https://github.com/gsnsw-felixs) and
[@alexwlchan](https://github.com/alexwlchan).
