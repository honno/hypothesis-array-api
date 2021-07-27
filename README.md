# Hypothesis strategies for Array API libraries

## Install

You can get the strategies from PyPI.

```bash
pip install hypothesis-array-api
```

To install from source,
[get Poetry](https://python-poetry.org/docs/#installation)
and then `poetry install` inside the repository.
Using `poetry shell` is a good idea for development,
where you can use `pytest` to run the full test suite
(note there a lot of expected warnings I need to declutter.)

## Quickstart

```python
from numpy import _array_api as xp

from hypothesis import given
from hypothesis_array import get_strategies_namespace

xpst = get_strategies_namespace(xp)

@given(xpst.arrays(dtype=xpst.scalar_strategies(), shape=xpst.array_shapes()))
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

Additionaly inspiration was taken from the
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
