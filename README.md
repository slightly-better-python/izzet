# izzet
Ensure that ``value`` matches ``expected_type``.


## About
Borrowed from [Typeguard's](https://typeguard.readthedocs.io/en/latest/index.html) [check_type](https://github.com/agronholm/typeguard/blob/2.13.3/src/typeguard/__init__.py#L716)
function - The purpose of this package is to provide an easy and reliable way to verify that a given input matches a 
given class or generic type instance - and nothing else.

## Usage

```python
from typing import List
from izzet import izzet

izzet([1,2, "3"]).a(List[int]) # False
izzet([1,2, 3]).a(List[int]) # True
```
