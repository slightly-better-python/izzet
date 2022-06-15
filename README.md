# izzet
Ensure that ``value`` matches ``expected_type``.


## About
Inspired and borrowed from [Typeguard's](https://typeguard.readthedocs.io/en/latest/index.html) [check_type](https://github.com/agronholm/typeguard/blob/f87c1c0b8689b294a1e9120a92038f1a7fb68321/src/typeguard/__init__.py#L69)
function - The purpose of this package is to provide an easy and reliable way to verify a given input matches a given
class or generic type instance - and nothing else.

## Usage

```python
from typing import List
from izzet import izzet

izzet([1,2, "3"], List[int]) # False
izzet([1,2, 3], List[int]) # True
```
