"""Extra Principal Payments on a Mortgage."""

#  Copyright (c) 2024 Erik VanderWerf
__all__ = ["Extra", "StaticExtra", "SumExtra", "VariableExtra"]

from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
from operator import add
from typing import Mapping

from moneyed import Money
from typing_extensions import override


class Extra(ABC):
    """Extra principal payments on a mortgage."""

    def __add__(self, other):
        if not isinstance(other, Extra):
            return NotImplemented
        extras = []
        if isinstance(self, SumExtra):
            extras.extend(self.extras)
        else:
            extras.append(self)
        if isinstance(other, SumExtra):
            extras.extend(other.extras)
        else:
            extras.append(other)
        return SumExtra(Counter(extras))

    @abstractmethod
    def on(self, payment: int) -> Money:
        """Return the additional amount paid for a given payment."""
        raise NotImplementedError()


class SumExtra(Extra):
    """Summation of multiple extra mortgage payment amounts."""

    def __init__(self, extras: Counter[Extra]):
        """Summation of multiple extra mortgage payment amounts."""
        self.extras: Counter[Extra] = extras

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, SumExtra):
            return False
        return self.extras == other.extras

    def __hash__(self) -> int:
        return hash(self.extras)

    def __repr__(self):
        extras = self.extras
        return f"{type(self).__name__}({extras=})"

    @override
    def on(self, payment: int) -> Money:
        return reduce(add, (e.on(payment) for e in self.extras))


class StaticExtra(Extra):
    """Same extra amount paid on every mortgage payment."""

    def __init__(self, value: Money):
        """Same extra amount paid on every mortgage payment."""
        self.value: Money = value

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, StaticExtra):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self):
        extra = self.value
        return f"{type(self).__name__}({extra=})"

    @override
    def on(self, _: int) -> Money:
        return self.value


class VariableExtra(Extra):
    """Additional amount that changes depending on the specific payment."""

    def __init__(self, values: Mapping[int, Money]):
        """Additional amount that changes depending on the specific payment."""
        self.default_currency = next(iter(values.values())).currency
        self.values = values

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, VariableExtra):
            return False
        return self.values == other.values

    def __hash__(self) -> int:
        return hash(tuple(self.values.items()))

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.values)})"

    @override
    def on(self, payment: int) -> Money:
        try:
            return self.values[payment]
        except KeyError:
            return Money(0, self.default_currency)
