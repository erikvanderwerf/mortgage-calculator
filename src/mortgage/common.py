"""Common methods for formatting mortgage reports."""

#  Copyright (c) 2024 Erik VanderWerf
__all__ = ["b", "p", "pct", "q", "usd"]

import textwrap
from decimal import Decimal
from typing import Any

from moneyed import Money


def b(text: str, only_if: Any) -> str:
    """Return text only if a condition is True, otherwise return empty string."""
    return text if only_if else ""


def p(paragraph: str, suffix="\n\n") -> str:
    """Wrap text into a paragraph, with newlines appended."""
    return textwrap.fill(paragraph, width=120) + suffix


def pct(amount: str) -> Decimal:
    """Return amount as a Decimal divided by 100."""
    return Decimal(amount) / 100


def q(name: str) -> str:
    """Wrap a name in quotes if it contains any space characters."""
    if any(c.isspace() for c in name):
        return repr(name)
    return name


def usd(amount: Any) -> Money:
    """Return an amount as dollars."""
    return Money(amount, "USD")
