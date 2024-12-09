"""Fixed Rate Mortgage."""

#  Copyright (c) 2024 Erik VanderWerf
from __future__ import annotations

__all__ = ["FixedRateMortgage", "MortgageSummary"]

from decimal import Decimal
from datetime import date as d
from typing import Literal, Hashable

from moneyed import Money

from mortgage.common import usd
from mortgage.loan import FixedRateLoan, LoanSummary
from mortgage.extra import Extra, StaticExtra


class FixedRateMortgage(Hashable):
    """Fixed Rate Mortgage."""

    def __init__(
        self,
        name: str,
        home_price: Money,
        down_payment: Money | Decimal,
        origination: Money,
        seller_concessions: Money,
        start: d,
        term_years: int,
        rate: Decimal,
        extra: Extra,
    ):
        """Create a Fixed Rate Mortgage."""
        if not isinstance(down_payment, Money):
            down_payment = home_price * down_payment
        self.name: str = name
        self.home_price: Money = home_price
        self.down_payment: Money = down_payment
        self.origination: Money = origination
        self.seller_concessions: Money = seller_concessions

        self.loan = FixedRateLoan(
            amount=home_price - down_payment,
            start=start,
            payments=12 * term_years,
            payments_per_year=12,
            rate=rate,
            extra=extra,
        )

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, FixedRateMortgage):
            return False
        return (
            self.name == other.name
            and self.home_price == other.home_price
            and self.down_payment == other.down_payment
            and self.origination == other.origination
            and self.seller_concessions == other.seller_concessions
            and self.loan == other.loan
        )

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self):
        lender = self.name
        home_price = self.home_price
        down_payment = self.down_payment
        lender_costs = self.origination
        seller_concessions = self.seller_concessions
        loan = repr(self.loan)

        return (
            f"{type(self).__name__}({lender=}, {home_price=}, {down_payment=},"
            f" {lender_costs=}, {seller_concessions=}, {loan=})"
        )

    def cost(self, payment: int | None = None) -> Money:
        """Return the cost of the Mortgage at term or at a given payment."""
        return self.origination + (
            self.loan.schedule.cumulative_interest(-1)
            if payment is None
            else self.loan.schedule.cumulative_interest(payment)
        )

    def recalculate(
        self,
        adjust_down_payment: Money | None = None,
        adjust_extra: Extra | None = None,
    ) -> FixedRateMortgage:
        """Recalculate a fixed rate mortgage."""
        if adjust_down_payment is None:
            adjust_down_payment = Money(0, self.down_payment.currency)
        if adjust_extra is None:
            adjust_extra = StaticExtra(Money(0, self.loan.amount.currency))
        return FixedRateMortgage(
            name=self.name,
            home_price=self.home_price,
            down_payment=self.down_payment + adjust_down_payment,
            origination=self.origination,
            seller_concessions=self.seller_concessions,
            start=self.loan.start,
            term_years=self.loan.term_payments // 12,
            rate=self.loan.rate,
            extra=self.loan.extra + adjust_extra,
        )

    def is_costlier_than(
        self, other: FixedRateMortgage
    ) -> int | Literal["same"] | Literal["never"]:
        """Return the payment on which this mortgage is more expensive than `other`."""
        if self.loan == other.loan:
            return "same"
        for i in range(1, other.loan.term_payments + 1):
            if other.cost(i) < self.cost(i):
                return i
        return "never"


class MortgageSummary:
    """Summarize a fixed rate mortgage."""

    def __init__(self, mortgage: FixedRateMortgage):
        """Summarize a fixed rate mortgage."""
        self.locale = "en_US"
        self.mortgage = mortgage

    def __str__(self):
        return f"{self.mortgage.name} {LoanSummary(self.mortgage.loan, self.locale)}"


if __name__ == "__main__":
    example = FixedRateMortgage(
        name="Example",
        home_price=usd(385_000),
        down_payment=Decimal("0.22"),
        origination=usd(14_000),
        seller_concessions=usd(10_000),
        start=d(2024, 7, 1),
        term_years=15,
        rate=Decimal("0.05"),
        extra=StaticExtra(usd(0)),
    )
    print(MortgageSummary(example))
