"""A Fixed Rate Mortgage."""

#  Copyright (c) 2024 Erik VanderWerf
from __future__ import annotations

__all__ = [
    "FixedRateLoan",
    "LoanSideBySide",
    "LoanSummary",
    "LoanScheduleTable",
    "Payment",
]

from collections.abc import Collection

from dataclasses import dataclass
from datetime import date as d
from decimal import Decimal

from typing import List, Iterator, Tuple, Iterable

import tabulate
from babel.numbers import format_percent
from dateutil.relativedelta import relativedelta
from moneyed import Money, format_money
from more_itertools import intersperse

from mortgage.common import usd, p
from mortgage.extra import Extra, StaticExtra


@dataclass(order=True, frozen=True)
class Payment:
    """A mortgage payment."""

    i: int
    date: d
    rate: Decimal
    beginning_balance: Money
    scheduled_payment: Money
    extra_payment: Money
    total_payment: Money
    principal_paid: Money
    interest_paid: Money
    ending_balance: Money
    cumulative_extra: Money
    cumulative_interest: Money


class Schedule(Collection[Payment]):
    """Schedule of loan payments."""

    def __init__(self, payments: List[Payment]):
        """Schedule of loan payments."""
        for i, payment in enumerate(payments, start=1):
            if i != payment.i:
                raise ValueError("Payment indexes must match.")
        self.schedule: List[Payment] = payments

    def __contains__(self, item) -> bool:
        return item in self.schedule

    def __iter__(self) -> Iterator[Payment]:
        return iter(self.schedule)

    def __len__(self) -> int:
        return len(self.schedule)

    def cumulative_interest(self, payment: int) -> Money:
        """Return the cumulative interest up to a given payment on a loan schedule."""
        if 0 == payment:
            raise ValueError("There is no zeroth payment.")
        index = payment if payment < 0 else (min(len(self.schedule), payment) - 1)
        return self.schedule[index].cumulative_interest

    def last(self) -> Payment:
        """Return the last loan payment."""
        return self.schedule[-1]


class FixedRateLoan:
    """Loan with a fixed rate."""

    def __init__(
        self,
        amount: Money,
        start: d,
        payments: int,
        payments_per_year: int,
        rate: Decimal,
        extra: Extra,
    ):
        """Loan with a fixed rate."""
        self.amount: Money = amount
        self.start: d = start
        self.term_payments: int = payments
        self.payments_per_year: int = payments_per_year
        self.rate: Decimal = rate
        self.extra: Extra = extra

        principal = amount
        months_between_payments: int = 12 // payments_per_year
        cumulative_interest: Money = Money(0, amount.currency)
        cumulative_extra: Money = Money(0, amount.currency)
        schedule: List[Payment] = []

        r: Decimal = self.rate / payments_per_year
        scheduled_payment: Money = FixedRateLoan.calculate_payment_from_payment_rate(
            principal, r, payments
        )
        for payment_index in range(payments):
            today = start + relativedelta(
                months=months_between_payments * payment_index
            )
            extra_today = extra.on(payment_index + 1)
            interest_due = r * principal

            max_due = principal + interest_due
            from_scheduled: Money = min(scheduled_payment, max_due)
            from_extra: Money = min(extra_today, max_due - from_scheduled)
            total_paid: Money = from_scheduled + from_extra
            principal_paid = total_paid - interest_due
            ending_balance = principal - principal_paid

            cumulative_extra += from_extra
            cumulative_interest += interest_due
            schedule.append(
                Payment(
                    i=payment_index + 1,
                    date=today,
                    rate=self.rate,
                    beginning_balance=principal,
                    scheduled_payment=from_scheduled,
                    extra_payment=from_extra,
                    total_payment=total_paid,
                    principal_paid=principal_paid,
                    interest_paid=interest_due,
                    ending_balance=ending_balance,
                    cumulative_extra=cumulative_extra,
                    cumulative_interest=cumulative_interest,
                )
            )
            principal = ending_balance
            if principal.amount.is_zero():
                break

        self.schedule = Schedule(schedule)
        self.scheduled_payment: Money = scheduled_payment

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, FixedRateLoan):
            return False
        return (
            self.amount == other.amount
            # and self.start == other.start
            and self.term_payments == other.term_payments
            and self.rate == other.rate
            and self.extra == other.extra
        )

    def __repr__(self):
        amount = self.amount
        start = self.start
        payments = self.term_payments
        payments_per_year = self.payments_per_year
        rate = self.rate
        extra = self.extra

        return (
            f"{type(self).__name__}({amount=}, {start=}, {payments=},"
            f" {payments_per_year=}, {rate=}, {extra=})"
        )

    @staticmethod
    def calculate_payment_from_payment_rate(
        amount: Money, payment_rate: Decimal, total_payments: int
    ) -> Money:
        """Payment on a fixed rate loan using the normalized rate for each payment."""
        rn: Decimal = (1 + payment_rate) ** total_payments
        payment = amount * ((payment_rate * rn) / (rn - 1))
        return payment

    @staticmethod
    def calculate_payment_from_annual_rate(
        amount: Money, annual_rate: Decimal, payments_per_year: int, total_payments: int
    ) -> Money:
        """Payment on a fixed rate loan based on the annual rate."""
        r = annual_rate / payments_per_year
        return FixedRateLoan.calculate_payment_from_payment_rate(
            amount, r, total_payments
        )

    def recalculate(self, set_amount: Money, add_extra: Extra) -> FixedRateLoan:
        """Recalculate a loan."""
        return FixedRateLoan(
            amount=set_amount,
            start=self.start,
            payments=self.term_payments,
            payments_per_year=self.payments_per_year,
            rate=self.rate,
            extra=self.extra + add_extra,
        )


class LoanSummary:
    """Summarize a loan."""

    def __init__(self, loan: FixedRateLoan, locale: str):
        """Summarize a loan."""
        self.loan = loan
        self.locale = locale

    def __str__(self) -> str:
        loan = self.loan
        locale = self.locale
        amount = format_money(loan.amount, locale=locale)
        rate = format_percent(loan.rate, format="#.###%", locale=locale)
        return f"{amount} at {rate} over {len(loan.schedule)} payments."

    def detail(self) -> str:
        """Produce a detailed summary of the loan."""
        loan = self.loan
        locale = self.locale
        scheduled = format_money(loan.scheduled_payment, locale=locale)
        last_payment = loan.schedule.last()
        extra = format_money(last_payment.cumulative_extra, locale=locale)
        interest = format_money(last_payment.cumulative_interest, locale=locale)
        return p(
            str(self)
            + f" Scheduled monthly payment of {scheduled} with total extra payments of"
            f" {extra} incur {interest} in interest."
        )


class LoanScheduleTable:
    """Summarize a loan schedule."""

    def __init__(self, loan: FixedRateLoan, locale: str, pad: bool | int = False):
        """Summarize a loan schedule."""
        self.loan: FixedRateLoan = loan
        self.locale: str = locale
        self.pad: bool | int = pad

    def __str__(self) -> str:
        loan = self.loan
        locale = self.locale
        pad = self.pad
        return f"{type(self).__name__}({loan=}, {locale=}, {pad=})"

    def report(self) -> str:
        """Produce a detailed report of a loan schedule."""
        headers = [
            "no",
            "Date",
            "Rate",
            "Balance",
            "Scheduled",
            "Extra",
            "Total",
            "Principal",
            "Interest",
            "Ending",
            "Cumulative",
        ]
        payments: List[Tuple[int, str, str, str, str, str, str, str, str, str, str]]
        payments = []
        loan = self.loan
        i = 0
        for pmt in loan.schedule:
            i = pmt.i
            payments.append(
                (
                    i,
                    pmt.date.strftime("%b-%Y"),
                    format_percent(pmt.rate, format="#.###%", locale=self.locale),
                    format_money(pmt.beginning_balance, locale=self.locale),
                    format_money(pmt.scheduled_payment, locale=self.locale),
                    format_money(pmt.extra_payment, locale=self.locale),
                    format_money(pmt.total_payment, locale=self.locale),
                    format_money(pmt.principal_paid, locale=self.locale),
                    format_money(pmt.interest_paid, locale=self.locale),
                    format_money(pmt.ending_balance, locale=self.locale),
                    format_money(pmt.cumulative_interest, locale=self.locale),
                )
            )
        pad: int
        if isinstance(self.pad, int):
            pad = self.pad
        else:
            pad = 0 if self.pad is False else self.loan.term_payments
        for i in range(i + 1, pad + 1):
            payments.append((i, "", "", "", "", "", "", "", "", "", ""))
        return tabulate.tabulate(payments, headers=headers, stralign="right")


class LoanSideBySide:
    """Summarize multiple loans and schedules simultaneously."""

    def __init__(self, locale: str, *loans: FixedRateLoan):
        """Summarize multiple loans and schedules simultaneously."""
        self.locale = locale
        self.loans: Tuple[FixedRateLoan, ...] = loans

    def __str__(self) -> str:
        locale = self.locale
        loans = self.loans
        return f"{type(self).__name__}({locale=}, {loans=})"

    def report(self) -> str:
        """Produce the loan summaries."""
        lines: List[Iterable[str]] = []
        summaries = [
            LoanSummary(loan, self.locale).detail().split("\n") for loan in self.loans
        ]
        pad_to = max(loan.schedule.last().i for loan in self.loans)
        columns = [
            LoanScheduleTable(loan, self.locale, pad=pad_to).report().split("\n")
            for loan in self.loans
        ]
        lines.extend(intersperse(" | ", row) for row in zip(*summaries, strict=True))
        lines.extend(intersperse(" | ", row) for row in zip(*columns, strict=True))
        tabulate.PRESERVE_WHITESPACE = True
        table = tabulate.tabulate(lines, colalign=None)
        tabulate.PRESERVE_WHITESPACE = False
        return table


if __name__ == "__main__":
    example = FixedRateLoan(
        amount=usd(300_300),
        start=d(2024, 7, 1),
        payments=15 * 12,
        payments_per_year=12,
        rate=Decimal("0.05875"),
        extra=StaticExtra(usd("57.81")),
    )
    print(LoanSummary(example, "en_US").detail())
    print(LoanScheduleTable(example, "en_US", pad=False).report())
