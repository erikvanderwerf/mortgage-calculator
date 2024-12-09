"""Mortgage Comparison."""

#  Copyright (c) 2024 Erik VanderWerf

from __future__ import annotations

__all__ = [
    "CostSummary",
    "CheaperWhen",
    "FixedAnalyzer",
    "FixedRateMortgageTemplate",
    "MinimizeMortgageCostReportGenerator",
]

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime as d
from decimal import Decimal
from functools import partial
from typing import Dict, List, Self, Set, Tuple, cast, Iterable

import inflect
from babel.numbers import format_percent
from moneyed import Money, format_money
from more_itertools import intersperse
from tabulate import tabulate

from mortgage.common import b, p, pct, q, usd
from mortgage.extra import Extra, StaticExtra, VariableExtra
from mortgage.loan import FixedRateLoan, LoanSideBySide
from mortgage.mortgage import FixedRateMortgage


class CheaperWhen:
    """Summarizes at what point a mortgage costs more than another."""

    def __init__(self, original: FixedRateMortgage, other: FixedRateMortgage):
        """Summarizes at what point a mortgage costs more than another."""
        self.original = original
        self.other = other

    def __str__(self) -> str:
        break_even: str | int = self.original.is_costlier_than(self.other)
        if break_even == "never":
            break_even = "always"
        elif isinstance(break_even, int):
            break_even = f"< {break_even}"
        return break_even


@dataclass
class FixedRateMortgageTemplate:
    """Template for generating a fixed rate mortgage."""

    name: str
    home_price: Money
    down_payment: Money | Decimal
    origination: Money
    seller_concessions: Money
    start: d
    term_years: int
    rate: Decimal
    extra: Extra

    def realize(self, add_extra: Extra) -> FixedRateMortgage:
        """Build a mortgage with additional extras."""
        return FixedRateMortgage(
            name=self.name,
            home_price=self.home_price,
            down_payment=self.down_payment,
            origination=self.origination,
            seller_concessions=self.seller_concessions,
            start=self.start,
            term_years=self.term_years,
            rate=self.rate,
            extra=self.extra + add_extra,
        )


@dataclass(frozen=True, order=True)
class CostSummary:
    """Summarize the cost of a mortgage at a given payment."""

    cost: Money = field(compare=True)
    payment: int = field(compare=True)
    description: str = field(compare=False)
    original: FixedRateMortgage = field(compare=False)
    mortgage: FixedRateMortgage = field(compare=False)

    def __str__(self) -> str:
        return f"{self.mortgage.name} ({self.description})"


class MinimizeMortgageCostReportGenerator:
    """Report comparing multiple mortgages."""

    lowest_cost: FixedRateMortgage

    def __init__(
        self,
        templates: List[FixedRateMortgageTemplate],
        locale: str,
        skip_b: bool | None = None,
        pick: int = 5,
        max_recommendations_per_loan: int = 1,
    ):
        """Report comparing multiple mortgages."""
        if len(templates) < 2:
            raise ValueError("Nothing to compare against.")
        if 1 != len({t.term_years for t in templates}):
            raise ValueError("Templates have different number of term years!")

        self.templates: List[FixedRateMortgageTemplate] = templates
        self.locale: str = locale
        self.pick: int = pick
        self.max_recommendations_per_loan: int = max_recommendations_per_loan

        self.no_additional_payments = [
            mt.realize(StaticExtra(usd(0))) for mt in self.templates
        ]
        self.sorted_by_cost = sorted(
            self.no_additional_payments, key=lambda x: x.cost()
        )
        self.lowest_cost, *not_lowest_cost = self.sorted_by_cost
        self.lower_origination: List[FixedRateMortgage] = list(
            filter(
                lambda nl: (nl.origination < self.lowest_cost.origination),
                not_lowest_cost,
            )
        )
        self.worse: List[FixedRateMortgage] = list(
            filter(lambda x: x not in self.lower_origination, not_lowest_cost)
        )

        self.reduced: Dict[FixedRateMortgage, FixedRateMortgage] = {}
        self.accelerated: Dict[FixedRateMortgage, Dict[int, FixedRateMortgage]] = {}
        self.drop: Dict[FixedRateMortgage, FixedRateMortgage] = {}
        for lo in self.lower_origination:
            reinvest_origination: Money = self.lowest_cost.origination - lo.origination
            # Outcome A - Increase Down Payment and Reinvest Lower Payment Difference
            lower_amount: Money = lo.loan.amount - reinvest_origination
            new_payment: Money = FixedRateLoan.calculate_payment_from_annual_rate(
                lower_amount,
                lo.loan.rate,
                lo.loan.payments_per_year,
                lo.loan.term_payments,
            )
            payment_difference: Money = lo.loan.scheduled_payment - new_payment
            # Trial A - Apply Difference to Down Payment
            self.reduced[lo] = lo.recalculate(
                adjust_down_payment=reinvest_origination,
                adjust_extra=StaticExtra(payment_difference),
            )
            # Trial B - Apply Difference to Principal on First Payment
            self.drop[lo] = lo.recalculate(
                adjust_extra=VariableExtra({1: reinvest_origination})
            )
            # Trial C - Amortized Additional Principal Payments
            self.accelerated[lo] = lo_spread = {}
            for life in range(1, self.lowest_cost.loan.term_payments + 1):
                spread: Money = reinvest_origination / life
                extra = VariableExtra({i: spread for i in range(1, life+1)})
                lo_spread[life] = lo.recalculate(adjust_extra=extra)

        # Evaluate whether to skip Trial B results.
        if skip_b is None:
            for m in self.reduced:
                reduced, drop = self.reduced[m], self.drop[m]
                skip_b = not any(
                    reduced.cost(pmt) > drop.cost(pmt)
                    for pmt in range(1, self.lowest_cost.loan.term_payments + 1)
                )
        self.skip_b: bool = cast(bool, skip_b)

    def __str__(self):
        summaries = []
        for m in self.no_additional_payments:
            summaries.append(m.name)
        summaries = " / ".join(summaries)
        return f"{type(self).__name__}({summaries})"

    def cheapest_at_payment(self, payment: int) -> List:
        """Return a list of the cheapest mortgage options for a given payment."""
        # Originals
        options: List[CostSummary] = []
        for original in self.sorted_by_cost:
            options.append(
                CostSummary(
                    cost=original.cost(payment),
                    payment=payment,
                    description="/",
                    original=original,
                    mortgage=original,
                )
            )
        # Outcome A - Reduced Loan Amount & Outcome B - Drop Difference
        simple_summaries = [("A", self.reduced.items())]
        if not self.skip_b:
            simple_summaries.append(("B", self.drop.items()))
        for description, items in simple_summaries:
            for original, reduced in items:
                options.append(
                    CostSummary(
                        cost=reduced.cost(payment),
                        payment=payment,
                        description=description,
                        original=original,
                        mortgage=reduced,
                    )
                )
        # Outcome C - Amortize Difference
        for original, available in self.accelerated.items():
            accelerated = available[payment]
            options.append(
                CostSummary(
                    cost=accelerated.cost(payment),
                    payment=payment,
                    description="C",
                    original=original,
                    mortgage=accelerated,
                )
            )
        # noinspection PyTypeChecker
        return sorted(options)

    def lookup_accelerated(self, name: str, payment: int) -> FixedRateMortgage:
        """Return an accelerated mortgage by name and payment."""
        for key, value in self.accelerated.items():
            if key.name == name:
                return value[payment]
        else:
            raise KeyError(name)

    def lookup_drop(self, name: str) -> FixedRateMortgage:
        """Return a mortgage with a bulk extra payment by name."""
        for key, value in self.drop.items():
            if key.name == name:
                return value
        else:
            raise KeyError(name)

    def lookup_original(self, name: str) -> FixedRateMortgage:
        """Return the original mortgage by name."""
        for value in self.sorted_by_cost:
            if value.name == name:
                return value
        else:
            raise KeyError(name)

    def lookup_reduced(self, name: str) -> FixedRateMortgage:
        """Lookup the down payment reduced mortgage by name."""
        for key, value in self.reduced.items():
            if key.name == name:
                return value
        else:
            raise KeyError(name)

    def report(
        self, months_of_interest: List[int] | None = None, quiet: bool = False
    ) -> str:
        """Generate a human-facing report to choose the best mortgage."""
        inf = inflect.engine()
        # Process Inputs
        if months_of_interest is None:
            months_of_interest = []
        working: Set[int] = set(months_of_interest)
        working.add(self.lowest_cost.loan.term_payments)
        at_months: List[Tuple[int, str]] = sorted(
            (i, f"@{i} {inf.plural_noun("month", i)}") for i in working
        )
        del working
        # Begin Report
        report = p(
            'This report aims to suggest the loan with the lowest "cost", determined as'
            " the summation of origination fees and interest paid over the time the"
            " loan is held."
            ' Using this definition, the loan only "costs" the amount of money paid to'
            " the lender which is not retained as equity in the property."
        )
        report += p(
            "The loan options below are sorted by their total cost if the loan is seen"
            " to term:"
        )
        headers = [
            "Name",
            "Amount",
            "Rate",
            "Payment",
            "Origination",
            "Total Interest",
        ] + [a[1] for a in at_months]
        options = []
        m: FixedRateMortgage
        for m in self.sorted_by_cost:
            amount = format_money(m.loan.amount, locale=self.locale)
            rate = format_percent(m.loan.rate, format="#.###%", locale=self.locale)
            origination = format_money(m.origination, locale=self.locale)
            new_payment: str = format_money(
                m.loan.scheduled_payment, locale=self.locale
            )
            total_interest = format_money(
                m.loan.schedule.last().cumulative_interest, locale=self.locale
            )
            options.append(
                [m.name, amount, rate, new_payment, origination, total_interest]
                + [format_money(m.cost(a[0]), locale=self.locale) for a in at_months]
            )
        report += tabulate(options, headers=headers, stralign="right") + "\n\n"
        lcn = q(self.lowest_cost.name)
        if not quiet:
            inf.num(len(self.worse))
            eliminated = (
                f"{inf.join([q(w.name) for w in self.worse])} {inf.plural_verb("has")}"
                f" been eliminated because"
                f" {inf.plural_noun("it")} {inf.plural_verb("has")}"
                f" a higher origination cost with no beneficial tradeoff."
            )
            inf.num(len(at_months))
            report += p(
                f"The @-month {inf.plural_noun("column")} represent the sunk cost of"
                f" the loan by that payment. This includes the origination fees and all"
                f" cumulative interest by that payment."
                f" {b(eliminated, self.worse)}"
            )
            lower_origination: List[FixedRateMortgage] = self.lower_origination
            inf.num(len(lower_origination))
            report += p(
                f"{lcn} is the benchmark to beat on total cost."
                f" This analysis cannot change the loan rate because those terms are"
                f" set by the lender, but you are encouraged to ask for loan estimates"
                f" from your lenders to \"buy-down\" the rate. This would result in a"
                f" higher origination (upfront) cost, but a lower interest rate. Run"
                f" this report again to find the tipping point where such a buy-down"
                f" would be worth the cost over time."
            )
            report += p(
                f"{inf.join([e.name for e in lower_origination])}"
                f" {inf.plural_verb("has")}{b(" a", len(lower_origination) == 1)}"
                f" lower origination {inf.plural_noun("fee")}, of which the"
                f" savings compared to {lcn} can be applied in a few different trials:"
                f" A) a larger down payment,"
                f" B) a larger first principal payment,"
                f" or C) additional principal payments spread over the course of the"
                f" expected loan lifetime."
            )
        report += (
            "Trial A: Reduce the Loan Amount by the Origination Difference and Reinvest"
            " Payment Savings\n"
        )
        report += self.tabulate_simple_loans(at_months, self.reduced) + "\n\n"
        if not quiet:
            report += p(
                f"By investing the difference in origination costs back into the down"
                f" payment, the cost of the loan is reduced two-fold."
                f" First by the reduced loan amount, and then again by the compound"
                f" reduced monthly interest payments from the reduced loan amount."
                f" The difference between the old and new monthly payments can be"
                f" reinvested as additional payments against the principal for the life"
                f" of the loan."
                f" The \"Cheaper Before\" column notes that doing this is cheaper than"
                f" {lcn} for only the first X-payments (months)."
                f" If the loan is held for longer, then {lcn} will be cheaper"
                f" overall due to smaller interest payments, despite the higher"
                f" origination costs."
                f" Correspondingly, if the loan is held for less time, the smaller"
                f" origination cost will have been worth it because there were fewer"
                f" interest payments, despite the higher amount."
            )
            report += p(
                f"You might not hold any of these loans \"to term\" for the full"
                f" {
                    inf.join(
                        [str(z) for z in sorted(set(x.term_years for x in self.templates))],
                        conj="or"
                    )
                } years. Lenders often allow you to refinance at any time, but will"
                " charge a new set of origination fees at the time."
            )
        if self.skip_b:
            if not quiet:
                report += p(
                    "Hiding Trial B results because there were all more expensive than"
                    " trial A."
                )
        else:
            report += "Trial B: Make a Large First Principal Payment\n"
            report += self.tabulate_simple_loans(at_months, self.drop) + "\n\n"
            if not quiet:
                report += p(
                    "Instead of applying the difference as a larger down payment, the"
                    " difference can be applied to a large principal payment on the"
                    " first payment."
                    " Intuitively one may think that this would help by producing a"
                    " higher monthly scheduled payment against with the same loan"
                    " principal after the first payment."
                    " However, the higher interest payment on the first month"
                    " snowballs over time and towards the end of the loan term leads"
                    " to material increases in the cost of the loan."
                    ' Even if compensated with a little extra "help", that difference'
                    " is still sunk, meaning that it is better to simply apply any"
                    " savings directly to the down payment. In other words, Trial A is"
                    " always more efficient."
                )
        report += (
            "Trial C: Spread Lower Origination Costs into Additional Principal"
            " Payments Over Select Periods\n"
        )
        headers = (
            ["Name", "Difference"]
            + [f"Addt'l. / Monthly\n{a[1]} (cheaper)" for a in at_months]
            + ["Amortized\nCheaper", "Pymts"]
        )
        trial_c: List[Iterable[str | int]] = []
        for m in [self.lowest_cost] + self.lower_origination:
            difference = self.lowest_cost.origination - m.origination
            row: List[str | int] = [
                m.name,
                format_money(difference, locale=self.locale),
            ]
            for a, _ in at_months:
                additional: Money = difference / a
                recalc = (
                    self.lowest_cost
                    if m is self.lowest_cost
                    else self.accelerated[m][a]
                )
                cost = recalc.cost(a)
                new_payment = format_money(
                    recalc.loan.scheduled_payment + additional, locale=self.locale
                )
                row.append(
                    f"+{format_money(additional, locale=self.locale)}"
                    f" / {new_payment}"
                    f"\n{format_money(cost, locale=self.locale)}"
                    f" ({CheaperWhen(recalc, self.lowest_cost)})"
                )
            break_even: str | None = None
            pay_down: int | None = None
            if m == self.lowest_cost:
                break_even = "same"
                pay_down = m.loan.term_payments
            else:
                life = 1
                while life <= m.loan.term_payments and any(
                    x is None for x in {break_even, pay_down}
                ):
                    accelerated: FixedRateMortgage = self.accelerated[m][life]
                    if break_even is None:
                        accelerated_cost = accelerated.cost(life)
                        lowest_cost_at_lifetime = self.lowest_cost.cost(life)
                        if accelerated_cost > lowest_cost_at_lifetime:
                            break_even = f"< {life}"
                    if pay_down is None:
                        if life == accelerated.loan.schedule.last().i:
                            pay_down = life
                    life += 1
                else:
                    if break_even is None:
                        break_even = "never"
                    if pay_down is None:
                        pay_down = m.loan.term_payments
            row.extend([break_even, pay_down])
            trial_c.append(row)
        report += tabulate(trial_c, headers=headers, stralign="right") + "\n\n"
        if not quiet:
            report += p(
                f"For a given expected loan lifetime, the difference in origination"
                f" cost can be smeared equally over each payment as an additional"
                f" principal contribution."
                f" Loans in Trial C will often perform worse than Trial A over the"
                f" long-term, i.e. they become more expensive than {lcn} faster -"
                f" because the smeared principal payments on the initially higher loan"
                f" amount are always playing catch-up with the reduced loans from Trial"
                f" A."
            )
            inf.num(len(at_months))
            report += p(
                f"The @-month {inf.plural("column")} {inf.plural_verb("assumes")} the"
                f" loan is held for approximately the indicated amount of time, in"
                f" order to apply the proper amortization of the origination savings."
                f" A shorter loan lifetime means the difference in origination fees can"
                f" be applied more aggressively."
            )
        report += "Summary: Choosing a Loan?\n\n"
        if not quiet:
            origination = format_money(self.lowest_cost.origination, locale=self.locale)
            report += p(
                f"If you have {origination} available in addition to your down payment"
                f" to cover the loan origination, and you expect to"
                f" hold the loan for X number of years, here are the cheapest"
                f" loan choices:"
            )
        start_year = 1
        pick = self.pick
        spacing: List[Tuple[int, int]] = [(0, 0)] * pick
        table_fmt = []
        for payment in range(
            start_year * 12, self.lowest_cost.loan.term_payments + 1, 12
        ):

            def cap_per_original(
                count: Counter[FixedRateMortgage], cs: CostSummary
            ) -> bool:
                if count[cs.original] >= self.max_recommendations_per_loan:
                    return False
                count[cs.original] += 1
                return True

            seen: Counter[FixedRateMortgage] = Counter()
            top: List[CostSummary] = list(
                filter(
                    lambda cs: cap_per_original(seen, cs),
                    self.cheapest_at_payment(payment),
                )
            )[:pick]
            row_fmt: List[Tuple[str, str]] = []
            for i, summary in enumerate(top):
                fmt = (str(summary), format_money(summary.cost, locale=self.locale))
                row_fmt.append(fmt)
                spacing[i] = (
                    max(spacing[i][0], len(fmt[0])),
                    max(spacing[i][1], len(fmt[1])),
                )
            table_fmt.append(row_fmt)

        headers = list(
            intersperse(
                "|", ["yr"] + [inf.ordinal(i) for i in range(1, pick + 1)]  # type: ignore
            )
        )
        summary_table: List[List[str]] = [
            [f"{x:>2}", "|"]
            + list(
                intersperse(
                    "|",
                    [
                        f"{column[0]:<{spacing[y][0]}}  {column[1]:>{spacing[y][1]}}"
                        for y, column in enumerate(row)
                    ],
                )
            )
            for x, row in enumerate(table_fmt, start=start_year)
        ]
        report += tabulate(summary_table, headers=headers) + "\n\n"
        legend_b = b("(B) First Principal, ", not self.skip_b)
        report += (
            f"Legend: (/) is the original loan term, (A) Pay More Down,"
            f" {legend_b}(C) Amortize Savings."
        )
        report += (
            "\nRemember that \"cheapest\" in this table means origination plus"
            " interest, not the total amount paid or repaid."
        )
        return report

    def tabulate_simple_loans(
        self,
        at_months: List[Tuple[int, str]],
        lookup: Dict[FixedRateMortgage, FixedRateMortgage],
    ) -> str:
        """Generate a table of mortgages."""
        headers = (
            ["\nName", "\nReduction", "New Loan\nAmount", "Base\nPayment", "Reinvest\nPayment"]
            + ["Cost   \n" + a[1] for a in at_months]
            + ["Cheaper\nBefore", "\nPymts"]
        )
        table = []
        for m in [self.lowest_cost] + self.lower_origination:
            recalc: FixedRateMortgage = (
                self.lowest_cost if m is self.lowest_cost else lookup[m]
            )
            reduction: Money = self.lowest_cost.origination - recalc.origination
            table.append(
                [
                    m.name,
                    format_money(reduction, locale=self.locale),
                    format_money(recalc.loan.amount, locale=self.locale),
                    format_money(recalc.loan.scheduled_payment, locale=self.locale),
                    format_money(
                        m.loan.scheduled_payment - recalc.loan.scheduled_payment,
                        locale=self.locale,
                    ),
                ]
                + [
                    format_money(recalc.cost(a[0]), locale=self.locale)
                    for a in at_months
                ]
                + [CheaperWhen(recalc, self.lowest_cost), recalc.loan.schedule.last().i]
            )
        return tabulate(table, headers=headers, stralign="right")


class FixedAnalyzer:
    """Analyze and compare multiple of fixed rate mortgages."""

    def __init__(self, locale: str):
        """Analyze and compare multiple of fixed rate mortgages."""
        self.locale = locale
        self.templates: List[FixedRateMortgageTemplate] = []

    def add(self, template: FixedRateMortgageTemplate) -> Self:
        """Add a new mortgage to analyze."""
        self.templates.append(template)
        return self

    def minimize_cost(self, pick: int = 5, from_each: int = 1) -> MinimizeMortgageCostReportGenerator:
        """Report on the mortgages with the lowest costs."""
        return MinimizeMortgageCostReportGenerator(
            self.templates,
            self.locale,
            skip_b=None,
            pick=pick,
            max_recommendations_per_loan=from_each,
        )


if __name__ == "__main__":
    FRM = partial(
        FixedRateMortgageTemplate,
        home_price=usd(385_000),
        down_payment=pct("22"),
        seller_concessions=usd(10_000),
        start=d(2024, 7, 1),
        term_years=15,
        extra=StaticExtra(usd(0)),
    )
    comparison: MinimizeMortgageCostReportGenerator = (
        FixedAnalyzer("en_US")
        # Total Origination includes Points, Processing Fees, Origination Fees,
        # Appraisal, Credit Report Fee, Tax Service Fees
        .add(
            FRM("USAA", rate=pct("5"), origination=usd(11_928 + 1295 + 600 + 76 + 85))
        )
        .add(
            FRM("NOVA", rate=pct("5.875"), origination=usd(2_772 + 1250 + 650 + 125))
        )
        .add(
            FRM("Chase", rate=pct("5.75"), origination=usd(5_114 + 1595 + 600 + 63 + 87))
        )
        .add(
            FRM("Zillow", rate=pct("5.25"), origination=usd(12_579 + 1500 + 600 + 73))
        )
        .add(
            FRM(
                "Rocket",
                rate=pct("5.875"),
                origination=usd(9760 + 1125 + 375 + 200 + 730 + 60),
            )
        )
        .minimize_cost(pick=5, from_each=1)
    )
    print(comparison.report([i * 12 for i in [3, 4, 5, 6]], quiet=False).strip())
    # print(comparison.report([39, 40, 41, 42]))
    print("\n")
    # to_compare =[c.mortgage for c in comparison.cheapest_at_payment(5 * 12)[:2]]
    # print(LoanSideBySide("en_US",*(c.loan for c in to_compare)).report())
    print(
        LoanSideBySide(
            "en_US",
            *(
                comparison.lookup_original("NOVA").loan,
                # comparison.lookup_original("Zillow-2").loan,
                comparison.lookup_original("Chase").loan,
            ),
        ).report()
    )
    # print(LoanSummary(accelerated, "en_US"))
    # print("\n")
    # print(LoanScheduleTable(accelerated, "en_US"))
