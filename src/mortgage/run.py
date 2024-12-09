#  Copyright (c) 2024 Erik VanderWerf

import json
import sys

from argparse import ArgumentParser
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import List

from moneyed import Money
from pydantic import BaseModel, Field

from mortgage.comparison import FixedAnalyzer, FixedRateMortgageTemplate
from mortgage.extra import StaticExtra


class TemplateMortgage(BaseModel):
    name: str
    rate: Decimal
    origination: str = Field(pattern=r"^\d[ _.+\d]*$")
    extra: Decimal = Decimal(0)
    home_price: Decimal | None = None
    down_payment: Decimal | None = None


class Defaults(BaseModel):
    term_years: int
    extra: Decimal = Decimal(0)
    home_price: Decimal | None = None
    down_payment: Decimal | None = None


class Report(BaseModel):
    pick: int
    quiet: bool
    detail_years: List[int]


class Input(BaseModel):
    currency: str
    locale: str
    start_date: date
    defaults: Defaults
    report: Report
    mortgages: List[TemplateMortgage]


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("mortgages", type=Path, help="mortgage json file.")
    parser.add_argument("--schema", action="store_true", help="output schema.")
    args = parser.parse_args()
    mortgages: Path = args.mortgages
    schema: bool = args.schema

    if schema:
        print(f"Writing schema to schema.json")
        Path("schema.json").write_text(json.dumps(Input.model_json_schema(), indent=2))
        return 0

    i: Input = Input.model_validate_json(mortgages.read_text())
    analyzer = FixedAnalyzer(locale=i.locale)
    for mortgage in i.mortgages:
        price = mortgage.home_price or i.defaults.home_price
        down = mortgage.down_payment or i.defaults.down_payment
        origination = sum(Decimal(x.strip()) for x in mortgage.origination.split("+"))
        rate = mortgage.rate if mortgage.rate < 1 else mortgage.rate / 100
        extra = mortgage.extra or i.defaults.extra
        analyzer.add(
            FixedRateMortgageTemplate(
                name=mortgage.name,
                home_price=Money(price, currency=i.currency),
                down_payment=down,
                origination=Money(origination, currency=i.currency),
                seller_concessions=Money(0, currency=i.currency),
                start=i.start_date,
                term_years=i.defaults.term_years,
                rate=rate,
                extra=StaticExtra(Money(extra, currency=i.currency)),
            )
        )
    comparison = analyzer.minimize_cost(pick=i.report.pick, from_each=1)
    print(comparison.report([i * 12 for i in i.report.detail_years]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
