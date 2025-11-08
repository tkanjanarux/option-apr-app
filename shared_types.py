from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class OptionQuote:
    ticker: str
    option_type: str
    expiry: datetime
    strike: float
    premium: float
    underlying_price: float
    days_to_expiry: int
    apr: float
    break_even_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    implied_vol: Optional[float] = None
