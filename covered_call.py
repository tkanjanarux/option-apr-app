from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import yfinance as yf

import logging
logger = logging.getLogger(__name__)



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


def _get_underlying_price(ticker: yf.Ticker) -> Optional[float]:
    """Return the latest close price for the underlying symbol."""
    try:
        price = ticker.fast_info["lastPrice"]
        if price:
            return float(price)
    except Exception:
        pass

    try:
        history = ticker.history(period="1d")
        if not history.empty:
            return float(history["Close"].iloc[-1])
    except Exception:
        pass

    return None


def _calculate_premium(row: pd.Series) -> Optional[float]:
    """Use the bid price as the option premium."""
    bid = row.get("bid")
    if bid is None or pd.isna(bid) or bid <= 0:
        return None
    return float(bid)


def _calculate_days_to_expiry(expiry: datetime) -> int:
    today = datetime.now(timezone.utc).date()
    expiry_date = expiry.date()
    return max((expiry_date - today).days, 0)


def fetch_covered_call_quotes(symbol: str, expiry: str) -> List[OptionQuote]:
    """
    Fetch call options for the given symbol and expiry and compute covered call APR.

    :param symbol: Equity ticker symbol (e.g. "AAPL").
    :param expiry: Expiry date string in YYYY-MM-DD format provided by yfinance.
    """
    ticker = yf.Ticker(symbol)
    underlying_price = _get_underlying_price(ticker)
    logger.log('Getting underlying price')
    if underlying_price is None or underlying_price <= 0:
        logger.log('No underlying price')
        return []

    try:
        option_chain = ticker.option_chain(expiry)
        calls_df = option_chain.calls
    except Exception as exc:
        logger.exception("Failed to load option chain for %s %s", symbol, expiry)
        return []

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_to_expiry = _calculate_days_to_expiry(expiry_dt)
    if days_to_expiry == 0:
        logger.log('days_to_expiry == 0')
        return []

    results: List[OptionQuote] = []
    for _, row in calls_df.iterrows():
        premium = _calculate_premium(row)
        if premium is None or premium <= 0:
            continue

        strike = float(row["strike"])
        if strike <= underlying_price:
            continue
        apr = (premium / underlying_price) * (365 / days_to_expiry) * 100
        break_even = underlying_price - premium

        results.append(
            OptionQuote(
                ticker=symbol.upper(),
                option_type="call",
                expiry=expiry_dt,
                strike=strike,
                premium=premium,
                underlying_price=underlying_price,
                days_to_expiry=days_to_expiry,
                apr=apr,
                break_even_price=break_even,
                bid=row.get("bid"),
                ask=row.get("ask"),
                implied_vol=row.get("impliedVolatility"),
            )
        )

    return sorted(results, key=lambda quote: quote.apr, reverse=True)


def list_option_expiries(symbol: str) -> List[str]:
    """Return available expiry dates for the given symbol."""
    ticker = yf.Ticker(symbol)
    try:
        return list(ticker.options)
    except Exception:
        return []


def fetch_cash_secured_put_quotes(symbol: str, expiry: str) -> List[OptionQuote]:
    """
    Fetch put options for the given symbol and expiry and compute cash-secured put APR.

    APR is calculated as (premium / strike) annualized.
    """
    ticker = yf.Ticker(symbol)
    underlying_price = _get_underlying_price(ticker)
    if underlying_price is None or underlying_price <= 0:
        return []

    try:
        option_chain = ticker.option_chain(expiry)
        puts_df = option_chain.puts
    except Exception:
        return []

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_to_expiry = _calculate_days_to_expiry(expiry_dt)
    if days_to_expiry == 0:
        return []

    results: List[OptionQuote] = []
    for _, row in puts_df.iterrows():
        premium = _calculate_premium(row)
        if premium is None or premium <= 0:
            continue

        strike = float(row["strike"])
        if strike >= underlying_price:
            continue

        if strike <= 0:
            continue

        apr = (premium / strike) * (365 / days_to_expiry) * 100
        break_even = strike - premium

        results.append(
            OptionQuote(
                ticker=symbol.upper(),
                option_type="put",
                expiry=expiry_dt,
                strike=strike,
                premium=premium,
                underlying_price=underlying_price,
                days_to_expiry=days_to_expiry,
                apr=apr,
                break_even_price=break_even,
                bid=row.get("bid"),
                ask=row.get("ask"),
                implied_vol=row.get("impliedVolatility"),
            )
        )

    return sorted(results, key=lambda quote: quote.apr, reverse=True)
