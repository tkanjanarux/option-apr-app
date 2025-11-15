from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import yfinance as yf

from shared_types import OptionQuote


def _get_underlying_price(ticker: yf.Ticker) -> Optional[float]:
    """Return the latest available spot price for the ticker."""
    try:
        price = ticker.fast_info.get("lastPrice")
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
    """Return the best available premium for a row in the chain."""
    for field in ("bid", "lastPrice", "ask"):
        value = row.get(field)
        if value is not None and not pd.isna(value) and value > 0:
            return float(value)
    return None


def _calculate_days_to_expiry(expiry: datetime) -> int:
    today = datetime.now(timezone.utc).date()
    expiry_date = expiry.date()
    return max((expiry_date - today).days, 0)


def _load_option_frame(
    ticker: yf.Ticker, symbol: str, expiry: str, option_type: str
) -> Optional[pd.DataFrame]:
    try:
        chain = ticker.option_chain(expiry)
        return chain.calls if option_type == "call" else chain.puts
    except Exception:
        print(f"Failed to load {option_type} option chain for {symbol} {expiry}")
        return None


def fetch_covered_call_quotes(symbol: str, expiry: str) -> List[OptionQuote]:
    ticker = yf.Ticker(symbol)
    underlying_price = _get_underlying_price(ticker)
    if underlying_price is None or underlying_price <= 0:
        print(f"No underlying price for {symbol}")
        return []

    calls_df = _load_option_frame(ticker, symbol, expiry, "call")
    if calls_df is None or calls_df.empty:
        return []

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_to_expiry = _calculate_days_to_expiry(expiry_dt)
    if days_to_expiry == 0:
        print(f"Covered call {symbol} {expiry} has non-positive days_to_expiry")
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
    ticker = yf.Ticker(symbol)
    try:
        return list(ticker.options)
    except Exception:
        return []


def fetch_cash_secured_put_quotes(symbol: str, expiry: str) -> List[OptionQuote]:
    ticker = yf.Ticker(symbol)
    underlying_price = _get_underlying_price(ticker)
    if underlying_price is None or underlying_price <= 0:
        print(f"No underlying price for {symbol}")
        return []

    puts_df = _load_option_frame(ticker, symbol, expiry, "put")
    if puts_df is None or puts_df.empty:
        return []

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_to_expiry = _calculate_days_to_expiry(expiry_dt)
    if days_to_expiry == 0:
        print(f"Cash-secured put {symbol} {expiry} has non-positive days_to_expiry")
        return []

    results: List[OptionQuote] = []
    for _, row in puts_df.iterrows():
        premium = _calculate_premium(row)
        if premium is None or premium <= 0:
            continue

        strike = float(row["strike"])
        if strike >= underlying_price or strike <= 0:
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
