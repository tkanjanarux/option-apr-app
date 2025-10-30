from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

StatsDict = Dict[str, Any]
_FETCH_STATS: Dict[Tuple[str, str, str], StatsDict] = {}


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
    """Use the bid price as the option premium."""
    bid = row.get("bid")
    if bid is not None and not pd.isna(bid) and bid > 0:
        return float(bid)

    last_price = row.get("lastPrice")
    if last_price is not None and not pd.isna(last_price) and last_price > 0:
        return float(last_price)

    ask = row.get("ask")
    if ask is not None and not pd.isna(ask) and ask > 0:
        return float(ask)

    return None


def _calculate_days_to_expiry(expiry: datetime) -> int:
    today = datetime.now(timezone.utc).date()
    expiry_date = expiry.date()
    return max((expiry_date - today).days, 0)


def get_fetch_stats(symbol: str, expiry: str, option_type: str) -> StatsDict:
    """Return the last recorded fetch diagnostics for the given symbol/expiry."""
    key = (option_type.lower(), symbol.upper(), expiry)
    stats = _FETCH_STATS.get(key, {})
    return dict(stats)


def _init_stats(option_type: str, symbol: str, expiry: str) -> Tuple[Tuple[str, str, str], StatsDict]:
    key = (option_type, symbol.upper(), expiry)
    stats: StatsDict = {
        "total_rows": 0,
        "kept": 0,
        "missing_bid": 0,
        "not_otm": 0,
        "invalid_strike": 0,
        "error": "",
    }
    _FETCH_STATS[key] = stats
    return key, stats


def fetch_covered_call_quotes(symbol: str, expiry: str) -> List[OptionQuote]:
    """
    Fetch call options for the given symbol and expiry and compute covered call APR.
    """
    key, stats = _init_stats("call", symbol, expiry)

    ticker = yf.Ticker(symbol)
    underlying_price = _get_underlying_price(ticker)
    if underlying_price is None or underlying_price <= 0:
        stats["error"] = "missing_underlying_price"
        logger.info("No underlying price for %s", symbol)
        return []

    try:
        option_chain = ticker.option_chain(expiry)
        calls_df = option_chain.calls
    except Exception as exc:
        stats["error"] = "option_chain_error"
        logger.exception("Failed to load call option chain for %s %s", symbol, expiry)
        return []

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_to_expiry = _calculate_days_to_expiry(expiry_dt)
    if days_to_expiry == 0:
        stats["error"] = "non_positive_days_to_expiry"
        logger.info("Covered call %s %s has non-positive days_to_expiry", symbol, expiry)
        return []

    stats["total_rows"] = len(calls_df)
    results: List[OptionQuote] = []
    for _, row in calls_df.iterrows():
        premium = _calculate_premium(row)
        if premium is None or premium <= 0:
            stats["missing_bid"] += 1
            continue

        strike = float(row["strike"])
        if strike <= underlying_price:
            stats["not_otm"] += 1
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

    stats["kept"] = len(results)
    logger.info(
        "Covered calls %s %s -> total=%s kept=%s missing_bid=%s not_otm=%s",
        symbol,
        expiry,
        stats["total_rows"],
        stats["kept"],
        stats["missing_bid"],
        stats["not_otm"],
    )
    _FETCH_STATS[key] = stats
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
    key, stats = _init_stats("put", symbol, expiry)

    ticker = yf.Ticker(symbol)
    underlying_price = _get_underlying_price(ticker)
    if underlying_price is None or underlying_price <= 0:
        stats["error"] = "missing_underlying_price"
        logger.info("No underlying price for %s", symbol)
        return []

    try:
        option_chain = ticker.option_chain(expiry)
        puts_df = option_chain.puts
    except Exception:
        stats["error"] = "option_chain_error"
        logger.exception("Failed to load put option chain for %s %s", symbol, expiry)
        return []

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_to_expiry = _calculate_days_to_expiry(expiry_dt)
    if days_to_expiry == 0:
        stats["error"] = "non_positive_days_to_expiry"
        logger.info("Cash-secured put %s %s has non-positive days_to_expiry", symbol, expiry)
        return []

    stats["total_rows"] = len(puts_df)
    results: List[OptionQuote] = []
    for _, row in puts_df.iterrows():
        premium = _calculate_premium(row)
        if premium is None or premium <= 0:
            stats["missing_bid"] += 1
            continue

        strike = float(row["strike"])
        if strike >= underlying_price:
            stats["not_otm"] += 1
            continue

        if strike <= 0:
            stats["invalid_strike"] += 1
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

    stats["kept"] = len(results)
    logger.info(
        "Cash-secured puts %s %s -> total=%s kept=%s missing_bid=%s not_otm=%s invalid_strike=%s",
        symbol,
        expiry,
        stats["total_rows"],
        stats["kept"],
        stats["missing_bid"],
        stats["not_otm"],
        stats["invalid_strike"],
    )
    _FETCH_STATS[key] = stats
    return sorted(results, key=lambda quote: quote.apr, reverse=True)
