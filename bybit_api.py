from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from pybit.unified_trading import HTTP

import debug_log
from shared_types import OptionQuote


class BybitAPIError(RuntimeError):
    """Generic error raised when Bybit public endpoints fail."""


class BybitAPIForbidden(BybitAPIError):
    """Raised when Bybit rejects the IP for compliance or rate limits."""


def get_bybit_client() -> HTTP:
    """Return a public Bybit HTTP client for option endpoints."""
    return HTTP(testnet=False)


def _raise_api_error(action: str, message: str, *, exc: Optional[Exception] = None) -> None:
    debug_log.log(f"Bybit error during {action}: {message}")
    message_lower = message.lower()
    if "403" in message or "blocked" in message_lower:
        raise BybitAPIForbidden(
            f"{action} failed: {message}. Bybit may be blocking this IP."
        ) from exc
    raise BybitAPIError(f"{action} failed: {message}") from exc


def _call_api(action: str, request_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        response = request_fn()
    except Exception as exc:  # pragma: no cover - network failures
        _raise_api_error(action, str(exc), exc=exc)

    if not (response and response.get("retCode") == 0):
        message = response.get("retMsg") if response else "Unknown error"
        _raise_api_error(action, message)

    return response.get("result", {})


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_underlying_price(client: HTTP, symbol: str) -> Optional[float]:
    spot_symbol = f"{symbol.upper()}USDT"
    result = _call_api(
        f"fetch underlying price for {spot_symbol}",
        lambda: client.get_tickers(category="spot", symbol=spot_symbol),
    )

    ticker_list = result.get("list", [])
    if ticker_list:
        price = _to_float(ticker_list[0].get("lastPrice"))
        if price is not None:
            debug_log.log(f"Bybit underlying price for {spot_symbol}: {price}")
            return price

    debug_log.log(f"Bybit underlying response missing data for {spot_symbol}: {result}")
    return None


def _normalize_premium(premium_quote: Optional[float]) -> Optional[float]:
    """Premiums from Bybit are already quoted in USD terms."""
    return premium_quote


@lru_cache(maxsize=8)
def _get_all_instruments(base_coin: str) -> List[Dict[str, Any]]:
    """Return the full instrument list for a base coin (cached)."""
    client = get_bybit_client()
    instruments: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        result = _call_api(
            f"list instruments for {base_coin}",
            lambda cursor=cursor: client.get_instruments_info(
                category="option", baseCoin=base_coin, cursor=cursor
            ),
        )
        instruments.extend(result.get("list", []))
        cursor = result.get("nextPageCursor")
        if not cursor:
            break

    debug_log.log(f"Bybit instruments for {base_coin}: {len(instruments)} rows")
    return instruments


def _get_option_tickers(client: HTTP, base_coin: str) -> Dict[str, Dict[str, Any]]:
    """Return the latest quotes for every option under the base coin."""
    result = _call_api(
        f"fetch option tickers for {base_coin}",
        lambda: client.get_tickers(category="option", baseCoin=base_coin),
    )

    ticker_list = result.get("list", [])
    target_symbol = "BTC-10NOV25-102000-P-USDT"
    for item in ticker_list:
        if item.get("symbol") == target_symbol:
            debug_log.log(f"Bybit get_tickers snapshot for {target_symbol}: {item}")
            break
    return {item["symbol"]: item for item in ticker_list}


def _get_ticker_for_symbol(client: HTTP, option_symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch a single option ticker snapshot when bulk call misses a symbol."""
    result = _call_api(
        f"fetch ticker for {option_symbol}",
        lambda: client.get_tickers(category="option", symbol=option_symbol),
    )
    ticker_list = result.get("list", [])
    return ticker_list[0] if ticker_list else None


def _parse_option_symbol(option_symbol: str) -> Optional[Tuple[str, float, str]]:
    parts = option_symbol.split("-")
    if len(parts) < 4:
        return None

    expiry = parts[1].upper()
    strike = _to_float(parts[2])
    if strike is None:
        return None

    if parts[-1].upper() in {"C", "P"}:
        option_flag = parts[-1].upper()
    elif len(parts) >= 5 and parts[-2].upper() in {"C", "P"}:
        option_flag = parts[-2].upper()
    else:
        return None

    option_type = "Call" if option_flag == "C" else "Put"
    return expiry, strike, option_type


def list_option_expiries(symbol: str) -> List[str]:
    base_coin = symbol.upper()
    instruments = _get_all_instruments(base_coin)
    expiries = set()
    for instrument in instruments:
        parsed = _parse_option_symbol(instrument.get("symbol", ""))
        if parsed:
            expiries.add(parsed[0])
    return sorted(expiries)


def fetch_covered_call_quotes(symbol: str, expiry: str) -> List[OptionQuote]:
    return _fetch_bybit_quotes(symbol, expiry, "Call")


def fetch_cash_secured_put_quotes(symbol: str, expiry: str) -> List[OptionQuote]:
    return _fetch_bybit_quotes(symbol, expiry, "Put")


def _fetch_bybit_quotes(symbol: str, expiry: str, option_type: str) -> List[OptionQuote]:
    base_coin = symbol.upper()
    client = get_bybit_client()
    underlying_price = _get_underlying_price(client, base_coin)
    if underlying_price is None or underlying_price <= 0:
        debug_log.log(f"Could not determine underlying price for {symbol}.")
        return []

    instrument_list = _get_all_instruments(base_coin)
    if not instrument_list:
        debug_log.log(f"No instruments returned for {symbol.upper()} when fetching {expiry}.")
        return []

    option_tickers = _get_option_tickers(client, base_coin)

    quotes: List[OptionQuote] = []
    debug_log.log(
        f"Bybit fetch {symbol.upper()} {option_type} instruments for {expiry}: {len(instrument_list)} rows"
    )
    expiry_dt = datetime.strptime(expiry, "%d%b%y").replace(tzinfo=timezone.utc)
    days_to_expiry = (expiry_dt.date() - datetime.now(timezone.utc).date()).days
    if days_to_expiry <= 0:
        return []

    skip_reasons: Counter[str] = Counter()

    processed = 0
    for instrument in instrument_list:
        option_symbol = instrument.get("symbol", "")
        match = _parse_option_symbol(option_symbol)
        strike: Optional[float] = None
        matched = False

        if match:
            instrument_expiry, parsed_strike, parsed_option_type = match
            if parsed_option_type == option_type and instrument_expiry == expiry.upper():
                strike = parsed_strike
                matched = True
        else:
            skip_reasons["unparseable_symbol"] += 1

        if not matched:
            delivery_raw = (
                instrument.get("deliveryDate")
                or instrument.get("deliveryTime")
                or instrument.get("expireDate")
                or instrument.get("expDate")
            )
            delivery_value = str(delivery_raw).upper() if delivery_raw else ""
            if delivery_value == expiry.upper() and instrument.get("optionsType") == option_type:
                strike = _to_float(instrument.get("strike"))
                matched = strike is not None

        if not matched or strike is None:
            continue

        processed += 1

        ticker_snapshot = option_tickers.get(option_symbol)
        if not ticker_snapshot:
            ticker_snapshot = _get_ticker_for_symbol(client, option_symbol)
        if not ticker_snapshot:
            skip_reasons["missing_ticker"] += 1
            debug_log.log(f"Missing ticker snapshot for {option_symbol}")
            continue

        if strike <= 0:
            skip_reasons["invalid_strike"] += 1
            continue

        premium_quote = _to_float(
            ticker_snapshot.get("bid1Price")
            or ticker_snapshot.get("lastPrice")
            or ticker_snapshot.get("markPrice")
        )
        premium = _normalize_premium(premium_quote)
        if premium is None or premium <= 0:
            skip_reasons["no_premium"] += 1
            continue

        if option_type == "Call":
            if strike <= underlying_price:
                skip_reasons["itm_call"] += 1
                continue
            apr = (premium / underlying_price) * (365 / days_to_expiry) * 100
            break_even = underlying_price - premium
        else:
            if strike >= underlying_price:
                skip_reasons["otm_put"] += 1
                continue
            apr = (premium / strike) * (365 / days_to_expiry) * 100
            break_even = strike - premium

        quotes.append(
            OptionQuote(
                ticker=base_coin,
                option_type=option_type.lower(),
                expiry=expiry_dt,
                strike=strike,
                premium=premium,
                underlying_price=underlying_price,
                days_to_expiry=days_to_expiry,
                apr=apr,
                break_even_price=break_even,
                bid=_to_float(ticker_snapshot.get("bid1Price")),
                ask=_to_float(ticker_snapshot.get("ask1Price")),
                implied_vol=_to_float(ticker_snapshot.get("impliedVolatility")),
            )
        )

    debug_log.log(
        f"Bybit {symbol.upper()} {option_type} {expiry}: "
        f"processed={processed}, quotes={len(quotes)}, skips={dict(skip_reasons)}"
    )

    return sorted(quotes, key=lambda q: q.apr, reverse=True)
