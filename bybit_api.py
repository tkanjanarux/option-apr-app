from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests

import debug_log
from shared_types import OptionQuote

try:  # Streamlit secrets are only available when running inside Streamlit.
    import streamlit as _st
except Exception:  # pragma: no cover - optional dependency
    _st = None


class BybitAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, ret_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.ret_code = ret_code


class BybitAPIForbidden(BybitAPIError):
    pass


def _load_streamlit_secrets() -> Optional[Any]:
    if not _st:
        return None
    try:
        return _st.secrets
    except Exception as exc:  # pragma: no cover - depends on hosting env
        debug_log.log(f"Streamlit secrets unavailable: {exc}")
        return None


def _get_setting(key: str) -> Optional[str]:
    env_value = os.environ.get(key)
    if env_value:
        return env_value
    secrets = _load_streamlit_secrets()
    if secrets and key in secrets:
        value = secrets[key]
        return str(value) if value is not None else None
    return None


def _get_setting_with_source(key: str) -> Tuple[Optional[str], Optional[str]]:
    env_value = os.environ.get(key)
    if env_value:
        return env_value, "env"
    secrets = _load_streamlit_secrets()
    if secrets and key in secrets:
        value = secrets[key]
        str_value = str(value) if value is not None else None
        return str_value, "secret"
    return None, None


_base_url_value, _base_url_source = _get_setting_with_source("BYBIT_API_BASE_URL")
_base_urls_setting = _get_setting("BYBIT_API_BASE_URLS")
_BASE_URL = _base_url_value or "https://api.bybit.com"
_HTTP_PROXY = _get_setting("BYBIT_HTTP_PROXY")
_USER_AGENT = _get_setting("BYBIT_API_USER_AGENT") or "option-apr-app/1.0"
_REQUEST_TIMEOUT = float(_get_setting("BYBIT_HTTP_TIMEOUT") or 10.0)

_session = requests.Session()
_session.headers.update({"User-Agent": _USER_AGENT})
if _HTTP_PROXY:
    _session.proxies.update({"http": _HTTP_PROXY, "https": _HTTP_PROXY})
    debug_log.log(f"Bybit proxy configured for requests session: {_HTTP_PROXY}")


def _normalize_base(url: str) -> str:
    return url.rstrip("/")


def _build_base_url_list() -> List[str]:
    urls: List[str] = []
    base_url_source = _base_url_source or "default"
    if _base_urls_setting:
        for entry in _base_urls_setting.split(","):
            trimmed = entry.strip()
            if trimmed:
                urls.append(_normalize_base(trimmed))
        debug_log.log("Bybit base URLs sourced from BYBIT_API_BASE_URLS setting")
    elif _base_url_value:
        urls.append(_normalize_base(_base_url_value))
        debug_log.log(f"Bybit base URL sourced from {_base_url_source}")
    else:
        debug_log.log("Bybit base URL falling back to defaults")

    defaults = [
        _normalize_base(_BASE_URL),
        "https://api.bytick.com",
        "https://api.bybitglobal.com",
    ]
    for default_url in defaults:
        if default_url and default_url not in urls:
            urls.append(default_url)

    debug_log.log(f"Bybit API base URL options: {urls}")
    return urls


_BASE_URLS = _build_base_url_list()


def _request(path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    last_error: Optional[BybitAPIError] = None
    for base in _BASE_URLS:
        try:
            return _request_single(base, path, params=params)
        except BybitAPIForbidden as exc:
            last_error = exc
            debug_log.log(f"Bybit request forbidden for base {base}. Trying next option…")
            continue
        except BybitAPIError as exc:
            last_error = exc
            # Retry on server-side issues; otherwise, stop.
            if exc.status_code and exc.status_code >= 500:
                debug_log.log(f"Bybit server error on {base}, retrying alternate base…")
                continue
            break

    if last_error:
        raise last_error
    raise BybitAPIError("Failed to contact Bybit: no base URLs available")


def _request_single(base_url: str, path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{base_url}{path}"
    debug_log.log(f"Bybit GET {path} base={base_url} params={params}")
    try:
        response = _session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
    except requests.RequestException as exc:  # pragma: no cover
        debug_log.log(f"HTTP request to {url} failed: {exc}")
        raise BybitAPIError(f"Failed to reach Bybit base {base_url}: {exc}") from exc

    if response.status_code == 403:
        message = response.text.strip() or response.reason
        debug_log.log(f"Bybit HTTP 403 for {url}: {message}")
        raise BybitAPIForbidden(
            f"HTTP 403 from Bybit: {message}. This often happens when the IP is rate-limited "
            "or originates from a blocked region.",
            status_code=403,
        )

    if response.status_code >= 400:
        debug_log.log(f"Bybit HTTP {response.status_code} for {url}: {response.text}")
        raise BybitAPIError(
            f"Bybit HTTP error {response.status_code} while calling {path}",
            status_code=response.status_code,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        debug_log.log(f"Invalid JSON from Bybit {url}: {response.text[:200]}")
        raise BybitAPIError(f"Received invalid JSON from {path}") from exc

    if payload.get("retCode") != 0:
        message = payload.get("retMsg") or "Unknown error"
        ret_code = payload.get("retCode")
        debug_log.log(f"Bybit retCode {ret_code} for {url}: {message}")
        raise BybitAPIError(
            f"Bybit API returned retCode={ret_code} message={message}", ret_code=ret_code
        )

    return payload.get("result", {})


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_underlying_price(symbol: str) -> Optional[float]:
    spot_symbol = f"{symbol.upper()}USDT"
    result = _request(
        "/v5/market/tickers",
        params={"category": "spot", "symbol": spot_symbol},
    )
    ticker_list = result.get("list", [])
    if ticker_list:
        price = _to_float(ticker_list[0].get("lastPrice"))
        if price is not None:
            debug_log.log(f"Bybit underlying price for {spot_symbol}: {price}")
            return price
    debug_log.log(f"Bybit underlying price missing for {spot_symbol}: {result}")
    return None


def _normalize_premium(premium_quote: Optional[float]) -> Optional[float]:
    """
    Bybit option tickers report premiums in USD terms already (even for BTC-settled contracts),
    so we simply validate/return the quoted value without any additional conversion.
    """
    return premium_quote


@lru_cache(maxsize=8)
def _get_all_instruments_cached(base_coin: str) -> List[Dict[str, Any]]:
    """Fetch all option instruments for a base coin (handles pagination)."""
    instruments: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        params = {"category": "option", "baseCoin": base_coin}
        if cursor:
            params["cursor"] = cursor
        result = _request("/v5/market/instruments-info", params=params)
        instruments.extend(result.get("list", []))
        cursor = result.get("nextPageCursor")
        if not cursor:
            break

    debug_log.log(f"Bybit instruments for {base_coin}: {len(instruments)} rows")
    return instruments


def _get_all_instruments(base_coin: str) -> List[Dict[str, Any]]:
    # Use cached data when available; if cache is empty fall back to live fetch.
    instruments = _get_all_instruments_cached(base_coin)
    if instruments:
        return instruments
    return []


def _get_option_tickers(base_coin: str) -> Dict[str, Dict[str, Any]]:
    """Return the latest quotes for every option under the base coin."""
    result = _request("/v5/market/tickers", params={"category": "option", "baseCoin": base_coin})
    ticker_list = result.get("list", [])
    target_symbol = "BTC-10NOV25-102000-P-USDT"
    for item in ticker_list:
        if item.get("symbol") == target_symbol:
            debug_log.log(f"Bybit get_tickers snapshot for {target_symbol}: {item}")
            break
    return {item["symbol"]: item for item in ticker_list}


def _get_ticker_for_symbol(option_symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch a single option ticker snapshot. Used as a fallback when the bulk call is missing a symbol."""
    result = _request("/v5/market/tickers", params={"category": "option", "symbol": option_symbol})
    ticker_list = result.get("list", [])
    return ticker_list[0] if ticker_list else None


def _parse_option_symbol(option_symbol: str) -> Optional[Tuple[str, float, str]]:
    """
    Parse Bybit option symbol into (expiry, strike, option_type).
    Option symbol formats:
      BTC-29NOV24-65000-C
      BTC-29NOV24-65000-C-USDT
    """
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
    """Return available expiry dates for the given symbol from Bybit."""
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
    """Fetch option quotes from Bybit."""
    base_coin = symbol.upper()
    underlying_price = _get_underlying_price(base_coin)
    if underlying_price is None or underlying_price <= 0:
        debug_log.log(f"Could not determine underlying price for {symbol}.")
        return []

    instrument_list = _get_all_instruments(base_coin)
    if not instrument_list:
        debug_log.log(f"No instruments returned for {symbol.upper()} when fetching {expiry}.")
        return []

    option_tickers = _get_option_tickers(base_coin)

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
            ticker_snapshot = _get_ticker_for_symbol(option_symbol)
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
