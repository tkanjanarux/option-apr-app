from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import covered_call
from shared_types import OptionQuote


class FakeTicker:
    def __init__(self, symbol: str, chain: SimpleNamespace):
        self.symbol = symbol
        self.fast_info = {"lastPrice": 100.0}
        self._chain = chain

    def history(self, period: str):  # pragma: no cover - fallback path
        return pd.DataFrame({"Close": [self.fast_info["lastPrice"]]})

    def option_chain(self, expiry: str):
        return self._chain


def _setup_monkeypatched_ticker(monkeypatch, calls: pd.DataFrame, puts: pd.DataFrame):
    chain = SimpleNamespace(calls=calls, puts=puts)

    def fake_ticker(symbol: str):
        return FakeTicker(symbol, chain)

    monkeypatch.setattr(covered_call.yf, "Ticker", fake_ticker)


def _expiry_in_days(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


def _expected_days(expiry_str: str) -> int:
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return covered_call._calculate_days_to_expiry(expiry_dt)


def test_fetch_covered_call_quotes_filters_invalid_rows(monkeypatch):
    expiry = _expiry_in_days(30)
    calls = pd.DataFrame(
        [
            {"strike": 110, "bid": 2.5, "ask": 3.0, "impliedVolatility": 0.5},
            {"strike": 95, "bid": 1.0, "ask": 1.2, "impliedVolatility": 0.4},
        ]
    )
    _setup_monkeypatched_ticker(monkeypatch, calls, pd.DataFrame())

    quotes = covered_call.fetch_covered_call_quotes("AAPL", expiry)

    assert len(quotes) == 1
    quote = quotes[0]
    assert isinstance(quote, OptionQuote)
    assert quote.strike == 110
    expected_days = _expected_days(expiry)
    expected_apr = (2.5 / 100.0) * (365 / expected_days) * 100
    assert quote.days_to_expiry == expected_days
    assert quote.apr == pytest.approx(expected_apr)


def test_fetch_cash_secured_put_quotes_filters_invalid_rows(monkeypatch):
    expiry = _expiry_in_days(20)
    puts = pd.DataFrame(
        [
            {"strike": 90, "bid": 3.0, "ask": 3.5, "impliedVolatility": 0.6},
            {"strike": 110, "bid": 4.0, "ask": 4.5, "impliedVolatility": 0.7},
        ]
    )
    _setup_monkeypatched_ticker(monkeypatch, pd.DataFrame(), puts)

    quotes = covered_call.fetch_cash_secured_put_quotes("AAPL", expiry)

    assert len(quotes) == 1
    quote = quotes[0]
    expected_days = _expected_days(expiry)
    expected_apr = (3.0 / 90.0) * (365 / expected_days) * 100
    assert quote.option_type == "put"
    assert quote.strike == 90
    assert quote.apr == pytest.approx(expected_apr)
