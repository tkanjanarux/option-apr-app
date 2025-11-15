from datetime import datetime, timedelta, timezone

import pandas as pd

import app
from shared_types import OptionQuote


def make_quote(**overrides) -> OptionQuote:
    base = dict(
        ticker="TEST",
        option_type="call",
        expiry=datetime(2024, 1, 1, tzinfo=timezone.utc),
        strike=105.0,
        premium=5.0,
        underlying_price=100.0,
        days_to_expiry=30,
        apr=12.5,
        break_even_price=95.0,
        bid=4.5,
        ask=5.5,
        implied_vol=0.6,
    )
    base.update(overrides)
    return OptionQuote(**base)


def test_quotes_to_dataframe_converts_numeric_fields():
    quote = make_quote()
    df = app.quotes_to_dataframe([quote])
    for column in app.NUMERIC_COLUMNS:
        assert column in df.columns
        assert pd.api.types.is_numeric_dtype(df[column])
    assert df.iloc[0]["expiry"] == quote.expiry.strftime("%Y-%m-%d")


def test_format_strike_with_percent_handles_underlying():
    formatted = app.format_strike_with_percent(105, 100)
    assert formatted.startswith("105.00")
    assert "5.00%" in formatted


def test_days_until_supports_multiple_formats():
    target = datetime.now(timezone.utc) + timedelta(days=10)
    formatted = target.strftime("%Y-%m-%d")
    assert app._days_until(formatted) == 10
    formatted_alt = target.strftime("%d%b%y").upper()
    assert app._days_until(formatted_alt) == 10
