from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterable, List

import pandas as pd
import streamlit as st

import bybit_api
import covered_call as yfinance_api
from shared_types import OptionQuote


DATA_SOURCES = ("yfinance", "Bybit")
DEFAULT_SYMBOLS = {"yfinance": "AAPL", "Bybit": "BTC"}
DISPLAY_COLUMNS = ["Strike ($)", "Premium ($)", "APR (%)"]
NUMERIC_COLUMNS = [
    "strike",
    "premium",
    "underlying_price",
    "apr",
    "break_even_price",
    "bid",
    "ask",
    "implied_vol",
]


def quotes_to_dataframe(quotes: Iterable[OptionQuote]) -> pd.DataFrame:
    records = [
        {
            **asdict(quote),
            "expiry": quote.expiry.strftime("%Y-%m-%d"),
        }
        for quote in quotes
    ]
    df = pd.DataFrame(records)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def format_strike_with_percent(strike: float, underlying: float) -> str:
    if underlying is None or underlying == 0:
        return f"{strike:.2f}"
    pct_diff = ((strike - underlying) / underlying) * 100
    return f"{strike:.2f} ({pct_diff:.2f}%)"


def _days_until(expiry: str) -> int:
    for fmt in ("%Y-%m-%d", "%d%b%y", "%Y%m%d"):
        try:
            expiry_dt = datetime.strptime(expiry, fmt).replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue
    else:
        return 0

    today = datetime.now(timezone.utc).date()
    return max((expiry_dt.date() - today).days, 0)


def _resolve_api(source: str):
    return yfinance_api if source == "yfinance" else bybit_api


def _render_symbol_input(source: str) -> str:
    default_symbol = DEFAULT_SYMBOLS.get(source, "AAPL")
    return st.text_input("Underlying symbol", value=default_symbol).upper().strip()


def _render_quotes_table(
    *,
    expiry: str,
    quotes: List[OptionQuote],
    strategy: str,
    top_n: int,
) -> None:
    df = quotes_to_dataframe(quotes)
    df["APR (%)"] = df["apr"].apply(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    df["Premium ($)"] = df["premium"].round(2)
    df["Strike ($)"] = df.apply(
        lambda row: format_strike_with_percent(row["strike"], row.get("underlying_price")),
        axis=1,
    )
    df["Days to Expiry"] = df["days_to_expiry"].astype(int)
    days_to_expiry = int(df["Days to Expiry"].iloc[0])
    sort_ascending = strategy == "Covered Call"
    df = df.sort_values("strike", ascending=sort_ascending)

    available_columns = [column for column in DISPLAY_COLUMNS if column in df.columns]
    st.markdown(f"**{strategy} - Expiry: {expiry} - {days_to_expiry} days remaining**")
    st.dataframe(df[available_columns].head(top_n))


def main() -> None:
    st.set_page_config(page_title="Option Income APR Explorer", layout="wide")
    st.title("Option Income APR Explorer")
    st.write(
        "Enter an equity ticker to view annualized returns for covered calls or cash-secured puts."
    )

    data_source = st.radio("Data Source", options=DATA_SOURCES, horizontal=True)

    strategy = st.radio(
        "Strategy",
        options=["Covered Call", "Cash-Secured Put"],
        horizontal=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = _render_symbol_input(data_source)
    with col2:
        top_n = st.number_input("Show top N by APR", min_value=5, max_value=100, value=25, step=5)

    try:
        if not symbol:
            st.info("Provide a ticker symbol to load available expirations.")
            return

        api = _resolve_api(data_source)

        expiries = api.list_option_expiries(symbol)
        if not expiries:
            st.warning("No options expirations found. Check the symbol and try again.")
            return

        default_expiries = expiries[: min(3, len(expiries))]
        selected_expiries = st.multiselect(
            "Option expiries",
            options=expiries,
            default=default_expiries,
            format_func=lambda expiry: f"{expiry} ({_days_until(expiry)} days)",
        )

        if not selected_expiries:
            st.info("Select at least one expiry to view option income data.")
            return

        fetch_fn = (
            api.fetch_covered_call_quotes
            if strategy == "Covered Call"
            else api.fetch_cash_secured_put_quotes
        )

        with st.spinner("Fetching option chain…"):
            quotes_by_expiry = {
                expiry: fetch_fn(symbol, expiry) for expiry in selected_expiries
            }

        if not any(quotes_by_expiry.values()):
            st.warning("No option quotes available for the selected expiries.")
            return

        first_quotes = next((quotes for quotes in quotes_by_expiry.values() if quotes), [])
        underlying_price = first_quotes[0].underlying_price if first_quotes else None

        if underlying_price is not None:
            st.metric("Underlying Price", f"${underlying_price:,.2f}")
            if strategy == "Covered Call":
                apr_caption = "APR = (premium / underlying price) * (365 / days to expiry)."
            else:
                apr_caption = "APR = (premium / strike price) * (365 / days to expiry)."
            st.caption(apr_caption)

        for expiry in selected_expiries:
            quotes = quotes_by_expiry.get(expiry, [])
            if not quotes:
                st.info(f"No {strategy.lower()} quotes found for {expiry}.")
                continue
            _render_quotes_table(expiry=expiry, quotes=quotes, strategy=strategy, top_n=top_n)
    except bybit_api.BybitAPIForbidden as exc:
        st.error(
            "Bybit rejected the request (HTTP 403). The hosting IP is likely blocked for compliance reasons. "
            "Try deploying from a different region or host with Bybit access."
        )
        st.caption(str(exc))
        return
    except bybit_api.BybitAPIError as exc:
        st.error(f"Unable to load Bybit option data: {exc}")
        return

if __name__ == "__main__":
    main()
