from dataclasses import asdict
from typing import List

import pandas as pd
import streamlit as st

import bybit_api
import covered_call as yfinance_api
import debug_log
from shared_types import OptionQuote


def quotes_to_dataframe(quotes: List[OptionQuote]) -> pd.DataFrame:
    records = [
        {
            **asdict(quote),
            "expiry": quote.expiry.strftime("%Y-%m-%d"),
        }
        for quote in quotes
    ]
    df = pd.DataFrame(records)
    numeric_cols = [
        "strike",
        "premium",
        "underlying_price",
        "apr",
        "break_even_price",
        "bid",
        "ask",
        "implied_vol",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_strike_with_percent(strike: float, underlying: float) -> str:
    if underlying is None or underlying == 0:
        return f"{strike:.2f}"
    pct_diff = ((strike - underlying) / underlying) * 100
    return f"{strike:.2f} ({pct_diff:.2f}%)"


def _days_until(expiry: str) -> int:
    from datetime import datetime, timezone

    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            expiry_dt = datetime.strptime(expiry, "%d%b%y").replace(tzinfo=timezone.utc)
        except ValueError:
            expiry_dt = datetime.strptime(expiry, "%Y%m%d").replace(tzinfo=timezone.utc)
    today = datetime.now(timezone.utc).date()
    return max((expiry_dt.date() - today).days, 0)


def main() -> None:
    st.set_page_config(page_title="Option Income APR Explorer", layout="wide")
    st.title("Option Income APR Explorer")
    st.write(
        "Enter an equity ticker to view annualized returns for covered calls or cash-secured puts."
    )

    show_debug_logs = st.sidebar.checkbox("Show Bybit debug log", value=False)
    debug_log.clear()

    data_source = st.radio(
        "Data Source",
        options=["yfinance", "Bybit"],
        horizontal=True,
    )

    strategy = st.radio(
        "Strategy",
        options=["Covered Call", "Cash-Secured Put"],
        horizontal=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if data_source == "Bybit":
            symbol = st.text_input("Underlying symbol", value="BTC").upper().strip()
        else:
            symbol = st.text_input("Underlying symbol", value="AAPL").upper().strip()
    with col2:
        top_n = st.number_input("Show top N by APR", min_value=5, max_value=100, value=25, step=5)

    try:
        if not symbol:
            st.info("Provide a ticker symbol to load available expirations.")
            return

        if data_source == "yfinance":
            api = yfinance_api
        else:
            api = bybit_api

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

        display_columns = [
            "Strike ($)",
            "Premium ($)",
            "APR (%)",
        ]

        for expiry in selected_expiries:
            quotes = quotes_by_expiry.get(expiry, [])
            if not quotes:
                st.info(f"No {strategy.lower()} quotes found for {expiry}.")
                continue

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

            available_columns = [c for c in display_columns if c in df.columns]
            st.markdown(f"**{strategy} - Expiry: {expiry} - {days_to_expiry} days remaining**")
            st.dataframe(df[available_columns].head(top_n))
    except bybit_api.BybitAPIForbidden as exc:
        st.error(
            "Bybit rejected the request (HTTP 403). Streamlit Cloud IPs are often blocked or rate-limited. "
            "Set a `BYBIT_API_BASE_URL` (for example https://api.bytick.com) or provide a `BYBIT_HTTP_PROXY` "
            "via Streamlit secrets/env vars to route traffic through a permitted region."
        )
        st.caption(str(exc))
        return
    except bybit_api.BybitAPIError as exc:
        st.error(f"Unable to load Bybit option data: {exc}")
        return
    finally:
        if show_debug_logs:
            log_text = "\n".join(debug_log.get_logs()) or "No logs captured yet."
            st.sidebar.text_area("Bybit Debug Log", value=log_text, height=240)
        elif data_source == "Bybit":
            st.sidebar.caption("Enable 'Show Bybit debug log' to see request diagnostics.")

if __name__ == "__main__":
    main()
