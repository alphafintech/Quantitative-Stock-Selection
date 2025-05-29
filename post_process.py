"""
post_process.py

Utility functions for post‑run reporting and portfolio rebalancing.

The key public entry point is ``rebalance_portfolio`` which:
1. Reads the current holdings Excel produced by Gemini (`Gemini_current_holdings.xlsx`).
2. Pulls the latest closing prices for every ticker from the SQLite database that stores
   S&P 500 price history (`SP500_price_data.db`).
3. Calculates the current market value of every equity position plus the existing cash
   balance to obtain *current_total_asset*.
4. Parses the text file that describes the desired target allocation
   (`Gemini_portfolio_adjustment.txt`).
5. Generates a trade list that will transform the current portfolio into the target
   allocation, assuming execution at the latest close.
6. Writes the full “before → trades → after” story into a multi‑sheet Excel file whose
   path is supplied by the caller.

Author: ChatGPT‑o3
Date: 2025‑05‑28
"""
from __future__ import annotations

import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import datetime
from contextlib import suppress
with suppress(ImportError):
    import yfinance as yf

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "[Rebalance] %(levelname)s – %(asctime)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _get_latest_price(conn: sqlite3.Connection, ticker: str) -> float | None:
    """
    Return the latest close price for *ticker* from any plausible table in *conn*.

    We attempt several heuristics because the exact table schema can vary:
        • A table literally named ``ticker`` (case sensitive).
        • Upper‑case / lower‑case variations.
        • A consolidated table named ``price_data`` with columns (ticker, date, close).

    Returns
    -------
    float | None
        Latest close price if found, else ``None``.
    """
    # helper to extract close-like column
    def _extract_close(df: pd.DataFrame) -> float | None:
        for col in ("close", "adj_close", "adjusted_close"):
            if col in df.columns and pd.notna(df[col].iloc[0]):
                return float(df[col].iloc[0])
        return None

    # 1) table per ticker
    for table in {ticker, ticker.upper(), ticker.lower()}:
        try:
            df = pd.read_sql(
                f"SELECT * FROM '{table}' ORDER BY date DESC LIMIT 1",
                conn,
            )
            if not df.empty:
                price = _extract_close(df)
                if price is not None:
                    return price
        except Exception:
            # table might not exist or schema mismatch
            pass

    # 2) consolidated table
    try:
        df = pd.read_sql(
            "SELECT * FROM price_data WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            conn,
            params=(ticker,),
        )
        if not df.empty:
            price = _extract_close(df)
            if price is not None:
                return price
    except Exception:
        pass

    # 3) fallback to yfinance
    if "yf" in globals():
        try:
            ticker_obj = yf.Ticker(ticker)
            end = datetime.datetime.now().date()
            start = end - datetime.timedelta(days=7)
            hist = ticker_obj.history(start=start.isoformat(), end=end.isoformat())
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass

    _LOGGER.warning("Could not find latest price for %s", ticker)
    return None


def _parse_adjustment_file(path: Path) -> Dict[str, float]:
    """
    Parse ``Gemini_portfolio_adjustment.txt`` style file.

    Expected pattern per line:  ``<ticker>: <pct>%`` (case‑insensitive).
    Additional commentary is ignored.

    Returns
    -------
    Dict[str, float]
        Mapping ticker → target weight (0‑100, *not* 0‑1).
    """
    # Support both ASCII and full‑width variants of colon (:) and percent (%)
    pattern = re.compile(
        r"^\s*([A-Za-z0-9\.\-\_]+)\s*[：:]\s*([\d\.]+)\s*[％%]",
        flags=re.UNICODE,
    )
    weights: Dict[str, float] = {}

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        # Remove common bullet symbols or list markers before attempting to parse
        line = raw_line.lstrip().lstrip("•●*-").strip()
        m = pattern.search(line)
        if m:
            ticker, pct = m.groups()
            weights[ticker.upper()] = float(pct)

    if not weights:
        raise ValueError(f"No weights parsed from {path}")

    # normalise: if total != 100 we *do not* scale – missing portion → CASH
    total = sum(weights.values())
    if total > 100.01:
        raise ValueError(
            f"Target weights sum to {total:.2f} %, which exceeds 100 %"
        )
    if "CASH" not in weights:
        weights["CASH"] = max(0.0, 100.0 - total)

    return weights


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def rebalance_portfolio(
    holdings_xlsx_path: str | Path,
    price_db_path: str | Path,
    adjustment_txt_path: str | Path,
    output_html_path: str | Path,
    report_title: str = "Portfolio Rebalance Report",
) -> None:
    """
    End‑to‑end rebalance planner.

    Parameters
    ----------
    holdings_xlsx_path
        Path to ``Gemini_current_holdings.xlsx``.
    price_db_path
        Path to ``SP500_price_data.db`` with historic prices.
    adjustment_txt_path
        Path to ``Gemini_portfolio_adjustment.txt`` that specifies target weights.
    output_html_path
        Where to save the resulting HTML report.
    report_title
        HTML report <title> and <h1> text (default "Portfolio Rebalance Report").
    """
    holdings_path = Path(holdings_xlsx_path)
    db_path = Path(price_db_path)
    adj_path = Path(adjustment_txt_path)
    out_path = Path(output_html_path)

    # ------------------------------------------------------------------ #
    # 1) Load current holdings
    # ------------------------------------------------------------------ #
    _LOGGER.info("Reading current holdings from %s", holdings_path)
    df_holdings = pd.read_excel(holdings_path)

    if df_holdings.empty:
        raise ValueError("Current holdings file is empty")

    # identify cash row
    cash_mask = df_holdings["Ticker/Cash"].str.upper() == "CASH"
    if cash_mask.sum() != 1:
        raise ValueError("Holdings sheet must contain exactly one 'Cash' row")
    cash_value = float(df_holdings.loc[cash_mask, "TOTAL VALUE"].iloc[0])

    # equities
    df_equities = df_holdings.loc[~cash_mask].copy()
    df_equities["Ticker"] = df_equities["Ticker/Cash"].str.upper()

    # ------------------------------------------------------------------ #
    # 2) Fetch latest prices & position values
    # ------------------------------------------------------------------ #
    _LOGGER.info("Querying latest prices from %s", db_path)
    with sqlite3.connect(db_path) as conn:
        prices = {
            t: _get_latest_price(conn, t) for t in df_equities["Ticker"].unique()
        }

    df_equities["Latest Price"] = df_equities["Ticker"].map(prices)
    missing = df_equities["Latest Price"].isna()
    if missing.any():
        _LOGGER.warning(
            "Missing prices for: %s",
            ", ".join(df_equities.loc[missing, "Ticker"]),
        )
        df_equities = df_equities.loc[~missing]  # drop rows we cannot value

    df_equities["Position Value"] = (
        df_equities["Share count"] * df_equities["Latest Price"]
    )

    current_total_asset = df_equities["Position Value"].sum() + cash_value
    _LOGGER.info("Current total asset: %.2f", current_total_asset)

    # snapshot before state
    df_before = df_equities[
        [
            "Ticker",
            "Share count",
            "Latest Price",
            "Position Value",
        ]
    ].copy()
    df_before["Weight %"] = (
        df_before["Position Value"] / current_total_asset * 100
    )

    # ------------------------------------------------------------------ #
    # 3) Parse target allocation
    # ------------------------------------------------------------------ #
    tgt_weights = _parse_adjustment_file(adj_path)
    _LOGGER.info("Target weights: %s", tgt_weights)

    # ensure every current ticker present (default 0 % for clearing)
    for t in df_before["Ticker"]:
        tgt_weights.setdefault(t, 0.0)

    # ------------------------------------------------------------------ #
    # 4) Build trade list & after state
    # ------------------------------------------------------------------ #
    trade_rows: List[Dict[str, object]] = []
    after_rows: List[Tuple] = []

    # iterate over union of tickers
    tickers = sorted(tgt_weights.keys())
    for ticker in tickers:
        target_pct = tgt_weights[ticker]
        target_value = current_total_asset * target_pct / 100

        if ticker == "CASH":
            after_rows.append(
                (
                    "CASH",
                    math.nan,
                    math.nan,
                    target_value,
                    target_pct,
                )
            )
            continue

        current_shares = (
            df_before.set_index("Ticker")["Share count"].get(ticker, 0.0)
        )
        price = prices.get(ticker)
        if price is None:
            # cannot trade without price – skip
            continue

        target_shares = math.floor(target_value / price)
        delta_shares = target_shares - current_shares

        # action description
        if delta_shares > 0:
            action = f"Buy {delta_shares}"
        elif delta_shares < 0:
            action = f"Sell {abs(delta_shares)}"
        else:
            action = "No action"

        trade_rows.append(
            {
                "Ticker": ticker,
                "Latest Price": price,
                "Δ Shares": delta_shares,
                "Action": action,
                "Est. Trade Value": round(delta_shares * price, 2),
            }
        )

        # after position
        new_value = target_shares * price
        new_weight = new_value / current_total_asset * 100
        after_rows.append(
            (
                ticker,
                target_shares,
                price,
                new_value,
                new_weight,
            )
        )

    # ------------------------------------------------------------------ #
    # 5) Assemble DataFrames
    # ------------------------------------------------------------------ #
    df_target = (
        pd.DataFrame.from_records(
            [
                {"Ticker": k, "Target %": v}
                for k, v in tgt_weights.items()
                if k != "CASH"
            ]
        )
        .sort_values("Target %", ascending=False)
        .reset_index(drop=True)
    )

    # sort trades by absolute estimated trade value, highest first
    df_trades = (
        pd.DataFrame(trade_rows)
        .sort_values("Est. Trade Value", ascending=False)
        .reset_index(drop=True)
    )

    df_after = pd.DataFrame(
        after_rows,
        columns=[
            "Ticker",
            "New Shares",
            "Latest Price",
            "New Position Value",
            "Weight %",
        ],
    ).sort_values("Ticker")

    summary = pd.DataFrame(
        {
            "Metric": [
                "Current Total Asset",
                "Cash Before",
                "Cash After",
            ],
            "Value": [
                current_total_asset,
                cash_value,
                tgt_weights.get("CASH", 0) / 100 * current_total_asset,
            ],
        }
    )

    # ------------------------------------------------------------------ #
    # 6) Build and write HTML
    # ------------------------------------------------------------------ #
    _LOGGER.info("Writing rebalance HTML report to %s", out_path)
    css = """
    body { font-family: Arial, sans-serif; margin: 1.5rem; }
    h1 { color: #2c3e50; }
    h2 { border-bottom: 2px solid #34495e; padding-bottom: .3rem; color:#34495e; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
    th { background: #f2f2f2; }
    td:first-child, th:first-child { text-align: left; }
    """
    def _tbl(df: pd.DataFrame, index=False):
        """
        Convert DataFrame to HTML while showing full numbers (no scientific notation)
        and adding thousands separators where appropriate.
        """
        df_fmt = df.copy()
        numeric_cols = df_fmt.select_dtypes(include=["number"]).columns
        for c in numeric_cols:
            # keep integers without decimals, floats with 2 decimals
            if pd.api.types.is_integer_dtype(df_fmt[c]):
                df_fmt[c] = df_fmt[c].apply(lambda x: f"{x:,}" if pd.notna(x) else "")
            else:
                df_fmt[c] = df_fmt[c].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else ""
                )
        return df_fmt.to_html(index=index, border=0, classes="tbl", escape=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{report_title}</title>
    <style>{css}</style>
</head>
<body>
    <h1>{report_title}</h1>
    <h2>Summary</h2>
    {_tbl(summary, index=False)}
    <h2>Before Rebalance</h2>
    {_tbl(df_before, index=False)}
    <h2>Target Allocation</h2>
    {_tbl(df_target, index=False)}
    <h2>Proposed Trades</h2>
    {_tbl(df_trades, index=False)}
    <h2>After Rebalance (Projected)</h2>
    {_tbl(df_after, index=False)}
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    _LOGGER.info("Rebalance HTML report successfully saved.")

# --------------------------------------------------------------------------- #
# convenience wrapper
# --------------------------------------------------------------------------- #
def rebalance_GPT_portfolio() -> None:
    """
    Rebalance the *GPT* portfolio using canonical file locations
    relative to the project’s root directory (the directory that
    contains this ``post_process.py``).

    Paths resolved:
      • Portfolio/GPT_current_holdings.xlsx          → holdings_xlsx_path
      • SP500_price_data.db                          → price_db_path
      • Portfolio/GPT_portfolio_adjustment.txt       → adjustment_txt_path
      • result_output/GPT_portfolio_adjustment.html  → output_html_path
    """
    root = Path(__file__).resolve().parent

    holdings_xlsx_path = root / "Portfolio" / "GPT_current_holdings.xlsx"
    price_db_path = root / "SP500_price_data.db"
    adjustment_txt_path = root / "Portfolio" / "GPT_portfolio_adjustment.txt"
    output_html_path = root / "result_output" / "GPT_portfolio_adjustment.html"

    rebalance_portfolio(
        holdings_xlsx_path=holdings_xlsx_path,
        price_db_path=price_db_path,
        adjustment_txt_path=adjustment_txt_path,
        output_html_path=output_html_path,
        report_title = "GPT Portfolio Rebalance Report",
    )

def rebalance_Gemini_portfolio() -> None:
    """
    Rebalance the *Gemini* portfolio using canonical file locations
    relative to the project's root directory (the directory that
    contains this ``post_process.py``).

    Paths resolved:
      • Portfolio/Gemini_current_holdings.xlsx          → holdings_xlsx_path
      • SP500_price_data.db                             → price_db_path
      • Portfolio/Gemini_portfolio_adjustment.txt       → adjustment_txt_path
      • result_output/Gemini_portfolio_adjustment.html  → output_html_path
    """
    root = Path(__file__).resolve().parent

    holdings_xlsx_path = root / "Portfolio" / "Gemini_current_holdings.xlsx"
    price_db_path = root / "SP500_price_data.db"
    adjustment_txt_path = root / "Portfolio" / "Gemini_portfolio_adjustment.txt"
    output_html_path = root / "result_output" / "Gemini_portfolio_adjustment.html"

    rebalance_portfolio(
        holdings_xlsx_path=holdings_xlsx_path,
        price_db_path=price_db_path,
        adjustment_txt_path=adjustment_txt_path,
        output_html_path=output_html_path,
        report_title = "Gemini Portfolio Rebalance Report",
    )
def main() -> None:
    """
    Convenience entry point:
      1. Rebalance GPT and Gemini portfolios.
      2. Optionally invoke an external `portfolio_yield` routine if available.
    """
    rebalance_GPT_portfolio()
    rebalance_Gemini_portfolio()

    # Optional post‑processing: portfolio yield calculation
    try:
        import importlib

        py_mod = importlib.import_module("portfolio_yield")
        if hasattr(py_mod, "portfolio_yield"):
            py_mod.portfolio_yield()
        elif hasattr(py_mod, "calculate_portfolio_yield"):
            py_mod.calculate_portfolio_yield()
        elif hasattr(py_mod, "compute_and_plot_yield"):
            py_mod.compute_and_plot_yield()
        else:
            _LOGGER.warning(
                "Module 'portfolio_yield' imported but no recognised function found."
            )
    except ModuleNotFoundError:
        _LOGGER.info("Module 'portfolio_yield' not present; skipping yield calculation.")
    except Exception as exc:
        _LOGGER.warning("Calling portfolio_yield failed: %s", exc)
if __name__ == "__main__":  # pragma: no cover
    main()