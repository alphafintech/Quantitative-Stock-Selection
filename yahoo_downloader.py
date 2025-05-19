import configparser
import datetime
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_cfg(config_file: str = "config.ini") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if os.path.exists(config_file):
        cfg.read(config_file)
    return cfg


def _get_sp500_tickers() -> List[str]:
    df = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0
    )[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    tickers.append("SPY")
    return tickers


# -----------------------------------------------------------------------------
# Price data download
# -----------------------------------------------------------------------------


def _ensure_price_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_data (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.commit()


def _insert_price_df(cur: sqlite3.Cursor, df: pd.DataFrame, ticker: str) -> None:
    df = df.copy().reset_index()
    if "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df.rename(columns=rename, inplace=True)
    df["ticker"] = ticker
    df = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    for row in df.itertuples(index=False):
        cur.execute(
            """
            INSERT OR IGNORE INTO stock_data
            (ticker, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row),
        )


def _latest_price_date(cur: sqlite3.Cursor, ticker: str) -> datetime.date | None:
    cur.execute(
        "SELECT MAX(date) FROM stock_data WHERE ticker=?",
        (ticker,),
    )
    res = cur.fetchone()[0]
    return pd.to_datetime(res).date() if res else None


def download_price_data(
    db_path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    config_file: str = "config.ini",
) -> None:
    """Download daily price data for all S&P 500 tickers."""

    cfg = _load_cfg(config_file)
    if db_path is None:
        db_path = cfg.get("database", "price_db", fallback="SP500_price_data.db")
    if start_date is None:
        start_date = cfg.get("data_download", "start_date", fallback="1900-01-01")
        if not start_date:
            start_date = "1900-01-01"
    if end_date is None:
        end_date = cfg.get("data_download", "end_date")
        if not end_date:
            end_date = datetime.date.today().strftime("%Y-%m-%d")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_price_schema(conn)
        cur = conn.cursor()
        tickers = _get_sp500_tickers()

        ticker_start: Dict[str, str] = {}
        for tk in tickers:
            last = _latest_price_date(cur, tk)
            if last is not None:
                start = (last + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                if start > end_date:
                    start = None
            else:
                start = start_date
            ticker_start[tk] = start

        batch_size = 50
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        for bidx, i in enumerate(range(0, len(tickers), batch_size), start=1):
            batch = tickers[i : i + batch_size]
            min_start = min(
                s for s in [ticker_start[tk] for tk in batch] if s is not None
            ) if any(ticker_start[tk] for tk in batch) else end_date
            try:
                df_all = yf.download(
                    batch,
                    start=min_start,
                    end=end_date,
                    group_by="ticker",
                    progress=False,
                    threads=False,
                )
            except Exception as exc:
                logger.info("[Batch] error %s; retrying after 60s", exc)
                time.sleep(60)
                df_all = yf.download(
                    batch,
                    start=min_start,
                    end=end_date,
                    group_by="ticker",
                    progress=False,
                    threads=False,
                )

            for tk in batch:
                start = ticker_start[tk]
                if start is None:
                    logger.info("%s already up to date", tk)
                    continue
                if tk not in df_all.columns.get_level_values(0):
                    logger.info("[Batch] no data for %s", tk)
                    continue
                df_single = df_all[tk].dropna(how="all")
                if df_single.empty:
                    continue
                df_single = df_single[df_single.index >= start]
                _insert_price_df(cur, df_single, tk)
            conn.commit()
            logger.info("Processed batch %d/%d", bidx, total_batches)
            time.sleep(1)


# -----------------------------------------------------------------------------
# Financial data download
# -----------------------------------------------------------------------------

RETRIES = 3
SLEEP_SEC = 0.2
RAW_TABLE = "raw_financials"
RAW_COLS = [
    "ticker",
    "report_date",
    "period",
    "total_revenue",
    "eps",
    "gross_profit",
    "operating_income",
    "net_income",
    "research_development",
    "interest_expense",
    "ebitda",
    "total_assets",
    "total_current_assets",
    "total_current_liabilities",
    "cash_and_eq",
    "minority_interest",
    "total_debt",
    "shares_outstanding",
    "total_liabilities",
    "operating_cash_flow",
    "capital_expenditures",
    "free_cash_flow",
    "price",
    "forward_eps",
]


def _ensure_fin_schema(conn: sqlite3.Connection) -> None:
    pd.DataFrame(columns=RAW_COLS).to_sql(
        RAW_TABLE, conn, if_exists="append", index=False
    )


def _norm(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def _first_available(
    df: pd.DataFrame, keys: List[str], default=np.nan, idx: pd.Index | None = None
) -> pd.Series:
    if df.empty:
        return pd.Series(default, index=idx if idx is not None else [])
    cols = list(df.columns)
    for k in keys:
        if k in cols:
            s = df[k]
            return s.reindex(idx) if idx is not None else s
    norm_map = {_norm(c): c for c in cols}
    for k in keys:
        nk = _norm(k)
        if nk in norm_map:
            s = df[norm_map[nk]]
            return s.reindex(idx) if idx is not None else s
    norm_keys = [_norm(k) for k in keys]
    for col in cols:
        nc = _norm(col)
        if any(nk in nc or nc in nk for nk in norm_keys):
            s = df[col]
            return s.reindex(idx) if idx is not None else s
    return pd.Series(default, index=idx if idx is not None else df.index)


def _latest_report_date(conn: sqlite3.Connection, ticker: str):
    cur = conn.execute(
        f"SELECT MAX(report_date) FROM {RAW_TABLE} WHERE ticker=?",
        (ticker,),
    )
    res = cur.fetchone()[0]
    return pd.to_datetime(res) if res else None


def _safe_get(getter) -> pd.DataFrame:
    for attempt in range(1, RETRIES + 1):
        try:
            df = getter().T
            if isinstance(df, dict) or df.empty:
                raise ValueError("empty")
            return df
        except Exception:
            if attempt == RETRIES:
                return pd.DataFrame()
            time.sleep(SLEEP_SEC)


def _download_single_ticker(ticker: str):
    yf_tic = yf.Ticker(ticker)
    results: List[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    q_inc = _safe_get(lambda: yf_tic.quarterly_income_stmt)
    q_bal = _safe_get(lambda: yf_tic.quarterly_balance_sheet)
    q_cf = _safe_get(lambda: yf_tic.quarterly_cashflow)
    if not q_inc.empty:
        results.append(("Q", q_inc, q_bal, q_cf))
    a_inc = _safe_get(lambda: yf_tic.income_stmt)
    a_bal = _safe_get(lambda: yf_tic.balance_sheet)
    a_cf = _safe_get(lambda: yf_tic.cashflow)
    if not a_inc.empty:
        union_idx = a_inc.index.union(a_bal.index).union(a_cf.index)
        a_inc = a_inc.reindex(union_idx)
        a_bal = a_bal.reindex(union_idx)
        a_cf = a_cf.reindex(union_idx)
        results.append(("A", a_inc, a_bal, a_cf))
    return results


def _save_batches(conn: sqlite3.Connection, ticker: str, batches) -> None:
    if not batches:
        return
    for period, inc, bal, cf in batches:
        for df in (inc, bal, cf):
            df.index = pd.to_datetime(df.index).tz_localize(None)
        idx = inc.index
        last = _latest_report_date(conn, ticker)
        if last is not None:
            mask = idx > last
            inc, bal, cf = inc[mask], bal[mask], cf[mask]
            idx = inc.index
            if inc.empty:
                continue
        try:
            y = yf.Ticker(ticker)
            price = y.history(period="1d")["Close"].iloc[-1]
            fwd = y.info.get("forwardEps", np.nan)
        except Exception:
            price = fwd = np.nan
        out = pd.DataFrame(
            {
                "ticker": ticker,
                "report_date": idx,
                "total_revenue": _first_available(inc, ["Total Revenue", "Revenue"], idx=idx),
                "eps": _first_available(inc, ["Diluted EPS", "EPS"], idx=idx),
                "gross_profit": _first_available(inc, ["Gross Profit"], idx=idx),
                "operating_income": _first_available(inc, ["Operating Income"], idx=idx),
                "net_income": _first_available(inc, ["Net Income"], idx=idx),
                "research_development": _first_available(
                    inc, ["Research and development", "R&D"], default=np.nan, idx=idx
                ),
                "interest_expense": _first_available(
                    inc, ["Interest Expense"], default=np.nan, idx=idx
                ),
                "ebitda": _first_available(inc, ["EBITDA"], default=np.nan, idx=idx),
                "total_assets": _first_available(bal, ["Total Assets"], idx=idx),
                "total_current_assets": _first_available(
                    bal, ["Total Current Assets"], default=np.nan, idx=idx
                ),
                "total_current_liabilities": _first_available(
                    bal, ["Total Current Liabilities"], default=np.nan, idx=idx
                ),
                "cash_and_eq": _first_available(
                    bal, ["Cash And Cash Equivalents"], default=np.nan, idx=idx
                ),
                "minority_interest": _first_available(
                    bal, ["Minority Interest"], default=0, idx=idx
                ),
                "total_debt": _first_available(bal, ["Long Term Debt"], default=0, idx=idx)
                + _first_available(bal, ["Short Long Term Debt"], default=0, idx=idx),
                "shares_outstanding": _first_available(
                    bal, ["Ordinary Shares Number", "Share Issued"], default=np.nan, idx=idx
                ),
                "total_liabilities": _first_available(
                    bal, ["Total Liab", "Total Liabilities"], idx=idx
                ),
                "operating_cash_flow": _first_available(
                    cf, ["Operating Cash Flow"], idx=idx
                ),
                "capital_expenditures": _first_available(
                    cf, ["Capital Expenditures", "CapEx"], default=np.nan, idx=idx
                ),
                "free_cash_flow": pd.Series(dtype=float),
                "price": price,
                "forward_eps": fwd,
            }
        )
        out["period"] = period
        out["free_cash_flow"] = (
            pd.to_numeric(out["operating_cash_flow"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["capital_expenditures"], errors="coerce").fillna(0.0)
        )
        out = out.reindex(columns=RAW_COLS)
        out["report_date"] = pd.to_datetime(out["report_date"]).dt.strftime("%Y-%m-%d")
        numeric_cols = [c for c in out.columns if c not in ("ticker", "report_date", "period")]
        out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
        out.to_sql(RAW_TABLE, conn, if_exists="append", index=False)


def download_financial_data(
    db_path: str | None = None, config_file: str = "config.ini"
) -> None:
    """Download Yahoo Finance financial statements for all S&P 500 tickers."""

    cfg = _load_cfg(config_file)
    if db_path is None:
        db_path = cfg.get("database", "finance_db", fallback="SP500_finance_data.db")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    tickers = _get_sp500_tickers()
    with sqlite3.connect(db_path) as conn:
        _ensure_fin_schema(conn)
        total = len(tickers)
        for idx, tk in enumerate(tickers, start=1):
            batches = _download_single_ticker(tk)
            _save_batches(conn, tk, batches)
            logger.info("Saved %s (%d/%d)", tk, idx, total)
            time.sleep(0.1)
