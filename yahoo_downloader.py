import configparser
import datetime
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json # Added for JSON serialization
from io import StringIO # Added for reading JSON string to DataFrame

import numpy as np
import pandas as pd
import yfinance as yf 
from yahooquery import Ticker as YQTicker

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
    return tickers


# -----------------------------------------------------------------------------
# Price data download (Unchanged as per user request)
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
    if "adj_close" not in df.columns and "close" in df.columns:
        logger.info("Ticker %s: 'adj_close' column missing; using 'close' as fallback", ticker)
        df["adj_close"] = df["close"]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df["ticker"] = ticker
    df = df.loc[:, ~df.columns.duplicated()]
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
    """Download daily price data for all S&P 500 tickers using yfinance."""
    cfg = _load_cfg(config_file)
    if db_path is None:
        db_path = cfg.get("database", "price_db", fallback="SP500_price_data.db")
    if start_date is None:
        start_date = cfg.get("data_download", "start_price_date", fallback="1900-01-01")
        if not start_date: start_date = "1900-01-01"
    if end_date is None:
        end_date = cfg.get("data_download", "end_price_date")
        if not end_date: end_date = datetime.date.today().strftime("%Y-%m-%d")

    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            _ensure_price_schema(conn)
            cur = conn.cursor()
            tickers = _get_sp500_tickers()
            if "SPY" not in tickers:
                tickers.append("SPY")

            ticker_start: Dict[str, str | None] = {}
            for tk in tickers:
                last = _latest_price_date(cur, tk)
                s_date: str | None = start_date
                if last is not None:
                    s_date = (last + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                    if s_date > end_date:
                        s_date = None
                ticker_start[tk] = s_date
            
            valid_tickers_to_download = {tk: s for tk, s in ticker_start.items() if s is not None}
            if not valid_tickers_to_download:
                logger.info("All price data is up to date.")
                return

            min_overall_start = min(s for s in valid_tickers_to_download.values() if s is not None)

            logger.info(f"Downloading price data for {len(valid_tickers_to_download)} tickers from {min_overall_start} to {end_date}")
            
            df_all = yf.download(
                list(valid_tickers_to_download.keys()),
                start=min_overall_start,
                end=end_date,
                group_by="ticker",
                progress=True,
                auto_adjust=False, 
                actions=False 
            )

            if df_all.empty:
                logger.info("No price data returned from yf.download.")
                return

            processed_count = 0
            for tk, tk_start_date in valid_tickers_to_download.items():
                logger.info("Processing price data for %s (%d/%d)", tk, processed_count + 1, len(valid_tickers_to_download))
                if tk_start_date is None: 
                    logger.info("%s already up to date or no start date", tk)
                    processed_count += 1
                    continue
                
                try:
                    if isinstance(df_all.columns, pd.MultiIndex):
                        df_single = df_all.get(tk)
                        if df_single is None or df_single.empty:
                            logger.warning("No data for ticker %s in downloaded batch.", tk)
                            processed_count += 1
                            continue
                    else: 
                        df_single = df_all
                    
                    df_single = df_single.dropna(how="all")
                    if df_single.empty:
                        processed_count += 1
                        continue
                    
                    df_single_filtered = df_single[df_single.index >= pd.to_datetime(tk_start_date)]
                    
                    if not df_single_filtered.empty:
                        _insert_price_df(cur, df_single_filtered, tk)
                        logger.debug("Stored %d price rows for %s", len(df_single_filtered), tk)
                    else:
                        logger.info("No new price data to store for %s after date filtering.", tk)
                except Exception as e_tk:
                    logger.error("Error processing price data for ticker %s: %s", tk, e_tk)
                
                processed_count += 1
            conn.commit()
            logger.info("Finished downloading and storing price data.")
    except Exception as exc:
        logger.exception("download_price_data failed: %s", exc)


# -----------------------------------------------------------------------------
# Financial data download (Split into acquisition and processing)
# -----------------------------------------------------------------------------

RAW_TABLE = "raw_financials" # Final table for processed data
RAW_COLS = [ 
    "ticker", "report_date", "period", "total_revenue", "eps", "gross_profit",
    "operating_income", "net_income", "research_development", "interest_expense",
    "ebitda", "total_assets", "total_current_assets", "total_current_liabilities",
    "cash_and_eq", "minority_interest", "total_debt", "shares_outstanding",
    "total_liabilities", "operating_cash_flow", "capital_expenditures",
    "free_cash_flow", "price", "forward_eps",
]

# --- Staging Database Helper ---
def _ensure_raw_stage_table_exists(conn: sqlite3.Connection, table_name: str) -> None:
    """Ensures a table for raw staged data exists."""
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                ticker TEXT PRIMARY KEY,
                data_json TEXT
            )
            """
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error creating staging table {table_name}: {e}")
        raise

# --- Final Database Schema Helper (for raw_financials) ---
def _ensure_fin_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{RAW_TABLE}'")
    if cur.fetchone() is None:
        cols_sql = ", ".join([f"{col} REAL" if col not in ['ticker', 'report_date', 'period'] else f"{col} TEXT" for col in RAW_COLS])
        primary_key_sql = "PRIMARY KEY (ticker, report_date, period)"
        if not all(pk_col in RAW_COLS for pk_col in ['ticker', 'report_date', 'period']):
            logger.error(f"Primary key columns for {RAW_TABLE} are not all in RAW_COLS. Aborting schema creation.")
            raise ValueError(f"Primary key columns for {RAW_TABLE} are not all in RAW_COLS.")
        create_table_sql = f"CREATE TABLE {RAW_TABLE} ({cols_sql}, {primary_key_sql})"
        try:
            cur.execute(create_table_sql)
            conn.commit()
            logger.info(f"Created table '{RAW_TABLE}' as it did not exist with schema: {RAW_COLS}")
        except sqlite3.Error as e:
            logger.error(f"Error creating table {RAW_TABLE}: {e}")
            raise
    else: 
        cur.execute(f"PRAGMA table_info({RAW_TABLE})")
        existing_cols = [row[1] for row in cur.fetchall()]
        if not all(col in existing_cols for col in RAW_COLS):
            logger.warning(f"Table '{RAW_TABLE}' exists but schema might mismatch. Expected: {RAW_COLS}, Found: {existing_cols}")
        else:
            logger.info(f"Table '{RAW_TABLE}' schema seems to be in place.")
# =============================================================================
# New helper – use yfinance to fetch raw statements
# =============================================================================
def _fetch_fin_statements_yf(ticker: str) -> Dict[str, Any]:
    """
    Fetch quarterly/annual income, balance-sheet, cash-flow plus last price & forward EPS
    using yfinance (same approach as download_profits.py/get_financial_data).

    返回：
        {
            "q_incs": pd.DataFrame, "a_incs": pd.DataFrame,
            "q_bals": pd.DataFrame, "a_bals": pd.DataFrame,
            "q_cfs": pd.DataFrame, "a_cfs": pd.DataFrame,
            "price": float | np.nan,
            "forward_eps": float | np.nan
        }
    DataFrame 的索引已统一为 DatetimeIndex（report period 末日），列名保持原始。
    """
    try:
        ytk = yf.Ticker(ticker)

        # ------------------------------------------------------------------
        # Robustly grab DataFrames regardless of yfinance version
        # ------------------------------------------------------------------
        def _grab(names: list[str]) -> pd.DataFrame:
            """
            Iterate over a list of attribute names and return the first
            DataFrame found; otherwise return an empty DataFrame.
            """
            for attr in names:
                df = getattr(ytk, attr, None)
                if isinstance(df, pd.DataFrame):
                    return df.copy()
            return pd.DataFrame()

        # Annual / quarterly statements
        a_inc = _grab(["income_stmt", "financials"])
        q_inc = _grab(["quarterly_income_stmt", "quarterly_financials"])
        a_bal = _grab(["balance_sheet"])
        q_bal = _grab(["quarterly_balance_sheet"])
        a_cf  = _grab(["cashflow"])
        q_cf  = _grab(["quarterly_cashflow"])

        # ------------------------------------------------------------------
        # Normalise DataFrames: transpose, ensure DatetimeIndex, sort
        # ------------------------------------------------------------------
        def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            if not isinstance(df.columns, pd.DatetimeIndex):
                df.columns = pd.to_datetime(df.columns, errors="coerce")
            df = df.T
            df.index = pd.to_datetime(df.index, errors="coerce")
            return df.sort_index()

        # Try to obtain last price & forward EPS safely
        price_val = np.nan
        try:
            fast_info = getattr(ytk, "fast_info", {})
            if isinstance(fast_info, dict):
                price_val = fast_info.get("last_price", np.nan)
        except Exception:
            pass

        forward_eps_val = np.nan
        if hasattr(ytk, "info") and isinstance(ytk.info, dict):
            forward_eps_val = ytk.info.get("forwardEps", np.nan)

        out = {
            "q_incs": _norm_df(q_inc),
            "a_incs": _norm_df(a_inc),
            "q_bals": _norm_df(q_bal),
            "a_bals": _norm_df(a_bal),
            "q_cfs":  _norm_df(q_cf),
            "a_cfs":  _norm_df(a_cf),
            "price": price_val,
            "forward_eps": forward_eps_val,
        }
        return out
    except Exception as e:
        logger.error("yfinance fetch failed for %s: %s", ticker, e)
        return {}
    
def _norm(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())

def _first_available(
    df: pd.DataFrame, keys: List[str], default: Any = np.nan, idx: pd.Index | None = None
) -> pd.Series:
    if df.empty:
        return pd.Series(default, index=idx if idx is not None else pd.Index([]), dtype=object)
    target_idx = idx if idx is not None else df.index
    if target_idx.empty and df.empty :
         return pd.Series(default, index=pd.Index([]), dtype=object)
    result_series = pd.Series(default, index=target_idx, dtype=object)
    for k in keys:
        if k in df.columns:
            source_series = df[k]
            if default is not np.nan: 
                 source_series = source_series.fillna(default)
            aligned_series = source_series.reindex(target_idx)
            return aligned_series.fillna(default)
    norm_map = {_norm(c): c for c in df.columns}
    for k in keys:
        nk = _norm(k)
        if nk in norm_map:
            source_series = df[norm_map[nk]]
            if default is not np.nan:
                 source_series = source_series.fillna(default)
            aligned_series = source_series.reindex(target_idx)
            return aligned_series.fillna(default)
    norm_keys = [_norm(k) for k in keys]
    for col_name in df.columns:
        nc = _norm(col_name)
        if any(nk in nc or nc in nk for nk in norm_keys):
            source_series = df[col_name]
            if default is not np.nan:
                 source_series = source_series.fillna(default)
            aligned_series = source_series.reindex(target_idx)
            return aligned_series.fillna(default)
    return result_series

def _latest_report_date(conn: sqlite3.Connection, ticker: str, period_type: str):
    cur = conn.execute(
        f"SELECT MAX(report_date) FROM {RAW_TABLE} WHERE ticker=? AND period=?",
        (ticker, period_type),
    )
    res = cur.fetchone()[0]
    return pd.to_datetime(res) if res else None

def _save_financial_data_for_ticker(
    conn: sqlite3.Connection, 
    ticker: str,
    statement_data: List[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    price_info: float | None,
    forward_eps_info: float | None
) -> None:
    if not statement_data:
        logger.debug("No statement data provided for %s to save to final DB.", ticker)
        return
    all_dfs_to_save = []
    for period_type, inc_df, bal_df, cf_df in statement_data:
        valid_inc = isinstance(inc_df, pd.DataFrame) and not inc_df.empty and isinstance(inc_df.index, pd.DatetimeIndex)
        valid_bal = isinstance(bal_df, pd.DataFrame) and not bal_df.empty and isinstance(bal_df.index, pd.DatetimeIndex)
        valid_cf  = isinstance(cf_df, pd.DataFrame) and not cf_df.empty and isinstance(cf_df.index, pd.DatetimeIndex)
        
        if not (valid_inc and valid_bal and valid_cf): 
            if not valid_inc: logger.debug("Income statement for %s, period %s is invalid or has non-DatetimeIndex.", ticker, period_type)
            if not valid_bal: logger.debug("Balance sheet for %s, period %s is invalid or has non-DatetimeIndex.", ticker, period_type)
            if not valid_cf:  logger.debug("Cash flow statement for %s, period %s is invalid or has non-DatetimeIndex.", ticker, period_type)
            logger.warning("Skipping period %s for ticker %s due to invalid/empty DataFrame(s) or non-DatetimeIndex after loading.", period_type, ticker)
            continue

        all_indices = pd.DatetimeIndex([]) # Initialize as DatetimeIndex
        if valid_inc: all_indices = all_indices.union(inc_df.index)
        if valid_bal: all_indices = all_indices.union(bal_df.index)
        if valid_cf:  all_indices = all_indices.union(cf_df.index)
        
        if all_indices.empty: 
            logger.debug("No common dates found for %s, period %s (final processing), though individual DFs were valid.", ticker, period_type)
            continue
            
        idx_to_process = all_indices
        
        # Ensure idx_to_process is indeed a DatetimeIndex before proceeding
        if not isinstance(idx_to_process, pd.DatetimeIndex):
            logger.error(f"Critical: idx_to_process for {ticker}, period {period_type} is not a DatetimeIndex (type: {type(idx_to_process)}). Skipping this period.")
            continue

        current_idx_tz = idx_to_process.tz # Now this should be safe

        last_db_date = _latest_report_date(conn, ticker, period_type) 
        if last_db_date is not None:
            # Timezone comparison logic
            if current_idx_tz is not None and last_db_date.tzinfo is None:
                last_db_date = pd.Timestamp(last_db_date).tz_localize(current_idx_tz)
            elif current_idx_tz is None and last_db_date.tzinfo is not None:
                last_db_date = last_db_date.tz_localize(None)
            
            mask = idx_to_process > last_db_date
            idx_to_process = idx_to_process[mask]
            if idx_to_process.empty:
                logger.info("No new %s data for %s since %s (final DB check).", period_type, ticker, last_db_date.strftime('%Y-%m-%d'))
                continue
        logger.info("Processing %d new %s report(s) for %s (for final DB).", len(idx_to_process), period_type, ticker)
        
        inc_df_reindexed = inc_df.reindex(idx_to_process) 
        bal_df_reindexed = bal_df.reindex(idx_to_process) 
        cf_df_reindexed  = cf_df.reindex(idx_to_process) 
        
        out = pd.DataFrame(index=idx_to_process)
        out["ticker"] = ticker
        out["report_date"] = idx_to_process 
        out["period"] = period_type
        out["total_revenue"] = _first_available(inc_df_reindexed, ["TotalRevenue"], idx=idx_to_process)
        out["eps"] = _first_available(inc_df_reindexed, ["DilutedEPS", "BasicEPS"], idx=idx_to_process) 
        out["gross_profit"] = _first_available(inc_df_reindexed, ["GrossProfit"], idx=idx_to_process)
        out["operating_income"] = _first_available(inc_df_reindexed, ["OperatingIncome"], idx=idx_to_process)
        out["net_income"] = _first_available(inc_df_reindexed, ["NetIncomeContinuousOperations", "NetIncome", "NetIncomeCommonStockholders"], idx=idx_to_process)
        out["research_development"] = _first_available(inc_df_reindexed, ["ResearchAndDevelopment"], default=0.0, idx=idx_to_process) 
        out["interest_expense"] = _first_available(inc_df_reindexed, ["InterestExpense"], default=0.0, idx=idx_to_process)
        out["ebitda"] = _first_available(inc_df_reindexed, ["EBITDA", "NormalizedEBITDA"], default=np.nan, idx=idx_to_process)
        out["total_assets"] = _first_available(bal_df_reindexed, ["TotalAssets"], idx=idx_to_process)
        out["total_current_assets"] = _first_available(bal_df_reindexed, ["TotalCurrentAssets"], idx=idx_to_process)
        out["total_current_liabilities"] = _first_available(bal_df_reindexed, ["TotalCurrentLiabilities"], idx=idx_to_process)
        out["cash_and_eq"] = _first_available(bal_df_reindexed, ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"], idx=idx_to_process)
        out["minority_interest"] = _first_available(bal_df_reindexed, ["MinorityInterest"], default=0.0, idx=idx_to_process)
        out["total_debt"] = _first_available(bal_df_reindexed, ["TotalDebt"], default=0.0, idx=idx_to_process)
        out["shares_outstanding"] = _first_available(bal_df_reindexed, ["ShareIssued", "OrdinarySharesNumber"], idx=idx_to_process)
        if out["shares_outstanding"].isnull().all(): 
            out["shares_outstanding"] = _first_available(inc_df_reindexed, ["DilutedAverageShares", "BasicAverageShares"], idx=idx_to_process)
        out["total_liabilities"] = _first_available(bal_df_reindexed, ["TotalLiabilitiesNetMinorityInterest", "TotalLiab"], idx=idx_to_process)
        out["operating_cash_flow"] = _first_available(cf_df_reindexed, ["OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"], idx=idx_to_process)
        out["capital_expenditures"] = _first_available(cf_df_reindexed, ["CapitalExpenditure", "PurchaseOfPPE", "NetPPEPurchaseAndSale"], default=0.0, idx=idx_to_process)
        ocf_num = pd.to_numeric(out["operating_cash_flow"], errors="coerce").fillna(0.0)
        capex_num = pd.to_numeric(out["capital_expenditures"], errors="coerce").fillna(0.0)
        out["free_cash_flow"] = ocf_num + capex_num 
        out["price"] = price_info
        out["forward_eps"] = forward_eps_info
        for col in RAW_COLS:
            if col not in out.columns:
                out[col] = np.nan
        out = out.reindex(columns=RAW_COLS) 
        if isinstance(out["report_date"], pd.Series) and pd.api.types.is_datetime64_any_dtype(out["report_date"]):
            if out["report_date"].dt.tz is not None:
                out["report_date"] = out["report_date"].dt.tz_localize(None)
            out["report_date"] = out["report_date"].dt.strftime("%Y-%m-%d")
        numeric_cols = [c for c in RAW_COLS if c not in ("ticker", "report_date", "period")]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        all_dfs_to_save.append(out)
    if all_dfs_to_save:
        final_df_to_save = pd.concat(all_dfs_to_save).reset_index(drop=True)
        final_df_to_save.dropna(subset=["report_date"], inplace=True) 
        if not final_df_to_save.empty:
            try:
                final_df_to_save.to_sql(RAW_TABLE, conn, if_exists="append", index=False)
                conn.commit()
                logger.info("Saved %d records for %s to %s.", len(final_df_to_save), ticker, RAW_TABLE)
            except sqlite3.IntegrityError as ie:
                logger.error(f"Integrity error saving data for ticker {ticker} to {RAW_TABLE}: {ie}. This might be due to duplicate primary key.")
                conn.rollback()
            except Exception as e:
                logger.error("Error saving data for ticker %s to SQL table %s: %s", ticker, RAW_TABLE, e)
                conn.rollback()
        else:
            logger.info("No new data to save for ticker %s to final DB after processing all periods.", ticker)
    else:
        logger.info("No dataframes were prepared for saving for ticker %s to final DB.", ticker)

# --- Helper function to save single ticker data to staging DB ---
def _save_single_ticker_data_to_stage(stage_conn: sqlite3.Connection, ticker: str, table_name: str, data_to_save: Any):
    """
    Serializes and saves data for a single ticker for a single data type to the staging DB.
    'data_to_save' can be a DataFrame, dict, or an error string from yahooquery.
    """
    json_data_str = None
    if isinstance(data_to_save, pd.DataFrame):
        if not data_to_save.empty:
            try:
                # orient='split' is good for DataFrames with DatetimeIndex
                json_data_str = data_to_save.to_json(orient="split", date_format="iso", default_handler=str)
            except Exception as e_json:
                logger.error(f"Error serializing DataFrame to JSON for {ticker}, table {table_name}: {e_json}")
    elif isinstance(data_to_save, dict):
        try:
            json_data_str = json.dumps(data_to_save)
        except Exception as e_json:
            logger.error(f"Error serializing dict to JSON for {ticker}, table {table_name}: {e_json}")
    elif isinstance(data_to_save, str): # Likely an error message from yahooquery
        logger.warning(f"yahooquery returned error for {ticker}, data intended for {table_name}: {data_to_save}")
    elif isinstance(data_to_save, (int, float, np.floating)):
        # Skip NaN values
        if not (isinstance(data_to_save, float) and np.isnan(data_to_save)):
            json_data_str = json.dumps(data_to_save)
    elif data_to_save is not None: 
        logger.warning(f"Unexpected data type for {ticker}, table {table_name}: {type(data_to_save)}. Cannot serialize.")

    if json_data_str:
        try:
            stage_conn.execute(
                f"INSERT OR REPLACE INTO {table_name} (ticker, data_json) VALUES (?, ?)",
                (ticker, json_data_str)
            )
        except sqlite3.Error as e_sql:
            logger.error(f"Error saving to staging DB for {ticker}, {table_name}: {e_sql}")


# --- Helper function to save single ticker data to staging DB ---
def _save_single_ticker_data_to_stage(stage_conn: sqlite3.Connection, ticker: str, table_name: str, data_to_save: Any):
    """
    Serializes and saves data for a single ticker for a single data type to the staging DB.
    'data_to_save' can be a DataFrame, dict, or an error string from yahooquery.
    """
    json_data_str = None
    if isinstance(data_to_save, pd.DataFrame):
        if not data_to_save.empty:
            try:
                # orient='split' is good for DataFrames with DatetimeIndex
                json_data_str = data_to_save.to_json(orient="split", date_format="iso", default_handler=str)
            except Exception as e_json:
                logger.error(f"Error serializing DataFrame to JSON for {ticker}, table {table_name}: {e_json}")
    elif isinstance(data_to_save, dict):
        try:
            json_data_str = json.dumps(data_to_save)
        except Exception as e_json:
            logger.error(f"Error serializing dict to JSON for {ticker}, table {table_name}: {e_json}")
    elif isinstance(data_to_save, str): # Likely an error message from yahooquery
        logger.warning(f"yahooquery returned error for {ticker}, data intended for {table_name}: {data_to_save}")
    elif isinstance(data_to_save, (int, float, np.floating)):
        # Skip NaN values
        if not (isinstance(data_to_save, float) and np.isnan(data_to_save)):
            json_data_str = json.dumps(data_to_save)
    elif data_to_save is not None: 
        logger.warning(f"Unexpected data type for {ticker}, table {table_name}: {type(data_to_save)}. Cannot serialize.")

    if json_data_str:
        try:
            stage_conn.execute(
                f"INSERT OR REPLACE INTO {table_name} (ticker, data_json) VALUES (?, ?)",
                (ticker, json_data_str)
            )
        except sqlite3.Error as e_sql:
            logger.error(f"Error saving to staging DB for {ticker}, {table_name}: {e_sql}")


# -----------------------------------------------------------------------------
# Phase 1 – Data Acquisition  (re-implemented, yfinance version)
# -----------------------------------------------------------------------------
def acquire_raw_financial_data_to_staging(config_file: str = "config.ini") -> bool:
    """
    使用 yfinance 获取财报数据并保存到原有 Staging DB。
    保存逻辑、路径、表名与旧实现完全一致；唯一变化是 → 数据来源。
    """
    logger.info("--- Phase 1 (yfinance): 开始获取并写入 Staging DB ---")
    cfg = _load_cfg(config_file)
    raw_stage_db_path = cfg.get("database", "raw_stage_db", fallback="data/SP500_raw_stage.db")
    Path(raw_stage_db_path).parent.mkdir(parents=True, exist_ok=True)

    # 1) 读取 S&P500 列表
    try:
        tickers = _get_sp500_tickers()
        if not tickers:
            logger.error("未获取到 S&P500 列表，终止。")
            return False
    except Exception as e:
        logger.error("读取 S&P500 列表失败: %s", e)
        return False

    # 2) 确保 staging 表存在
    stage_table_map = {
        "q_incs": "stg_quarterly_income", "q_bals": "stg_quarterly_balance", "q_cfs": "stg_quarterly_cashflow",
        "a_incs": "stg_annual_income",   "a_bals": "stg_annual_balance",   "a_cfs": "stg_annual_cashflow",
        "price": "stg_price_summary",    "forward_eps": "stg_key_stats"
    }
    try:
        with sqlite3.connect(raw_stage_db_path) as conn:
            for tbl in stage_table_map.values():
                _ensure_raw_stage_table_exists(conn, tbl)
    except Exception as e:
        logger.error("初始化 staging DB 失败: %s", e)
        return False

    # 3) 下载并保存（支持并发，可按需拆批）
    success = True
    for idx, tk in enumerate(tickers, 1):
        logger.info("(%d/%d) 下载 %s 财报…", idx, len(tickers), tk)
        fin_data = _fetch_fin_statements_yf(tk)
        if not fin_data:
            logger.warning("跳过 %s – 无数据", tk)
            success = False
            continue

        # 写入 DB：沿用旧的工具函数
        try:
            with sqlite3.connect(raw_stage_db_path) as conn:
                for key, value in fin_data.items():
                    tbl = stage_table_map[key]
                    _save_single_ticker_data_to_stage(conn, tk, tbl, value)
                conn.commit()
        except Exception as e:
            logger.error("写入 %s 财报到 staging 失败: %s", tk, e, exc_info=True)
            success = False

        time.sleep(0.2)  # 轻微节流防封

    if success:
        logger.info("--- Phase 1 完成：全部数据写入成功 (%s) ---", raw_stage_db_path)
    else:
        logger.warning("--- Phase 1 完成，但存在下载/写入错误，请检查日志 ---")
    return success