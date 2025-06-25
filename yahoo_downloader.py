import configparser
import datetime
import logging
import os
import sqlite3
import time
from pathlib import Path
# Absolute path to the repository / program root (folder that contains this file)
ROOT_DIR = Path(__file__).resolve().parent

def _abs_path(path_str: str | Path) -> str:
    """
    Return an absolute path string that is guaranteed to live under the
    project root (ROOT_DIR) *unless* an absolute path was explicitly given.
    """
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = ROOT_DIR / p
    return str(p)
from typing import Dict, List, Tuple, Any
import json # Added for JSON serialization
from io import StringIO # Added for reading JSON string to DataFrame

import numpy as np
import pandas as pd
import yfinance as yf 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_cfg(config_file: str = "config.ini") -> configparser.ConfigParser:
    """
    Load configuration file, resolving the provided path to an absolute
    path. This ensures relative paths work no matter the current
    working directory.
    """
    cfg = configparser.ConfigParser()
    cfg_path = Path(config_file).expanduser().resolve()
    if cfg_path.exists():
        cfg.read(cfg_path, encoding="utf-8")
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
    # ---- ensure absolute path ------------------------------------------------
    db_path = _abs_path(db_path)
    if start_date is None:
        start_date = cfg.get("data_download", "start_price_date", fallback="1900-01-01")
        if not start_date: start_date = "1900-01-01"
    if end_date is None:
        end_date = cfg.get("data_download", "end_price_date", fallback="")
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
            
            # ------------------------------------------------------------------
            # Batched, single‑threaded download to mitigate Yahoo rate limiting
            # ------------------------------------------------------------------
            BATCH_SIZE = 50  # ≤50 tickers per request
            df_parts: list[pd.DataFrame] = []

            ticker_list = list(valid_tickers_to_download.keys())
            for batch_start in range(0, len(ticker_list), BATCH_SIZE):
                sub = ticker_list[batch_start : batch_start + BATCH_SIZE]
                logger.info("Batch %d – downloading %d tickers: %s … %s",
                            batch_start // BATCH_SIZE + 1, len(sub), sub[0], sub[-1])
                try:
                    df_chunk = yf.download(
                        sub,
                        start=min_overall_start,
                        end=end_date,
                        group_by="ticker",
                        progress=False,
                        auto_adjust=False,
                        actions=False,
                        threads=False,     # force single‑threaded inside yfinance
                    )
                    if df_chunk is not None and not df_chunk.empty:
                        df_parts.append(df_chunk)
                except Exception as e_dl:
                    logger.warning("Batch %s…%s failed: %s", sub[0], sub[-1], e_dl)
                    raise  # escalate so caller knows batch failed

                time.sleep(0.1)  # small delay between batches to stay under rate limits

            if not df_parts:
                logger.warning("All batched downloads returned empty DataFrames.")
                raise RuntimeError("No price data fetched from yfinance; aborting.")

            df_all = pd.concat(df_parts, axis=1)

            if df_all.empty:
                logger.warning("yf.download returned an empty DataFrame – possible network/limit error.")
                raise RuntimeError("No price data fetched from yfinance; aborting.")

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
                            # raise so the caller knows which ticker is missing
                            raise RuntimeError(f"Price data missing for ticker {tk}")
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

# =============================================================================
# New helper – use yfinance to fetch raw statements
# =============================================================================
def _fetch_fin_statements_yf(ticker: str) -> Dict[str, Any]:
    """
    Fetch quarterly/annual income, balance-sheet, cash-flow plus last price & forward EPS
    using yfinance (same approach as download_profits.py/get_financial_data).

    Returns:
        {
            "q_incs": pd.DataFrame, "a_incs": pd.DataFrame,
            "q_bals": pd.DataFrame, "a_bals": pd.DataFrame,
            "q_cfs": pd.DataFrame, "a_cfs": pd.DataFrame,
            "price": float | np.nan,
            "forward_eps": float | np.nan
        }
    The DataFrame indices are standardized to ``DatetimeIndex`` (end of report
    period) and column names are left unchanged.
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
    

# -----------------------------------------------------------------------------
# Completeness check helper (horizontal + vertical)
# -----------------------------------------------------------------------------
def _filter_latest_quarter(
    df: pd.DataFrame,
    horiz_thresh: float = 0.5,
    vert_thresh: float = 0.2,
    history_qtrs: int = 8,
    structural_presence_thresh: float = 0.2,
) -> tuple[pd.DataFrame, bool]:
    """
    Apply horizontal + vertical completeness rules to a quarterly statement
    DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Quarterly statement where each row is a period (index = period end).
    horiz_thresh : float
        Initial “raw” missing‑ratio threshold for the latest quarter.
    vert_thresh : float
        Missing‑ratio threshold *after* removing structurally absent columns.
    history_qtrs : int
        How many previous quarters to inspect when determining structural
        absence.  Default 8 (~2 years).
    structural_presence_thresh : float
        A column is considered *optional* if its historical presence rate
        (non‑NA share) ≤ this value.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        (Possibly trimmed DataFrame, is_complete_flag).  If the latest quarter
        is deemed incomplete, that row is removed from the returned DataFrame.
    """
    if df is None or df.empty:
        return df, False

    df = df.sort_index()  # ensure ascending order
    latest_row = df.iloc[-1]

    # 1) Horizontal missing ratio (raw)
    horiz_missing = latest_row.isna().mean()

    # 2) Identify structurally optional columns based on history
    df_hist = df.iloc[-(history_qtrs + 1):-1]  # previous N quarters
    if df_hist.empty:
        presence_rate = pd.Series([], dtype=float)
    else:
        presence_rate = df_hist.notna().mean()

    optional_cols = presence_rate[presence_rate <= structural_presence_thresh].index

    # 3) Vertical missing ratio after removing optional columns
    considered_cols = latest_row.drop(labels=optional_cols, errors="ignore")
    if considered_cols.empty:
        vert_missing = 1.0  # treat as fully missing if nothing to check
    else:
        vert_missing = considered_cols.isna().mean()

    # 4) Final completeness decision
    is_complete = vert_missing <= vert_thresh

    # Keep a debug log
    logger.debug(
        "Completeness chk – horiz: %.2f vert: %.2f (opt cols: %d) → %s",
        horiz_missing, vert_missing, len(optional_cols), "OK" if is_complete else "FAIL"
    )

    # If incomplete, drop latest row before returning
    if not is_complete:
        df = df.iloc[:-1]

    return df, is_complete
    

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
    Use ``yfinance`` to fetch financial statements and store them in the existing
    staging database. Save logic, paths and table names are unchanged – only the
    data source differs.
    """
    logger.info("--- Phase 1 (yfinance): 开始获取并写入 Staging DB ---")
    cfg = _load_cfg(config_file)
    # ---------------- completeness thresholds -----------------------------
    horiz_thresh = cfg.getfloat("completeness", "latest_qtr_max_gap", fallback=0.5)
    vert_thresh  = cfg.getfloat("completeness", "after_filter_max_gap", fallback=0.2)
    history_qtrs = cfg.getint("completeness", "history_quarters", fallback=8)
    raw_stage_db_path = cfg.get("database", "raw_stage_db", fallback="data/SP500_raw_stage.db")
    # ---- ensure absolute path ------------------------------------------------
    raw_stage_db_path = _abs_path(raw_stage_db_path)
    Path(raw_stage_db_path).parent.mkdir(parents=True, exist_ok=True)

    # 1) Load the S&P 500 list
    try:
        tickers = _get_sp500_tickers()
        if not tickers:
            logger.error("Failed to retrieve the S&P 500 list. Aborting.")
            return False
    except Exception as e:
        logger.error("Failed reading S&P 500 list: %s", e)
        return False

    # 2) Ensure staging tables exist
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
        logger.error("Failed to initialise staging DB: %s", e)
        return False

    # 3) Download and save (supports concurrency and batching)
    success = True
    for idx, tk in enumerate(tickers, 1):
        logger.info("(%d/%d) Downloading financials for %s …", idx, len(tickers), tk)
        fin_data = _fetch_fin_statements_yf(tk)
        if not fin_data:
            logger.warning("Skipping %s – no data", tk)
            success = False
            continue

        # ------------------------------------------------------------------
        # Apply completeness filtering (horizontal + vertical) to quarterly dfs
        # ------------------------------------------------------------------
        for key in ("q_incs", "q_bals", "q_cfs"):
            df_q = fin_data.get(key)
            if isinstance(df_q, pd.DataFrame) and not df_q.empty:
                df_q_filt, ok = _filter_latest_quarter(
                    df_q,
                    horiz_thresh=horiz_thresh,
                    vert_thresh=vert_thresh,
                    history_qtrs=history_qtrs,
                )
                fin_data[key] = df_q_filt
                if not ok:
                    logger.info("%s latest quarter incomplete for %s – row dropped", key, tk)

        # Write into DB using the existing helper
        try:
            with sqlite3.connect(raw_stage_db_path) as conn:
                for key, value in fin_data.items():
                    tbl = stage_table_map[key]
                    _save_single_ticker_data_to_stage(conn, tk, tbl, value)
                conn.commit()
        except Exception as e:
            logger.error("写入 %s 财报到 staging 失败: %s", tk, e, exc_info=True)
            success = False

        time.sleep(0.2)  # small throttle to avoid bans

    if success:
        logger.info("--- Phase 1 complete: all data written successfully (%s) ---", raw_stage_db_path)
    else:
        logger.warning("--- Phase 1 finished with some download/write errors; check logs ---")
    return success


# =============================================================================
# Convenience – export finance DB data for a single ticker to Excel
# =============================================================================
def export_finance_data_to_excel(
    ticker: str,
    output_path: str | None = None,
    db_path: str | None = None,
    config_file: str = "config.ini",
) -> str:
    """
    Export **all** rows for ``ticker`` from every table inside the finance
    database (usually *SP500_raw_finance.db*) into a single Excel workbook.
    Each SQLite table is written to its own sheet (Excel's 31‑character limit
    per sheet name is respected).

    Parameters
    ----------
    ticker : str
        Stock symbol to export (e.g. ``"NVDA"``).
    output_path : str | None, default None
        Destination .xlsx path.  If *None*, the file will be created in the
        project root as ``"{ticker}_raw_finance.xlsx"``.
    db_path : str | None, default None
        Path to the SQLite finance DB.  If *None*, the value is taken from
        the ``[database] finance_db`` key in *config.ini*, falling back to
        ``"SP500_raw_finance.db"`` under the project root.
    config_file : str, default "config.ini"
        Config file used to resolve ``db_path`` when *db_path* is *None*.

    Returns
    -------
    str
        Absolute path of the created Excel workbook.
    """
    logger.info("=== Exporting finance DB for %s ===", ticker)
    cfg = _load_cfg(config_file)

    # ------------------------------------------------------------------ #
    # Resolve finance DB path & output file path
    # ------------------------------------------------------------------ #
    if db_path is None:
        db_path = cfg.get("database", "finance_db", fallback="SP500_raw_finance.db")
    db_path = _abs_path(db_path)

    if output_path is None:
        output_path = f"{ticker}_raw_finance.xlsx"
    output_path = _abs_path(output_path)

    # ------------------------------------------------------------------ #
    # Iterate over every SQLite table and export rows for this ticker
    # ------------------------------------------------------------------ #
    exported_cnt = 0
    with sqlite3.connect(db_path) as conn, pd.ExcelWriter(output_path) as writer:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        table_names = [row[0] for row in cur.fetchall()]

        for tbl in table_names:
            # ensure the table actually has a 'ticker' column
            cur.execute(f"PRAGMA table_info({tbl})")
            cols = [info[1] for info in cur.fetchall()]
            if "ticker" not in cols:
                continue  # skip tables without ticker segregation

            df = pd.read_sql_query(
                f"SELECT * FROM {tbl} WHERE ticker = ?",
                conn,
                params=(ticker,),
            )
            if df.empty:
                continue  # nothing to write for this table

            # defensive: drop duplicate columns if any
            df = df.loc[:, ~df.columns.duplicated()]

            # Write DataFrame – sheet name capped at 31 chars for Excel
            df.to_excel(writer, sheet_name=tbl[:31], index=False)
            logger.debug("  • %s → %d rows", tbl, len(df))
            exported_cnt += 1

    if exported_cnt == 0:
        logger.warning(
            "No rows found for ticker %s in finance DB (%s)",
            ticker,
            db_path,
        )
    else:
        logger.info(
            "Finished exporting %d table(s) for %s → %s",
            exported_cnt,
            ticker,
            output_path,
        )

    return output_path