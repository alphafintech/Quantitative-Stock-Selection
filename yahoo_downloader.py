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
        cfg.read(cfg_path)
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
                            # 抛出异常让上层知道具体缺失的 ticker
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
    # ---- ensure absolute path ------------------------------------------------
    raw_stage_db_path = _abs_path(raw_stage_db_path)
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