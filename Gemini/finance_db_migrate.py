import sqlite3
import os
import configparser
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import json
from io import StringIO
from typing import List, Any, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants for target table schemas ---
ANNUAL_FINANCIALS_TABLE = "annual_financials"
QUARTERLY_FINANCIALS_TABLE = "quarterly_financials"

# Target columns for the annual_financials and quarterly_financials tables
# These match the schema expected by Compute_growth_score_sp500.py
TARGET_FINANCIAL_COLS = [
    "ticker", "period", "revenue", "net_income", "eps", "op_income",
    "equity", "total_debt", "ocf", "capex", "ebit", "interest_exp"
]

CREATE_ANNUAL_SQL = f"""
CREATE TABLE IF NOT EXISTS {ANNUAL_FINANCIALS_TABLE} (
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,
    revenue REAL, net_income REAL, eps REAL, op_income REAL,
    equity REAL, total_debt REAL, ocf REAL, capex REAL,
    ebit REAL, interest_exp REAL,
    PRIMARY KEY (ticker, period)
);
"""

CREATE_QUARTERLY_SQL = f"""
CREATE TABLE IF NOT EXISTS {QUARTERLY_FINANCIALS_TABLE} (
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,
    revenue REAL, net_income REAL, eps REAL, op_income REAL,
    equity REAL, total_debt REAL, ocf REAL, capex REAL,
    ebit REAL, interest_exp REAL,
    PRIMARY KEY (ticker, period)
);
"""

# --- Helper functions ---
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
    
    dtype_for_series = type(default) if default is not np.nan else float
    result_series = pd.Series(default, index=target_idx, dtype=dtype_for_series)

    for k in keys:
        if k in df.columns:
            source_series = df[k]
            source_series_numeric = pd.to_numeric(source_series, errors='coerce')
            aligned_series = source_series_numeric.reindex(target_idx)
            return aligned_series.fillna(default)

    norm_map = {_norm(c): c for c in df.columns}
    for k in keys:
        nk = _norm(k)
        if nk in norm_map:
            source_series = df[norm_map[nk]]
            source_series_numeric = pd.to_numeric(source_series, errors='coerce')
            aligned_series = source_series_numeric.reindex(target_idx)
            return aligned_series.fillna(default)
            
    norm_keys = [_norm(k) for k in keys]
    for col_name in df.columns:
        nc = _norm(col_name)
        if any(nk in nc or nc in nk for nk in norm_keys):
            source_series = df[col_name]
            source_series_numeric = pd.to_numeric(source_series, errors='coerce')
            aligned_series = source_series_numeric.reindex(target_idx)
            return aligned_series.fillna(default)

    return result_series

def _get_sp500_tickers_from_source(source_conn: sqlite3.Connection, 
                                   sample_staging_table: str = "stg_quarterly_income") -> List[str]:
    """Gets a list of unique tickers from a sample table in the staging database."""
    try:
        cursor = source_conn.execute(f"SELECT DISTINCT ticker FROM {sample_staging_table}")
        tickers = [row[0] for row in cursor.fetchall()]
        if not tickers:
            logger.warning(f"No tickers found in staging table {sample_staging_table}. Migration might be empty.")
        return tickers
    except sqlite3.Error as e:
        logger.error(f"Error fetching tickers from staging table {sample_staging_table}: {e}")
        return []

def _load_staged_json_to_df(stage_cursor: sqlite3.Cursor, ticker_symbol: str, table_name: str) -> pd.DataFrame:
    """Loads and deserializes JSON data for a ticker from a staging table into a DataFrame."""
    stage_cursor.execute(f"SELECT data_json FROM {table_name} WHERE ticker=?", (ticker_symbol,))
    row = stage_cursor.fetchone()
    if row and row[0]:
        try:
            df = pd.read_json(StringIO(row[0]), orient="split")
            # Ensure DatetimeIndex and handle duplicates
            if not isinstance(df.index, pd.DatetimeIndex):
                original_index_type = type(df.index)
                df_copy_for_index_check = df.copy()
                try: 
                    df_copy_for_index_check.index = pd.to_datetime(df_copy_for_index_check.index, errors='coerce')
                    if not df_copy_for_index_check.index.isna().all() and isinstance(df_copy_for_index_check.index, pd.DatetimeIndex):
                        df = df_copy_for_index_check
                except Exception: pass
                if not isinstance(df.index, pd.DatetimeIndex):
                    date_col_candidates = ['asOfDate', 'reportDate', 'endDate']
                    actual_date_col = next((col for col in date_col_candidates if col in df.columns), None)
                    if actual_date_col:
                        try:
                            df_copy_set_index = df.copy()
                            df_copy_set_index[actual_date_col] = pd.to_datetime(df_copy_set_index[actual_date_col], errors='coerce')
                            df_copy_set_index = df_copy_set_index.set_index(actual_date_col)
                            if isinstance(df_copy_set_index.index, pd.DatetimeIndex) and not df_copy_set_index.index.isna().all():
                                df = df_copy_set_index
                        except Exception: pass
            
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.has_duplicates:
                    logger.debug(f"Duplicate dates found in index for {ticker_symbol}, {table_name}. Keeping first.")
                    df = df[~df.index.duplicated(keep='first')]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None) # Ensure timezone naive
            else:
                 logger.warning(f"Could not reliably convert index to DatetimeIndex for {ticker_symbol} from {table_name}. Final index type: {type(df.index)}")
                 return pd.DataFrame() # Return empty if index is not datetime, as it's crucial
            return df
        except Exception as e_load:
            logger.error(f"Error deserializing data for {ticker_symbol} from {table_name}: {e_load}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- Main Migration Logic ---
def migrate_connection(target_conn: sqlite3.Connection, source_stage_db_path: str) -> bool:
    """
    Migrates data from the source staging database (JSON format) to the target
    database's annual_financials and quarterly_financials tables.
    """
    logger.info(f"Starting migration from staging DB: {source_stage_db_path}")
    target_cursor = target_conn.cursor()

    try:
        target_cursor.execute(CREATE_ANNUAL_SQL)
        target_cursor.execute(CREATE_QUARTERLY_SQL)
        target_conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error creating target tables: {e}")
        target_conn.rollback()
        return False

    stage_table_map = {
        "q_incs": "stg_quarterly_income", "q_bals": "stg_quarterly_balance", "q_cfs": "stg_quarterly_cashflow",
        "a_incs": "stg_annual_income",   "a_bals": "stg_annual_balance",   "a_cfs": "stg_annual_cashflow",
    }

    all_annual_dfs = []
    all_quarterly_dfs = []

    try:
        with sqlite3.connect(source_stage_db_path) as stage_conn:
            stage_cursor = stage_conn.cursor()
            tickers = _get_sp500_tickers_from_source(stage_conn) 

            if not tickers:
                logger.info("No tickers to process from staging database.")
                return True 

            for ticker_idx, ticker in enumerate(tickers):
                logger.info(f"Processing ticker {ticker} ({ticker_idx + 1}/{len(tickers)}) for migration.")

                q_inc_df = _load_staged_json_to_df(stage_cursor, ticker, stage_table_map["q_incs"])
                q_bal_df = _load_staged_json_to_df(stage_cursor, ticker, stage_table_map["q_bals"])
                q_cf_df  = _load_staged_json_to_df(stage_cursor, ticker, stage_table_map["q_cfs"])

                if all(isinstance(df.index, pd.DatetimeIndex) and not df.empty for df in [q_inc_df, q_bal_df, q_cf_df]):
                    q_all_indices = pd.DatetimeIndex([]).union(q_inc_df.index).union(q_bal_df.index).union(q_cf_df.index)
                    if not q_all_indices.empty:
                        q_inc_r = q_inc_df.reindex(q_all_indices)
                        q_bal_r = q_bal_df.reindex(q_all_indices)
                        q_cf_r  = q_cf_df.reindex(q_all_indices)

                        quarterly_out_df = pd.DataFrame(index=q_all_indices)
                        quarterly_out_df["ticker"] = ticker
                        quarterly_out_df["period"] = q_all_indices.strftime('%Y-%m-%d')
                        
                        quarterly_out_df["revenue"] = _first_available(q_inc_r, ["TotalRevenue"])
                        quarterly_out_df["net_income"] = _first_available(q_inc_r, ["NetIncomeContinuousOperations", "NetIncome", "NetIncomeCommonStockholders"])
                        quarterly_out_df["eps"] = _first_available(q_inc_r, ["DilutedEPS", "BasicEPS"])
                        quarterly_out_df["op_income"] = _first_available(q_inc_r, ["OperatingIncome"])
                        quarterly_out_df["ebit"] = _first_available(q_inc_r, ["OperatingIncome"]) 
                        quarterly_out_df["interest_exp"] = _first_available(q_inc_r, ["InterestExpense"], default=0.0)
                        
                        total_assets = _first_available(q_bal_r, ["TotalAssets"])
                        total_liabilities = _first_available(q_bal_r, ["TotalLiabilitiesNetMinorityInterest", "TotalLiab"])
                        quarterly_out_df["equity"] = pd.to_numeric(total_assets, errors='coerce') - pd.to_numeric(total_liabilities, errors='coerce')
                        quarterly_out_df["total_debt"] = _first_available(q_bal_r, ["TotalDebt"], default=0.0)
                        
                        quarterly_out_df["ocf"] = _first_available(q_cf_r, ["OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"])
                        quarterly_out_df["capex"] = _first_available(q_cf_r, ["CapitalExpenditure", "PurchaseOfPPE", "NetPPEPurchaseAndSale"], default=0.0)
                        
                        all_quarterly_dfs.append(quarterly_out_df.reset_index(drop=True)[TARGET_FINANCIAL_COLS])
                else:
                    logger.warning(f"Skipping quarterly data for {ticker} due to missing or invalid (non-DatetimeIndex) statement(s).")

                a_inc_df = _load_staged_json_to_df(stage_cursor, ticker, stage_table_map["a_incs"])
                a_bal_df = _load_staged_json_to_df(stage_cursor, ticker, stage_table_map["a_bals"])
                a_cf_df  = _load_staged_json_to_df(stage_cursor, ticker, stage_table_map["a_cfs"])

                if all(isinstance(df.index, pd.DatetimeIndex) and not df.empty for df in [a_inc_df, a_bal_df, a_cf_df]):
                    a_all_indices = pd.DatetimeIndex([]).union(a_inc_df.index).union(a_bal_df.index).union(a_cf_df.index)
                    if not a_all_indices.empty:
                        a_inc_r = a_inc_df.reindex(a_all_indices)
                        a_bal_r = a_bal_df.reindex(a_all_indices)
                        a_cf_r  = a_cf_df.reindex(a_all_indices)

                        annual_out_df = pd.DataFrame(index=a_all_indices)
                        annual_out_df["ticker"] = ticker
                        annual_out_df["period"] = a_all_indices.year.astype(str)

                        annual_out_df["revenue"] = _first_available(a_inc_r, ["TotalRevenue"])
                        annual_out_df["net_income"] = _first_available(a_inc_r, ["NetIncomeContinuousOperations", "NetIncome", "NetIncomeCommonStockholders"])
                        annual_out_df["eps"] = _first_available(a_inc_r, ["DilutedEPS", "BasicEPS"])
                        annual_out_df["op_income"] = _first_available(a_inc_r, ["OperatingIncome"])
                        annual_out_df["ebit"] = _first_available(a_inc_r, ["OperatingIncome"])
                        annual_out_df["interest_exp"] = _first_available(a_inc_r, ["InterestExpense"], default=0.0)

                        total_assets_a = _first_available(a_bal_r, ["TotalAssets"])
                        total_liabilities_a = _first_available(a_bal_r, ["TotalLiabilitiesNetMinorityInterest", "TotalLiab"])
                        annual_out_df["equity"] = pd.to_numeric(total_assets_a, errors='coerce') - pd.to_numeric(total_liabilities_a, errors='coerce')
                        annual_out_df["total_debt"] = _first_available(a_bal_r, ["TotalDebt"], default=0.0)
                        
                        annual_out_df["ocf"] = _first_available(a_cf_r, ["OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"])
                        annual_out_df["capex"] = _first_available(a_cf_r, ["CapitalExpenditure", "PurchaseOfPPE", "NetPPEPurchaseAndSale"], default=0.0)
                        
                        all_annual_dfs.append(annual_out_df.reset_index(drop=True)[TARGET_FINANCIAL_COLS])
                else:
                    logger.warning(f"Skipping annual data for {ticker} due to missing or invalid (non-DatetimeIndex) statement(s).")

            if all_annual_dfs:
                final_annual_df = pd.concat(all_annual_dfs)
                for col in TARGET_FINANCIAL_COLS:
                    if col not in ["ticker", "period"]:
                        final_annual_df[col] = pd.to_numeric(final_annual_df[col], errors='coerce')
                
                target_cursor.execute(f"DELETE FROM {ANNUAL_FINANCIALS_TABLE}") 
                final_annual_df.to_sql(ANNUAL_FINANCIALS_TABLE, target_conn, if_exists="append", index=False)
                logger.info(f"Saved {len(final_annual_df)} records to {ANNUAL_FINANCIALS_TABLE}")

            if all_quarterly_dfs:
                final_quarterly_df = pd.concat(all_quarterly_dfs)
                for col in TARGET_FINANCIAL_COLS:
                    if col not in ["ticker", "period"]:
                        final_quarterly_df[col] = pd.to_numeric(final_quarterly_df[col], errors='coerce')

                target_cursor.execute(f"DELETE FROM {QUARTERLY_FINANCIALS_TABLE}") 
                final_quarterly_df.to_sql(QUARTERLY_FINANCIALS_TABLE, target_conn, if_exists="append", index=False)
                logger.info(f"Saved {len(final_quarterly_df)} records to {QUARTERLY_FINANCIALS_TABLE}")
            
            target_conn.commit()
            logger.info("Migration from staging DB completed successfully.")
            return True

    except sqlite3.Error as e:
        logger.error(f"SQLite error during migration: {e}")
        if target_conn: target_conn.rollback() # Ensure rollback on target_conn
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}", exc_info=True)
        if target_conn: target_conn.rollback() # Ensure rollback on target_conn
    
    return False


def _get_db_paths_from_config(config_file_path: str) -> Tuple[str | None, str | None]:
    """Reads staging and final finance DB paths from a given config file path."""
    parser = configparser.ConfigParser()
    if not os.path.exists(config_file_path):
        logger.error(f"Config file not found at: {config_file_path}")
        return None, None
        
    parser.read(config_file_path)
    
    config_dir = Path(config_file_path).resolve().parent

    raw_stage_db_path = parser.get("database", "raw_stage_db", fallback="data/SP500_raw_stage.db")
    final_finance_db_path = parser.get("database", "finance_db", fallback="data/SP500_finance_data_final.db")

    if not Path(raw_stage_db_path).is_absolute():
        raw_stage_db_path = str(config_dir / raw_stage_db_path)
    if not Path(final_finance_db_path).is_absolute():
        final_finance_db_path = str(config_dir / final_finance_db_path)
        
    return raw_stage_db_path, final_finance_db_path


def migrate_staged_db_to_final(config_file: str) -> bool:
    """
    Orchestrates the migration from the staged raw database to the final
    structured financial database.
    The config_file path should be the path to the root config.ini.
    """
    raw_stage_db_path, final_finance_db_path = _get_db_paths_from_config(config_file)

    if not raw_stage_db_path or not Path(raw_stage_db_path).exists():
        logger.error(f"Source staging database path not found or not configured: {raw_stage_db_path}")
        return False
    if not final_finance_db_path:
        logger.error(f"Target final finance database path not configured.")
        return False
    
    # Delete the final database if it already exists, as per user request
    final_db_file = Path(final_finance_db_path)
    if final_db_file.exists():
        try:
            os.remove(final_db_file)
            logger.info(f"Successfully deleted existing final database: {final_finance_db_path}")
        except OSError as e:
            logger.error(f"Error deleting existing final database {final_finance_db_path}: {e}")
            return False # Stop if deletion fails
            
    Path(final_finance_db_path).parent.mkdir(parents=True, exist_ok=True) 

    try:
        with sqlite3.connect(final_finance_db_path) as target_conn:
            return migrate_connection(target_conn, raw_stage_db_path)
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to target database {final_finance_db_path}: {e}")
        return False


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Migrate raw staged financial data to structured summary tables.")
    parser.add_argument(
        "--config", 
        default="config.ini", 
        help="Configuration file with database paths (default: config.ini in script's parent directory if run standalone)"
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent # sub_directory
    
    if args.config == "config.ini" and not Path(args.config).is_absolute():
        # Default "config.ini" is expected in the parent directory (root)
        config_path_to_use = script_dir.parent / "config.ini"
    else:
        config_path_to_use = Path(args.config)
        if not config_path_to_use.is_absolute():
            # If a relative path is given as --config, assume it's relative to CWD
            config_path_to_use = Path(os.getcwd()) / config_path_to_use
    
    if not config_path_to_use.exists():
        logger.error(f"Configuration file not found: {config_path_to_use}")
        # Create a dummy config.ini in the root if default is used and not found
        if args.config == "config.ini" and str(config_path_to_use.name) == "config.ini":
            logger.info(f"Attempting to create a dummy config.ini at {config_path_to_use}")
            root_dir_for_dummy = config_path_to_use.parent
            dummy_config = configparser.ConfigParser()
            dummy_config["database"] = {
                "finance_db": str(Path("data") / "SP500_finance_data_final.db"), 
                "price_db": str(Path("data") / "SP500_price_data.db"),       
                "raw_stage_db": str(Path("data") / "SP500_raw_stage.db")     
            }
            dummy_config["data_download"] = {"start_date": "2010-01-01", "end_date": ""}
            
            # Ensure data directory exists relative to the dummy config's location (root)
            (root_dir_for_dummy / "data").mkdir(parents=True, exist_ok=True)

            try:
                with open(config_path_to_use, "w") as configfile:
                    dummy_config.write(configfile)
                logger.info(f"Created a dummy {config_path_to_use}. Please review paths.")
            except OSError as e:
                logger.error(f"Could not create dummy config file at {config_path_to_use}: {e}")
                return 1
        else:
            return 1 

    migrated = migrate_staged_db_to_final(str(config_path_to_use))
    if migrated:
        logger.info("Migration process completed.")
        return 0
    else:
        logger.error("Migration process failed.")
        return 1

