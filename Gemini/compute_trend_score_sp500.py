# -*- coding: utf-8 -*-
"""
Script that downloads S&P 500 price data, calculates technical indicators and
stores them in a database.

Features:
1. Incrementally download the latest historical prices and volumes from Yahoo
   Finance.
2. Compute a variety of technical indicators (moving averages, MACD, RSI,
   Bollinger Bands, stochastic, OBV, ADX, ROC, 52‑week high/low).
3. Store the latest support/resistance and overbought/oversold status.
4. All parameters are configured via ``config.ini``.
5. Data update and indicator calculation steps can be run independently via CLI.
6. Compute weighted trend scores normalized to 0‑100 and save to Excel.

Dependencies: yfinance, pandas, pandas_ta, numpy, requests, beautifulsoup4,
openpyxl. Install via ``pip install yfinance pandas pandas_ta numpy requests
beautifulsoup4 lxml openpyxl``.
"""

import argparse
import configparser
import logging
import os
import sqlite3
import time
from datetime import date
from datetime import datetime
from datetime import timedelta # Split imports

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import math
import shutil


def _get_price_db(cfg_path: str = "config.ini") -> str:
    cfg = configparser.ConfigParser()
    if os.path.exists(cfg_path):
        cfg.read(cfg_path, encoding="utf-8")
    return cfg.get("database", "price_db", fallback="SP500_price_data.db")

# Ensure openpyxl is installed for Excel export
try:
    import openpyxl
except ImportError:
    logging.warning("`openpyxl` library not found. Please install it (`pip install openpyxl`) to save trend scores to Excel.")
    # Optionally raise an error or exit if Excel export is critical
    # raise ImportError("`openpyxl` is required for Excel export.")


# --- 全局配置占位符 ---
# 将由 load_configuration() 函数填充
CONFIG = {}


# --- 辅助函数: 数据库连接 ---
def create_connection(db_file):
    """
    创建到指定 SQLite 数据库文件的连接。
    如果文件或目录不存在，会尝试创建。

    Args:
        db_file (str): 数据库文件的路径。

    Returns:
        sqlite3.Connection or None: 数据库连接对象，如果失败则返回 None。
    """
    conn = None
    try:
        # 如果提供的是相对路径，则基于项目根目录解析为绝对路径
        if not os.path.isabs(db_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            db_file = os.path.join(project_root, db_file)

        # 确保目录存在
        db_dir = os.path.dirname(db_file)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logging.info(f"已创建数据库目录: {db_dir}")

        conn = sqlite3.connect(db_file)
        logging.debug(f"成功连接到 SQLite 数据库: {db_file}")
    except sqlite3.Error as e:
        logging.error(f"连接数据库 '{db_file}' 时出错: {e}", exc_info=True)
    return conn


# --- Configuration Loading (UPDATED for Trend Score Weights) ---
def load_configuration(config_file='config.ini'):
    """Loads configuration from file and sets up logging."""
    global CONFIG
    parser = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    try:
        # Read config, ignoring potential inline comments if parser handles them
        parser.read(config_file, encoding='utf-8')
    except configparser.Error as e:
        logging.error(f"解析配置文件 '{config_file}' 时出错: {e}")
        raise

    CONFIG = {} # Reset config
    for section in parser.sections():
        CONFIG[section] = {}
        for key, val in parser.items(section):
             # Strip potential whitespace around value
            cleaned_val = val.strip() if val else ""
            CONFIG[section][key] = cleaned_val

    # --- Setup Logging ---
    log_conf = CONFIG.get('logging', {})
    log_level_str = log_conf.get('log_level', 'INFO').upper()
    log_file = log_conf.get('log_file', 'data_download.log')
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    # Clear existing handlers before adding new ones
    # Using a loop is clearer than list slicing assignment here
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Setup basic config
    log_handlers = [
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
    logging.info(f"Configuration loaded from '{config_file}'. Logging configured.")


    # --- Type Conversions and Defaults ---
    # Data Download
    dd_conf = CONFIG.setdefault('data_download', {})
    dd_conf['start_date'] = dd_conf.get('start_date') or None
    dd_conf['end_date'] = dd_conf.get('end_date') or None

    # Database
    db_conf = CONFIG.setdefault('database', {})
    db_conf['db_file'] = db_conf.get('db_file') or _get_price_db()
    db_conf['main_table'] = db_conf.get('main_table', 'stock_data')
    db_conf['latest_analysis_table'] = db_conf.get('latest_analysis_table', 'latest_analysis')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_indicator_db = os.path.join(script_dir, 'trend_analysis.db')
    db_conf['indicator_db_file'] = db_conf.get('indicator_db_file', default_indicator_db)

    # Calculation Params
    cp_conf = CONFIG.setdefault('calculation_params', {})
    try:
        # Define helper locally or outside if used elsewhere
        def get_int_list(key, default_str):
            raw = cp_conf.get(key, default_str)
            items = []
            if raw:
                 items = [int(w.strip()) for w in raw.split(',') if w.strip()]
            return items

        cp_conf['ma_windows'] = get_int_list('ma_windows', '10,20,50,150,200')
        # Ensure 50 and 200 are calculated if MA score needs them
        # Separate checks for clarity
        if 50 not in cp_conf['ma_windows']:
            cp_conf['ma_windows'].append(50)
        if 200 not in cp_conf['ma_windows']:
            cp_conf['ma_windows'].append(200)
        # Unique and sorted
        cp_conf['ma_windows'] = sorted(list(set(cp_conf['ma_windows'])))

        cp_conf['volume_ma_windows'] = get_int_list('volume_ma_windows', '10,20')
        # Ensure 20 is calculated if OBV score needs it
        if 20 not in cp_conf['volume_ma_windows']:
            cp_conf['volume_ma_windows'].append(20)
        # Unique and sorted
        cp_conf['volume_ma_windows'] = sorted(list(set(cp_conf['volume_ma_windows'])))

        # Individual param conversions
        cp_conf['days_lookback_52'] = int(cp_conf.get('days_lookback_52', 252))
        cp_conf['macd_fast'] = int(cp_conf.get('macd_fast', 12))
        cp_conf['macd_slow'] = int(cp_conf.get('macd_slow', 26))
        cp_conf['macd_signal'] = int(cp_conf.get('macd_signal', 9))
        cp_conf['rsi_length'] = int(cp_conf.get('rsi_length', 14))
        cp_conf['rsi_overbought'] = int(cp_conf.get('rsi_overbought', 70))
        cp_conf['rsi_oversold'] = int(cp_conf.get('rsi_oversold', 30))
        cp_conf['bbands_length'] = int(cp_conf.get('bbands_length', 20))
        cp_conf['bbands_std'] = float(cp_conf.get('bbands_std', 2.0))
        cp_conf['stoch_k_period'] = int(cp_conf.get('stoch_k_period', 14))
        cp_conf['stoch_d_period'] = int(cp_conf.get('stoch_d_period', 3))
        cp_conf['stoch_smooth_k'] = int(cp_conf.get('stoch_smooth_k', 3))
        cp_conf['stoch_overbought'] = int(cp_conf.get('stoch_overbought', 80))
        cp_conf['stoch_oversold'] = int(cp_conf.get('stoch_oversold', 20))
        cp_conf['adx_period'] = int(cp_conf.get('adx_period', 14))
        cp_conf['roc_period'] = int(cp_conf.get('roc_period', 14))
        cp_conf['support_resistance_lookback'] = int(cp_conf.get('support_resistance_lookback', 30))
        cp_conf['obv_sma_period'] = int(cp_conf.get('obv_sma_period', 20))

    except ValueError as e:
        logging.error(f"Invalid numerical value in [calculation_params]: {e}.")
        raise

    # --- Load Trend Score Calculation Config ---
    trend_conf = CONFIG.setdefault('Calculate_trend', {})
    try:
        # Load Output Excel Filename
        trend_conf['output_excel_file'] = trend_conf.get('output_excel_file', 'sp500_trend_scores_weighted.xlsx')

        # Load SCORE WEIGHTS with defaults from analyst recommendation
        score_weights = {}
        score_weights['w_ma'] = float(trend_conf.get('w_ma', 0.30))
        score_weights['w_adx'] = float(trend_conf.get('w_adx', 0.30))
        score_weights['w_macd'] = float(trend_conf.get('w_macd', 0.20))
        score_weights['w_rsi'] = float(trend_conf.get('w_rsi', 0.10))
        score_weights['w_obv'] = float(trend_conf.get('w_obv', 0.10)) # New weight for OBV
        trend_conf['SCORE_WEIGHTS'] = score_weights # Store weights in config dict

        # Validate score weights sum
        total_score_weight = sum(score_weights.values())
        is_close_to_one = math.isclose(total_score_weight, 1.0, abs_tol=0.01)
        if not is_close_to_one:
            logging.warning(f"Trend score weights (w_ma, w_adx, ...) in config sum to {total_score_weight:.3f}, not 1.0. Normalization might be affected or results skewed.")
        else:
            logging.info(f"Loaded trend score weights: MA={score_weights['w_ma']:.2f}, ADX={score_weights['w_adx']:.2f}, MACD={score_weights['w_macd']:.2f}, RSI={score_weights['w_rsi']:.2f}, OBV={score_weights['w_obv']:.2f}")

    except ValueError as e:
        # Catch specific error if a weight cannot be converted to float
        logging.error(f"Invalid numerical value for a weight in [Calculate_trend] section: {e}. Please check config.ini.")
        raise # Re-raise to stop execution
    # Allow KeyError to propagate if section is missing essential keys not handled by .get()

    # Performance
    perf_conf = CONFIG.setdefault('performance', {})
    perf_conf['update_batch_size'] = int(perf_conf.get('update_batch_size', 5000))
    perf_conf['download_delay'] = float(perf_conf.get('download_delay', 0.15))

    # --- Generate Column Names ---
    CONFIG.setdefault('generated_columns', {})
    gc = CONFIG['generated_columns']
    cp = cp_conf # Alias for convenience

    # MA Columns
    gc['ma_cols'] = {window: f'MA{window}' for window in cp['ma_windows']}
    gc['ma50'] = gc['ma_cols'].get(50)
    gc['ma200'] = gc['ma_cols'].get(200)
    if not gc['ma50'] or not gc['ma200']:
        logging.error("MA50 or MA200 column name is missing. Ensure 50 and 200 are in ma_windows for trend scoring.")

    # Volume MA Columns
    gc['vol_ma_cols'] = {window: f'Volume_MA_{window}' for window in cp['volume_ma_windows']}
    gc['obv_sma'] = f'OBV_SMA_{cp["obv_sma_period"]}' # Name for OBV SMA

    # Other Indicator Columns
    gc['macd'] = f'MACD_{cp["macd_fast"]}_{cp["macd_slow"]}_{cp["macd_signal"]}'
    gc['macd_signal'] = f'MACDs_{cp["macd_fast"]}_{cp["macd_slow"]}_{cp["macd_signal"]}'
    gc['macd_hist'] = f'MACDh_{cp["macd_fast"]}_{cp["macd_slow"]}_{cp["macd_signal"]}'
    gc['adx'] = f'ADX_{cp["adx_period"]}'
    gc['dmp'] = f'DMP_{cp["adx_period"]}' # +DI
    gc['dmn'] = f'DMN_{cp["adx_period"]}' # -DI
    gc['rsi'] = f'RSI_{cp["rsi_length"]}'
    gc['roc'] = f'ROC_{cp["roc_period"]}'
    gc['obv'] = 'OBV'

    # Bollinger Bands
    bb_std_str_sql = str(cp['bbands_std']).replace('.', '_')
    gc['bb_lower'] = f'BBL_{cp["bbands_length"]}_{bb_std_str_sql}'
    gc['bb_middle'] = f'BBM_{cp["bbands_length"]}_{bb_std_str_sql}'
    gc['bb_upper'] = f'BBU_{cp["bbands_length"]}_{bb_std_str_sql}'
    # Stochastic
    gc['stoch_k'] = f'Stoch_K_{cp["stoch_k_period"]}_{cp["stoch_d_period"]}_{cp["stoch_smooth_k"]}'
    gc['stoch_d'] = f'Stoch_D_{cp["stoch_k_period"]}_{cp["stoch_d_period"]}_{cp["stoch_smooth_k"]}'
    # 52 Week High/Low
    gc['high_52w'] = 'High_52w'
    gc['low_52w'] = 'Low_52w'

    # List of all columns calculated and stored in the main history table
    # Break down list creation for clarity
    hist_cols = []
    hist_cols.extend(list(gc['ma_cols'].values()))
    hist_cols.extend(list(gc['vol_ma_cols'].values()))
    hist_cols.extend([
         gc['macd'], gc['macd_signal'], gc['macd_hist'], gc['rsi'],
         gc['bb_lower'], gc['bb_middle'], gc['bb_upper'],
         gc['stoch_k'], gc['stoch_d'], gc['obv'],
         gc['adx'], gc['dmp'], gc['dmn'],
         gc['high_52w'], gc['low_52w'], gc['roc']
    ])
    # Unique and sorted
    gc['all_hist_indicator_cols'] = sorted(list(set(c for c in hist_cols if c)))

    # logging.debug(f"Configuration loaded: {CONFIG}") # Can be very verbose
    return CONFIG


# --- 数据库模式管理 ---
def create_tables(conn):
    """如果数据库表不存在，则创建它们。"""
    main_table = CONFIG['database']['main_table']
    latest_analysis_table = CONFIG['database']['latest_analysis_table']
    cursor = conn.cursor()
    try:
        # Use triple quotes for potentially multi-line SQL or complex statements
        sql_main = f'''CREATE TABLE IF NOT EXISTS "{main_table}" (
                        Ticker TEXT NOT NULL,
                        Date TEXT NOT NULL,
                        Open REAL,
                        High REAL,
                        Low REAL,
                        Close REAL,
                        AdjClose REAL,
                        Volume INTEGER,
                        PRIMARY KEY (Ticker, Date)
                     )'''
        cursor.execute(sql_main)

        sql_latest = f'''CREATE TABLE IF NOT EXISTS "{latest_analysis_table}" (
                            ticker TEXT PRIMARY KEY,
                            latest_date TEXT,
                            latest_support REAL,
                            latest_resistance REAL,
                            obos_status TEXT
                         )'''
        cursor.execute(sql_latest)
        conn.commit()
        logging.info(f"已检查/创建表: '{main_table}', '{latest_analysis_table}'")
    except sqlite3.Error as e:
        logging.error(f"创建数据库表时出错: {e}")
        conn.rollback()
        raise


def check_add_columns(conn, table_name, required_columns):
    """检查表中是否存在列，如果缺少则添加。"""
    cursor = conn.cursor()
    try:
        pragma_sql = f'PRAGMA table_info("{table_name}")'
        cursor.execute(pragma_sql)
        table_info = cursor.fetchall()
        existing_columns = [col[1] for col in table_info]

        added_columns = False
        for col_name in required_columns:
            if col_name not in existing_columns:
                try:
                    # Ensure column names with spaces or special chars are quoted
                    alter_sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{col_name}" REAL'
                    cursor.execute(alter_sql)
                    logging.info(f"已向表 '{table_name}' 添加列: '{col_name}'。")
                    added_columns = True
                except sqlite3.OperationalError as e:
                    # Log warning but continue trying other columns
                    logging.warning(f"无法向 {table_name} 添加列 '{col_name}': {e}。可能已存在或名称无效。")

        if added_columns:
            conn.commit()
            logging.info(f"表 {table_name} 的模式更改已提交。")
        return True
    except sqlite3.Error as e:
        logging.error(f"检查/添加列到 '{table_name}' 时出错: {e}")
        conn.rollback() # Rollback on error during check/add process
        return False


# --- 数据获取和存储 ---
def get_sp500_tickers():
    """从维基百科获取 S&P 500 Ticker 列表。"""
    logging.info("正在从维基百科获取 S&P 500 Ticker 列表...")
    tickers = []
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() # Check for HTTP errors
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', {'id': 'constituents'})

        if table:
            header_row = table.find('tr')
            if not header_row:
                raise ValueError("Could not find header row in constituents table.")

            headers_list = [header.text.strip() for header in header_row.find_all(['th', 'td'])]
            symbol_index = -1
            for i, h in enumerate(headers_list):
                if h.lower() == 'symbol':
                    symbol_index = i
                    break # Found the symbol column index
            if symbol_index == -1:
                raise ValueError(f"在 BS4 解析的表格中未找到 'Symbol' 列头。找到的列头: {headers_list}")

            data_rows = table.find_all('tr')[1:] # Skip header row
            for row in data_rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) > symbol_index:
                     ticker_cell = cols[symbol_index]
                     ticker_text = ticker_cell.text.strip()
                     # Handle replacements like BRK.B -> BRK-B
                     ticker = ticker_text.replace('.', '-')
                     if ticker: # Ensure ticker is not empty
                         tickers.append(ticker)
        else:
             # Fallback using pandas
             logging.warning("未找到 id='constituents' 的表格，尝试使用 pandas.read_html 后备方案。")
             try:
                 tables = pd.read_html(response.text, attrs={'id': 'constituents'})
                 sp500_table = tables[0]
             except (ImportError, ValueError): # Catch potential read_html errors or missing table
                 logging.warning("Pandas read_html by ID failed, trying to read all tables.")
                 try:
                     tables = pd.read_html(response.text)
                 except ImportError:
                      logging.error("Pandas read_html requires lxml or html5lib. Please install.")
                      return []
                 except ValueError:
                     logging.error("Pandas 无法从维基百科页面读取任何表格。")
                     return []

                 # Heuristic: find table with 'Symbol' column
                 sp500_table = None
                 for tbl in tables:
                     if 'Symbol' in tbl.columns:
                         sp500_table = tbl
                         logging.info("Found potential S&P500 table via fallback.")
                         break
                 if sp500_table is None:
                     logging.error("Pandas 后备方案读取的表格中缺少 'Symbol' 列。")
                     return []

             if 'Symbol' not in sp500_table.columns:
                 logging.error("Pandas 后备方案读取的表格中缺少 'Symbol' 列。")
                 return []

             # Extract tickers from pandas DataFrame
             raw_tickers = sp500_table['Symbol'].tolist()
             for t in raw_tickers:
                  if isinstance(t, str) and t:
                      ticker = t.replace('.', '-')
                      tickers.append(ticker)

        # Final cleanup: unique, sorted, non-empty
        unique_tickers = sorted(list(set(t for t in tickers if t)))
        if not unique_tickers:
            logging.warning("未能从维基百科获取任何有效的 Ticker。")
            return []
        else:
            logging.info(f"成功获取 {len(unique_tickers)} 个 S&P 500 Tickers。")
            return unique_tickers

    except requests.exceptions.RequestException as e:
        logging.error(f"获取 Ticker 时发生网络错误: {e}")
        return []
    except Exception as e:
        # Catch other potential errors during parsing
        logging.error(f"解析维基百科 Ticker 时出错: {e}", exc_info=True)
        return []


def get_existing_date_range(conn, ticker):
    """获取数据库中特定 Ticker 已存在的最小和最大日期。"""
    min_date = None
    max_date = None
    cursor = conn.cursor()
    main_table = CONFIG['database']['main_table']
    sql = f'SELECT MIN(Date), MAX(Date) FROM "{main_table}" WHERE Ticker = ?'
    try:
        cursor.execute(sql, (ticker,))
        result = cursor.fetchone()
        # Check if result and its elements are not None
        if result and result[0] is not None and result[1] is not None:
            min_date_str = result[0]
            max_date_str = result[1]
            # Safely parse dates
            try:
                min_date = datetime.strptime(min_date_str, '%Y-%m-%d').date()
            except (ValueError, TypeError):
                logging.warning(f"Invalid min_date format '{min_date_str}' for {ticker}")
                min_date = None # Reset on failure
            try:
                max_date = datetime.strptime(max_date_str, '%Y-%m-%d').date()
            except (ValueError, TypeError):
                logging.warning(f"Invalid max_date format '{max_date_str}' for {ticker}")
                max_date = None # Reset on failure
    except Exception as e:
        logging.error(f"查询 {ticker} 日期范围时出错: {e}")
        # Ensure min_date/max_date remain None on error
        min_date = None
        max_date = None
    return min_date, max_date


def download_stock_data(ticker, start_date=None, end_date=None):
    """Disabled network download."""
    logging.info(f"[download_stock_data] Skipped download for {ticker} – using existing DB data.")
    return None


def store_data(conn, data, ticker):
    """将下载的 DataFrame 存储到主 SQLite 数据库表中，忽略重复项。"""
    if data is None or data.empty:
        return 0 # Nothing to store

    main_table = CONFIG['database']['main_table']
    data['Ticker'] = ticker # Add Ticker column

    # Define expected DB columns and ensure correct order
    db_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    # Ensure only existing columns are selected, in the correct order
    cols_to_insert = [col for col in db_cols if col in data.columns]
    data_to_insert = data[cols_to_insert].copy()

    # Prepare SQL query dynamically based on available columns
    cols_str = ', '.join([f'"{c}"' for c in cols_to_insert]) # Quote column names
    placeholders = ', '.join(['?'] * len(cols_to_insert))
    sql = f'INSERT OR IGNORE INTO "{main_table}" ({cols_str}) VALUES ({placeholders})'

    cursor = conn.cursor()
    rows_inserted = 0
    try:
        # Convert data to tuples, handling pandas NA and numpy NaN/inf
        data_tuples = []
        # Use itertuples for potentially better performance
        for row_tuple in data_to_insert.itertuples(index=False, name=None):
            processed_row = []
            for val in row_tuple:
                # Check for pandas NA, numpy nan, numpy inf
                is_invalid_numeric = isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val))
                if pd.isna(val) or is_invalid_numeric:
                    processed_row.append(None) # Use None for database NULL
                else:
                    processed_row.append(val)
            data_tuples.append(tuple(processed_row))

        # Only execute if there's valid data to insert
        if data_tuples:
            cursor.executemany(sql, data_tuples)
            conn.commit()
            # cursor.rowcount gives rows affected by the last statement (executemany)
            rows_inserted = cursor.rowcount
            if rows_inserted > 0:
                 # Use debug level for less verbose logging on successful storage
                 logging.debug(f"Stored/Ignored {rows_inserted} rows for {ticker}.")
            # Note: rowcount might be -1 or 0 depending on SQLite version and operation type with IGNORE
    except sqlite3.Error as e:
        logging.error(f"Database error storing data for {ticker}: {e}")
        conn.rollback() # Rollback on error
    except Exception as e:
        # Catch any other unexpected errors during tuple conversion or execution
        logging.error(f"Unexpected error storing data for {ticker}: {e}")
        conn.rollback()

    return rows_inserted


# --- 主要数据更新函数 (增量) ---
def update_stock_data(conn):
    """Disabled data update step."""
    logging.info("[update_stock_data] Data update disabled – using existing database.")
    return True
def calculate_all_indicators(conn):
    """
    计算并存储所有指标 (MAs, RSI, MACD, BBands, Stoch, OBV, ADX, 52w H/L, ROC)
    以及最新的分析 (S/R, OB/OS)。为数据库中所有 Ticker 重新计算。
    """
    logging.info("--- 开始计算所有指标 ---")
    calc_start_time = time.time()

    # 获取配置参数
    main_table = CONFIG['database']['main_table']
    latest_analysis_table = CONFIG['database']['latest_analysis_table']
    cp = CONFIG['calculation_params']
    gc = CONFIG['generated_columns']
    all_hist_indicator_cols = gc['all_hist_indicator_cols']
    batch_size = CONFIG['performance']['update_batch_size']

    cursor = conn.cursor()

    # 1. 确保表和列存在
    if not check_add_columns(conn, main_table, all_hist_indicator_cols):
        logging.error("未能验证/添加列到主表。中止计算。")
        return False
    try:
         # Create latest analysis table if it doesn't exist
         sql_create_latest = f'''CREATE TABLE IF NOT EXISTS "{latest_analysis_table}" (
                                    ticker TEXT PRIMARY KEY,
                                    latest_date TEXT,
                                    latest_support REAL,
                                    latest_resistance REAL,
                                    obos_status TEXT
                                 )'''
         cursor.execute(sql_create_latest)
         conn.commit()
    except sqlite3.Error as e:
        logging.error(f"未能创建/检查 '{latest_analysis_table}': {e}")
        return False # Cannot proceed without this table

    # 2. 获取 Tickers
    try:
        sql_get_tickers = f'SELECT DISTINCT Ticker FROM "{main_table}"'
        cursor.execute(sql_get_tickers)
        tickers_result = cursor.fetchall()
        tickers = [row[0] for row in tickers_result]
        if not tickers:
            logging.warning("数据库中未找到 Ticker。指标计算已跳过。")
            return True # No error, just nothing to do
        logging.info(f"找到 {len(tickers)} 个 Ticker 用于指标计算。")
    except sqlite3.Error as e:
        logging.error(f"从数据库获取 Ticker 失败: {e}")
        return False

    # 3. 逐个处理 Ticker
    processed_count = 0
    error_tickers = []
    latest_analysis_data = [] # Accumulate latest analysis results

    total_tickers_to_process = len(tickers)
    for i, ticker in enumerate(tickers):
        # Log progress periodically
        if (i + 1) % 25 == 0:
            logging.info(f"正在为 Ticker {i+1}/{total_tickers_to_process} ({ticker}) 计算指标...")

        try:
            # Load necessary historical data for this ticker.
            # The database created by ``yahoo_downloader.py`` stores the base
            # price columns in lowercase (date, open, high, ...).  Many parts of
            # this script expect the classic CamelCase column names used by
            # pandas/yfinance.  We therefore alias the lowercase DB columns back
            # to the expected names on SELECT so subsequent code continues to
            # work unchanged.

            base_aliases = [
                'date AS "Date"',
                'open AS "Open"',
                'high AS "High"',
                'low AS "Low"',
                'close AS "Close"',
                'adj_close AS "AdjClose"',
                'volume AS "Volume"',
            ]
            cols_needed_str = ", ".join(base_aliases)
            query = (
                f'SELECT {cols_needed_str} FROM "{main_table}" '
                f'WHERE ticker = ? ORDER BY date'
            )
            # Use index_col and parse_dates for efficiency
            df_ticker = pd.read_sql_query(
                query,
                conn,
                params=(ticker,),
                index_col='Date',
                parse_dates=['Date']
            )

            if df_ticker.empty:
                 logging.warning(f"Ticker {ticker} 在数据库中没有数据。跳过指标计算。")
                 continue # Skip to next ticker

            # Ensure index is timezone-naive just in case
            if df_ticker.index.tz is not None:
                df_ticker.index = df_ticker.index.tz_convert(None)

            # --- Data Preparation ---
            # Ensure columns are numeric, coercing errors
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
            for col in numeric_cols:
                if col in df_ticker.columns:
                    df_ticker[col] = pd.to_numeric(df_ticker[col], errors='coerce')
                else:
                    # Add missing essential columns as NaN
                    df_ticker[col] = np.nan

            # Drop rows where essential columns for calculation are NaN
            essential_cols = ['High', 'Low', 'Close', 'AdjClose', 'Volume']
            df_ticker.dropna(subset=essential_cols, inplace=True)
            if df_ticker.empty:
                logging.warning(f"Ticker {ticker} 清理 NaN 后无有效数据。跳过指标计算。")
                continue

            # --- Indicator Calculations using pandas_ta ---
            # MAs (using AdjClose)
            for window in cp['ma_windows']:
                col_name = gc['ma_cols'].get(window)
                if col_name: # Check if column name was generated
                    df_ticker.ta.sma(close=df_ticker['AdjClose'], length=window, append=True, col_names=col_name)

            # Volume MAs (using Volume)
            for window in cp['volume_ma_windows']:
                 col_name = gc['vol_ma_cols'].get(window)
                 if col_name:
                     df_ticker.ta.sma(close=df_ticker['Volume'], length=window, append=True, col_names=col_name)

            # MACD (using AdjClose)


            # Example for MACD
            try:
                df_ticker.ta.macd(close='AdjClose', fast=cp['macd_fast'], slow=cp['macd_slow'], signal=cp['macd_signal'], append=True, col_names=(gc['macd'], gc['macd_hist'], gc['macd_signal']))
            except Exception as e_macd:
                logging.error(f"Error calculating MACD for {ticker}: {e_macd}", exc_info=False) # Keep log concise



            # RSI (using AdjClose)
            df_ticker.ta.rsi(
                close='AdjClose', length=cp['rsi_length'],
                append=True, col_names=gc['rsi']
            )

            # Bollinger Bands (using AdjClose)
            try:
                df_ticker.ta.bbands(close='AdjClose', length=cp['bbands_length'], std=cp['bbands_std'], append=True, col_names=(gc['bb_lower'], gc['bb_middle'], gc['bb_upper']))
            except Exception as e_bbands:
                # 记录错误，包括 Ticker 和具体的异常信息
                logging.error(f"计算 BBands 时出错 - Ticker: {ticker} - 错误: {e_bbands}", exc_info=False) # exc_info=False 保持日志简洁


            # Stochastic (using High, Low, Close)
            df_ticker.ta.stoch(
                high='High', low='Low', close='Close',
                k=cp['stoch_k_period'], d=cp['stoch_d_period'], smooth_k=cp['stoch_smooth_k'],
                append=True,
                col_names=(gc['stoch_k'], gc['stoch_d'])
            )

            # OBV (using AdjClose, Volume)
            df_ticker.ta.obv(
                close='AdjClose', volume='Volume',
                append=True, col_names=gc['obv']
            )

            # ADX (using High, Low, Close)
            try:
                df_ticker.ta.adx(high='High', low='Low', close='Close', length=cp['adx_period'], append=True, col_names=(gc['adx'], gc['dmp'], gc['dmn']))
            except Exception as e_adx:
                # 记录错误，包括 Ticker 和具体的异常信息
                logging.error(f"计算 ADX 时出错 - Ticker: {ticker} - 错误: {e_adx}", exc_info=False) # exc_info=False 保持日志简洁


            # ROC (using AdjClose)
            df_ticker.ta.roc(
                close='AdjClose', length=cp['roc_period'],
                append=True, col_names=gc['roc']
            )

            # 52-Week High/Low (manual calculation)
            days_52w = cp['days_lookback_52']
            actual_52w = min(days_52w, len(df_ticker)) # Use actual available days if less than 52w
            high_52w_col = gc['high_52w']
            low_52w_col = gc['low_52w']
            if actual_52w >= 1:
                # Use min_periods=1 to get a value even if less than window size
                df_ticker[high_52w_col] = df_ticker['High'].rolling(window=actual_52w, min_periods=1).max()
                df_ticker[low_52w_col] = df_ticker['Low'].rolling(window=actual_52w, min_periods=1).min()
            else:
                # Handle case with no data rows after cleaning
                df_ticker[high_52w_col] = np.nan
                df_ticker[low_52w_col] = np.nan
            # --- Indicator calculations finished ---

            # --- Prepare Data for Database Update ---
            # Ensure index name so reset_index() produces a 'Date' column
            if df_ticker.index.name != "Date":
                df_ticker.index.name = "Date"
            df_ticker.reset_index(inplace=True)
            if "Date" not in df_ticker.columns and "index" in df_ticker.columns:
                df_ticker.rename(columns={"index": "Date"}, inplace=True)

            # Select columns that were *actually* calculated and expected to be stored
            # Use the definitive list from generated_columns config
            cols_for_update = [col for col in all_hist_indicator_cols if col in df_ticker.columns]
            if not cols_for_update:
                 logging.warning(f"没有为 {ticker} 成功生成指标列。跳过数据库更新。")
                 continue # Skip to next ticker

            df_update = df_ticker[['Date'] + cols_for_update].copy()
            # Format date string for WHERE clause compatibility
            df_update['Date'] = df_update['Date'].dt.strftime('%Y-%m-%d')

            # Prepare update tuples (handle NaN/Inf)
            update_tuples = []
            set_clause_parts = [f'"{col}" = ?' for col in cols_for_update]
            set_clause = ", ".join(set_clause_parts)
            sql_update = f'UPDATE "{main_table}" SET {set_clause} WHERE Ticker = ? AND Date = ?'

            # Iterate through rows to create tuples for executemany
            for record in df_update.itertuples(index=False, name=None):
                date_str = record[0] # Date is first element
                indicator_values = []
                # Indicator values start from index 1
                for k in range(len(cols_for_update)):
                    v = record[k + 1] # Get value corresponding to col_for_update
                    is_invalid_numeric = isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v))
                    if pd.isna(v) or is_invalid_numeric:
                        indicator_values.append(None) # Use None for DB NULL
                    else:
                        indicator_values.append(v)

                # Params order: indicator values, then Ticker, then Date for WHERE clause
                params = indicator_values + [ticker, date_str]
                update_tuples.append(tuple(params))

            # Execute DB update in batches within a transaction for efficiency
            if update_tuples:
                conn.execute("BEGIN TRANSACTION")
                try:
                    cursor = conn.cursor() # Use connection's cursor directly
                    # Split into batches
                    for j in range(0, len(update_tuples), batch_size):
                         batch = update_tuples[j:j + batch_size]
                         cursor.executemany(sql_update, batch)
                         # Debug log for batch update is very verbose, consider removing or raising level
                         # logging.debug(f"Updated batch {j // batch_size + 1} for {ticker} ({len(batch)} rows)")
                    conn.commit() # Commit transaction for this ticker
                except sqlite3.Error as e_update:
                    logging.error(f"更新 {ticker} 指标失败: {e_update}. 回滚。")
                    conn.rollback()
                    raise # Re-raise to mark ticker as failed
                except Exception as e_other:
                     logging.error(f"更新 {ticker} 指标时发生意外错误: {e_other}. 回滚。")
                     conn.rollback()
                     raise

            # --- Calculate and Store Latest Analysis ---
            # Set index back to Date for easier S/R calculation
            df_ticker.set_index('Date', inplace=True)

            # Check if enough data for lookback period
            sr_lookback_period = cp['support_resistance_lookback']
            if len(df_ticker) >= sr_lookback_period:
                # Get the latest row (already the last after chronological sort)
                latest_row = df_ticker.iloc[-1]
                # Use the required lookback period for calculation
                lookback_data = df_ticker.iloc[-sr_lookback_period:]

                # Calculate Support/Resistance
                lows = pd.to_numeric(lookback_data['Low'], errors='coerce')
                highs = pd.to_numeric(lookback_data['High'], errors='coerce')
                # Use .min()/.max() which handle NaN by default (skipna=True)
                support = lows.min()
                resistance = highs.max()
                # If all lows/highs were NaN, result will be NaN - convert to None for DB
                support = None if pd.isna(support) else support
                resistance = None if pd.isna(resistance) else resistance


                # Determine Overbought/Oversold status
                status = "Neutral"
                stoch_k_col = gc['stoch_k']
                rsi_col_name = gc['rsi'] # Reuse variable for clarity
                # Ensure column names exist before trying .get()
                latest_k = None
                if stoch_k_col in latest_row:
                     latest_k = pd.to_numeric(latest_row.get(stoch_k_col), errors='coerce')

                latest_rsi = None
                if rsi_col_name in latest_row:
                    latest_rsi = pd.to_numeric(latest_row.get(rsi_col_name), errors='coerce')

                # Check conditions safely using pd.notna
                is_stoch_ob = pd.notna(latest_k) and latest_k > cp['stoch_overbought']
                is_rsi_ob = pd.notna(latest_rsi) and latest_rsi > cp['rsi_overbought']
                is_stoch_os = pd.notna(latest_k) and latest_k < cp['stoch_oversold']
                is_rsi_os = pd.notna(latest_rsi) and latest_rsi < cp['rsi_oversold']

                if is_stoch_ob or is_rsi_ob:
                    status = "Overbought"
                elif is_stoch_os or is_rsi_os:
                    status = "Oversold"

                # Append data for bulk insert/replace later
                latest_date_str = latest_row.name.strftime('%Y-%m-%d') # Index is DatetimeIndex
                latest_analysis_data.append(
                    (ticker, latest_date_str, support, resistance, status)
                )
            else:
                 # Log if not enough data for S/R calculation
                 logging.debug(f"{ticker} 数据不足 ({len(df_ticker)} 行) 无法计算 S/R (需要 {sr_lookback_period})。")


            processed_count += 1 # Increment count for successfully processed ticker

        except pd.errors.EmptyDataError:
             # This might happen if read_sql returns empty for a ticker
             logging.warning(f"Ticker {ticker} 无数据。跳过指标计算。")
        except Exception as e_ticker:
            # Log error for this specific ticker and add to error list
            logging.error(f"处理 Ticker {ticker} 指标失败: {e_ticker}", exc_info=False) # Keep log concise
            error_tickers.append(ticker)
            # Ensure rollback if an error occurs mid-transaction for a ticker
            try:
                conn.rollback()
            except sqlite3.Error:
                # Ignore rollback error if no transaction was active
                pass

    # --- Update Latest Analysis Table (outside ticker loop) ---
    if latest_analysis_data:
        logging.info(f"更新/插入 {len(latest_analysis_data)} 行到 '{latest_analysis_table}'...")
        # Use INSERT OR REPLACE to handle existing tickers
        sql_replace = f'''INSERT OR REPLACE INTO "{latest_analysis_table}"
                           (ticker, latest_date, latest_support, latest_resistance, obos_status)
                           VALUES (?, ?, ?, ?, ?)'''
        conn.execute("BEGIN TRANSACTION")
        try:
            cursor = conn.cursor() # Get cursor for bulk operation
            cursor.executemany(sql_replace, latest_analysis_data)
            conn.commit() # Commit all analysis updates
            logging.info(f"成功更新 '{latest_analysis_table}'。")
        except sqlite3.Error as e_replace:
            logging.error(f"更新 {latest_analysis_table} 出错: {e_replace}. 回滚。")
            conn.rollback()

    # --- Indicator Calculation Summary ---
    logging.info("--- 指标计算摘要 ---")
    logging.info(f"成功处理 Ticker 数量: {processed_count}/{total_tickers_to_process}")
    unique_error_tickers = sorted(list(set(error_tickers)))
    if unique_error_tickers:
        logging.warning(f"出错 Ticker 数量: {len(unique_error_tickers)}")
        # Optionally log specific error tickers if needed, keeping list short
        log_limit = 10
        error_tickers_str = ', '.join(unique_error_tickers[:log_limit])
        if len(unique_error_tickers) > log_limit:
             error_tickers_str += f", ... ({len(unique_error_tickers) - log_limit} more)"
        logging.warning(f"出错 Ticker 列表 (部分): {error_tickers_str}")

    calc_end_time = time.time()
    duration = calc_end_time - calc_start_time
    logging.info(f"--- 指标计算完成 (耗时: {duration:.2f} 秒) ---")
    return True # Indicate completion


# --- Trend Score Calculation Function (UPDATED for Configurable Weights) ---
def calculate_and_save_trend_scores(conn):
    """
    Calculates trend scores based on configurable weights, normalizes them
    to 0-100, and saves the results to an Excel file specified in the config.
    """
    logging.info("--- 开始计算趋势分数 (使用可配置权重和归一化) ---")
    score_start_time = time.time()

    # --- Get Configuration ---
    try:
        db_conf = CONFIG['database']
        cp_conf = CONFIG['calculation_params']
        gc = CONFIG['generated_columns']
        trend_conf = CONFIG['Calculate_trend']
        # Retrieve score weights (loaded by load_configuration with defaults)
        score_weights = trend_conf.get('SCORE_WEIGHTS')
        if score_weights is None:
            logging.error("趋势分数权重未在配置中找到。请检查 load_configuration 函数。")
            return False
        output_filename = trend_conf['output_excel_file'] # Already has default
        main_table = db_conf['main_table']
        obv_sma_period = cp_conf['obv_sma_period'] # Already has default
    except KeyError as e:
        logging.error(f"配置错误：缺少必要的键 {e}。请检查 config.ini 和 load_configuration。")
        return False

    # Construct full output path
    script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    output_path = os.path.join(script_dir, output_filename)

    # --- Get Tickers ---
    cursor = conn.cursor()
    try:
        sql_get_tickers = f'SELECT DISTINCT Ticker FROM "{main_table}"'
        cursor.execute(sql_get_tickers)
        tickers_result = cursor.fetchall()
        tickers = [row[0] for row in tickers_result]
        if not tickers:
            logging.warning("数据库中未找到 Ticker。无法计算分数。")
            return True
        logging.info(f"找到 {len(tickers)} 个 Ticker 用于分数计算。")
    except sqlite3.Error as e:
        logging.error(f"从数据库获取 Ticker 失败: {e}")
        return False

    # --- Define Required Columns for Scoring ---
    ma50_col = gc.get('ma50')
    ma200_col = gc.get('ma200')
    macd_col = gc.get('macd')
    macds_col = gc.get('macd_signal')
    adx_col = gc.get('adx')
    dmp_col = gc.get('dmp') # +DI
    dmn_col = gc.get('dmn') # -DI
    rsi_col = gc.get('rsi')
    obv_col = gc.get('obv')
    # Base required columns
    required_cols_base = ['Date', 'AdjClose', obv_col]
    # Add optional columns only if they exist (based on config generation)
    optional_cols = [ma50_col, ma200_col, macd_col, macds_col, adx_col, dmp_col, dmn_col, rsi_col]
    required_cols = required_cols_base + [c for c in optional_cols if c]

    # Build SELECT list.  Base price columns come from the DB in lowercase, so
    # alias them back to CamelCase to keep the rest of the code (which expects
    # 'Date'/'AdjClose') functioning without change.
    base_aliases = ['date AS "Date"', 'adj_close AS "AdjClose"']
    required_cols_str = ", ".join(base_aliases + [f'"{c}"' for c in required_cols[2:]])

    # --- Iterate and Score ---
    trend_scores_data = []
    processed_count = 0
    error_tickers_score = []
    total_tickers_to_score = len(tickers)

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logging.info(f"正在计算分数 Ticker {i+1}/{total_tickers_to_score} ({ticker})...")
        try:
            # Fetch enough data for OBV SMA calculation
            # Ensure enough lookback for the rolling window
            fetch_limit = obv_sma_period + 20 # Add buffer

            query = (
                f'SELECT {required_cols_str} FROM "{main_table}" '
                f'WHERE ticker = ? ORDER BY date DESC LIMIT {fetch_limit}'
            )
            df_ticker = pd.read_sql_query(
                query,
                conn,
                params=(ticker,),
                parse_dates=['Date']
            )

            # Need at least obv_sma_period days for the rolling calculation
            if df_ticker.empty or len(df_ticker) < obv_sma_period:
                 logging.warning(f"数据不足 ({len(df_ticker)} 行, 需要 {obv_sma_period}) 用于 {ticker} 计算 OBV SMA。跳过分数计算。")
                 continue

            # Reverse to get chronological order for rolling calculations
            df_ticker = df_ticker.iloc[::-1].reset_index()
            if "Date" not in df_ticker.columns and "index" in df_ticker.columns:
                df_ticker.rename(columns={"index": "Date"}, inplace=True)

            # Calculate OBV SMA
            obv_sma_col_name = gc['obv_sma']
            if obv_col in df_ticker.columns:
                 # Calculate rolling mean - result will have NaNs at the start
                 df_ticker[obv_sma_col_name] = df_ticker[obv_col].rolling(
                     window=obv_sma_period, min_periods=obv_sma_period # Require full window
                 ).mean()
            else:
                 # OBV column itself was missing from DB fetch
                 logging.warning(f"OBV 列 ('{obv_col}') 未找到用于 {ticker}。无法计算 OBV SMA。")
                 df_ticker[obv_sma_col_name] = np.nan # Add column as NaN

            # Get the latest row (which now contains calculated OBV SMA if possible)
            latest_data = df_ticker.iloc[-1].copy()

            # --- Calculate Individual Raw Score Components ---
            # Initialize scores
            ma_score = 0
            macd_score = 0
            adx_score = 0
            rsi_score = 0
            obv_score = 0

            # Get latest indicator values safely using .get()
            latest_price = latest_data.get('AdjClose')
            latest_ma50 = latest_data.get(ma50_col)
            latest_ma200 = latest_data.get(ma200_col)
            latest_macd = latest_data.get(macd_col)
            latest_macds = latest_data.get(macds_col)
            latest_adx = latest_data.get(adx_col)
            latest_dmp = latest_data.get(dmp_col)
            latest_dmn = latest_data.get(dmn_col)
            latest_rsi = latest_data.get(rsi_col)
            latest_obv = latest_data.get(obv_col)
            latest_obv_sma = latest_data.get(obv_sma_col_name) # Get calculated OBV SMA

            # Calculate MA raw score
            # Check if all required values are not NA
            ma_check = pd.notna(latest_price) and pd.notna(latest_ma50) and pd.notna(latest_ma200)
            if ma_check:
                is_above_50 = latest_price > latest_ma50
                is_50_above_200 = latest_ma50 > latest_ma200
                if is_above_50 and is_50_above_200:
                    ma_score = 3 # Strong Uptrend
                elif is_above_50 and not is_50_above_200:
                    ma_score = 1 # Potential Recovery
                elif not is_above_50 and is_50_above_200:
                    ma_score = -1 # Potential Pullback
                elif not is_above_50 and not is_50_above_200:
                    ma_score = -3 # Strong Downtrend
            # else: ma_score remains 0 if data missing

            # Calculate MACD raw score
            macd_check = pd.notna(latest_macd) and pd.notna(latest_macds)
            if macd_check:
                is_macd_above_signal = latest_macd > latest_macds
                is_macd_positive = latest_macd > 0
                if is_macd_above_signal and is_macd_positive:
                    macd_score = 2 # Strong Bullish Momentum
                elif is_macd_above_signal and not is_macd_positive:
                    macd_score = 1 # Weakening Bearish / Potential Reversal
                elif not is_macd_above_signal and is_macd_positive:
                    macd_score = -1 # Weakening Bullish / Potential Reversal
                elif not is_macd_above_signal and not is_macd_positive:
                    macd_score = -2 # Strong Bearish Momentum
            # else: macd_score remains 0

            # Calculate ADX raw score
            adx_check = pd.notna(latest_adx) and pd.notna(latest_dmp) and pd.notna(latest_dmn)
            if adx_check:
                is_trending = latest_adx > 25
                if is_trending:
                    is_uptrend_stronger = latest_dmp > latest_dmn
                    if is_uptrend_stronger:
                        adx_score = 2 # Strong Uptrend Confirmed
                    else: # Downtrend is stronger
                        adx_score = -2 # Strong Downtrend Confirmed
                # else: adx_score remains 0 if not trending (ADX <= 25)
            # else: adx_score remains 0

            # Calculate RSI raw score
            rsi_check = pd.notna(latest_rsi)
            if rsi_check:
                if latest_rsi > 55:
                    rsi_score = 1 # Bullish Momentum Confirmation
                elif latest_rsi < 45:
                    rsi_score = -1 # Bearish Momentum Confirmation
                # else: rsi_score remains 0 if neutral (45-55)
            # else: rsi_score remains 0

            # Calculate OBV raw score
            # Check that OBV and its SMA were successfully calculated/retrieved
            obv_check = pd.notna(latest_obv) and pd.notna(latest_obv_sma)
            if obv_check:
                if latest_obv > latest_obv_sma:
                    obv_score = 1 # Volume Confirms Uptrend/Strength
                elif latest_obv < latest_obv_sma:
                    obv_score = -1 # Volume Confirms Downtrend/Weakness
                # else: obv_score remains 0 if equal
            # else: obv_score remains 0 if data missing

            # --- Calculate Weighted Raw Score using weights from config ---
            weighted_raw_score = (score_weights['w_ma'] * ma_score) + \
                                 (score_weights['w_adx'] * adx_score) + \
                                 (score_weights['w_macd'] * macd_score) + \
                                 (score_weights['w_rsi'] * rsi_score) + \
                                 (score_weights['w_obv'] * obv_score)

            # Append results for this ticker
            score_data_row = {
                'Ticker': ticker,
                'Latest_Date': latest_data['Date'].strftime('%Y-%m-%d'),
                'Weighted_Raw_Score': weighted_raw_score,
                # Optional: Include component scores for detail/debugging
                'MA_Score': ma_score,
                'MACD_Score': macd_score,
                'ADX_Score': adx_score,
                'RSI_Score': rsi_score,
                'OBV_Score': obv_score
            }
            trend_scores_data.append(score_data_row)
            processed_count += 1

        except pd.errors.EmptyDataError:
            # Catch error if read_sql query returns nothing
            logging.warning(f"Ticker {ticker} 无数据。跳过分数计算。")
        except Exception as e_score:
            logging.error(f"计算分数 {ticker} 失败: {e_score}", exc_info=False)
            error_tickers_score.append(ticker)

    # --- Prepare, Normalize, and Save Results ---
    if not trend_scores_data:
        logging.warning("没有计算出任何趋势分数。不会创建 Excel 文件。")
        return True # No data is not an error state for the function itself

    scores_df = pd.DataFrame(trend_scores_data)

    # --- Normalize the Weighted_Raw_Score to 0-100 ---
    raw_score_col = 'Weighted_Raw_Score' # Define the column to normalize
    min_score = scores_df[raw_score_col].min()
    max_score = scores_df[raw_score_col].max()

    # Check if min/max are valid numbers before normalization
    if pd.notna(min_score) and pd.notna(max_score):
        if max_score == min_score:
            # Handle case where all scores are the same -> assign neutral 50
            scores_df['Normalized_Trend_Score'] = 50.0
            logging.warning("所有计算出的加权原始分数都相同。归一化分数设置为 50。")
        else:
            # Define normalization function (can be lambda or separate def)
            def normalize(score):
                 # Handle potential NaN scores within the apply
                 if pd.isna(score):
                     return None
                 # Apply Min-Max Scaling
                 return ((score - min_score) / (max_score - min_score)) * 100

            # Apply the function
            scores_df['Normalized_Trend_Score'] = scores_df[raw_score_col].apply(normalize)
            # Round the normalized score for readability
            scores_df['Normalized_Trend_Score'] = scores_df['Normalized_Trend_Score'].round(2)
    else:
        # Handle case where min/max couldn't be determined (e.g., all NaN scores)
        scores_df['Normalized_Trend_Score'] = np.nan
        logging.warning("无法确定最小/最大加权原始分数进行归一化。")

    # --- Sort by Normalized Score Descending ---
    # Place stocks with NaN scores (e.g., due to errors) at the end
    scores_df.sort_values(
        by='Normalized_Trend_Score',
        ascending=False,
        inplace=True,
        na_position='last'
    )

    # --- Save to Excel ---
    try:
        logging.info(f"正在将 {len(scores_df)} 个归一化趋势分数保存到: {output_path}")
        # Ensure directory exists before saving
        output_dir = os.path.dirname(output_path)
        if output_dir: # Check if path includes a directory part
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.info(f"已创建目录: {output_dir}")

        # Define column order for the output Excel file
        output_columns = [
            'Ticker', 'Latest_Date', 'Normalized_Trend_Score', 'Weighted_Raw_Score',
            'MA_Score', 'MACD_Score', 'ADX_Score', 'RSI_Score', 'OBV_Score'
        ]
        # Ensure only columns that actually exist in the DataFrame are written
        output_columns_final = [col for col in output_columns if col in scores_df.columns]

        # Write to Excel using openpyxl engine
        scores_df.to_excel(
            output_path,
            index=False,
            engine='openpyxl',
            columns=output_columns_final
        )
        logging.info(f"趋势分数成功保存到 Excel 文件。")

    except ImportError:
         # Specific error if openpyxl is missing
         logging.error("无法保存到 Excel：缺少 `openpyxl` 库。请运行 `pip install openpyxl`。")
         return False # Indicate failure due to missing dependency
    except Exception as e_save:
        logging.error(f"保存趋势分数到 Excel 时出错: {e_save}")
        return False

    # --- Scoring Summary ---
    score_end_time = time.time()
    duration = score_end_time - score_start_time
    logging.info("--- 趋势分数计算摘要 ---")
    logging.info(f"成功计算 Ticker 数量: {processed_count}/{total_tickers_to_score}")
    unique_error_tickers_score = sorted(list(set(error_tickers_score)))
    if unique_error_tickers_score:
        logging.warning(f"计算分数出错 Ticker 数量: {len(unique_error_tickers_score)}")
    logging.info(f"--- 趋势分数计算完成 (耗时: {duration:.2f} 秒) ---")

    return True # Indicate successful completion


# --- 主执行块 ---
def compute_trend_score(db_path: str | None = None):
    """主函数，解析参数并执行操作。"""
    parser = argparse.ArgumentParser(description='下载、计算和分析 S&P 500 股票数据。')
    # Update help text for score action
    parser.add_argument(
        'action',
        choices=['update', 'calculate', 'score', 'all'],
        help='要执行的操作: "update" - 下载最新数据, "calculate" - 计算所有指标, "score" - 计算趋势分数(0-100, 加权)并保存到 Excel, "all" - 按顺序执行 update, calculate, score。'
    )
    parser.add_argument(
        '--config',
        default='config.ini',
        help='配置文件的路径 (默认: config.ini)'
    )
    args = parser.parse_args()

    # Load configuration early to catch errors
    try:
        load_configuration(args.config)
    except FileNotFoundError:
        # Log error specific to config file missing
        logging.error(f"错误：配置文件 '{args.config}' 未找到。")
        return # Exit if config is missing
    except Exception as e_load:
        # Catch other config loading errors
        logging.error(f"加载配置时出错: {e_load}")
        return # Exit on other loading errors

    # Determine paths for price DB (source) and indicator DB (target)
    price_db_path = CONFIG.get('database', {}).get('db_file') or _get_price_db()
    indicator_db_path = db_path or CONFIG['database'].get('indicator_db_file')
    if not indicator_db_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        indicator_db_path = os.path.join(script_dir, 'trend_analysis.db')

    # Prepare indicator database by copying from price DB if needed
    if not os.path.exists(indicator_db_path):
        try:
            if not os.path.exists(price_db_path):
                logging.error(f"价格数据库 '{price_db_path}' 不存在，无法初始化指标数据库。")
                return
            os.makedirs(os.path.dirname(indicator_db_path), exist_ok=True)
            shutil.copy2(price_db_path, indicator_db_path)
            logging.info(f"已创建指标数据库: {indicator_db_path}")
        except Exception as e:
            logging.error(f"无法创建指标数据库 '{indicator_db_path}': {e}")
            return

    CONFIG['database']['db_file'] = indicator_db_path
    db_file_path = indicator_db_path
    conn = create_connection(db_file_path)
    if conn is None:
        logging.error("无法连接到数据库。退出。")
        return

    # Track overall success across steps
    success = True
    try:
        # Always ensure tables exist before any operation
        create_tables(conn)

        # Execute actions based on arguments
        if args.action == 'update' or args.action == 'all':
            step_success = update_stock_data(conn)
            success = success and step_success # Update overall success

        # Proceed only if previous steps were successful (if 'all')
        if (args.action == 'calculate' or args.action == 'all') and success:
            step_success = calculate_all_indicators(conn)
            success = success and step_success

        # Proceed only if previous steps were successful (if 'all')
        if (args.action == 'score' or args.action == 'all') and success:
             step_success = calculate_and_save_trend_scores(conn)
             success = success and step_success

    except Exception as e_main:
        # Catch unexpected errors during the main execution flow
        logging.error(f"执行操作 '{args.action}' 时发生意外错误: {e_main}", exc_info=True)
        success = False # Mark as failed
    finally:
        # Ensure database connection is always closed
        if conn:
            conn.close()
            logging.info("数据库连接已关闭。")

    # Final status log
    if success:
         logging.info(f"操作 '{args.action}' 成功完成。")
    else:
         log_file_path = CONFIG.get('logging', {}).get('log_file', 'data_download.log')
         logging.error(f"操作 '{args.action}' 遇到错误。请检查日志文件 '{log_file_path}'。")


if __name__ == "__main__":
    compute_trend_score()
