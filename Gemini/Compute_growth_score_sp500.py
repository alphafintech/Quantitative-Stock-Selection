import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import configparser
import requests
from bs4 import BeautifulSoup
import time # <--- 确认 time 模块已导入
import logging
from datetime import datetime, timedelta
from scipy.stats import linregress
import os
import warnings
import sys


def _get_finance_db(cfg_path: str = "config.ini") -> str:
    """Return finance DB path from config or default."""
    parser = configparser.ConfigParser()
    if os.path.exists(cfg_path):
        parser.read(cfg_path)
    return parser.get("database", "finance_db", fallback="SP500_finance_data.db")

# --- Configuration & Logging Setup ---
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore yfinance/pandas future warnings
# Suppress the SettingWithCopyWarning *after* confirming the fix below works
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning, message="A value is trying to be set on a copy of a slice from a DataFrame")

from pathlib import Path
CONFIG_FILE = Path(__file__).with_name("config_finance.ini")

LOG_FORMAT = '%(asctime)s - %(levelname)s - [Growth] %(message)s' # 日志格式，添加模块前缀

# --- Logging Setup Function ---
def setup_logging(log_level_str='INFO', log_to_file=False, log_filename_base='growth_score_run'):
    """
    设置日志记录，输出到控制台和/或文件。

    Args:
        log_level_str (str): 日志级别 ('DEBUG', 'INFO', etc.).
        log_to_file (bool): 是否记录到文件.
        log_filename_base (str): 日志文件基础名.
    """
    level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger = logging.getLogger() # 获取根 logger
    # 清除之前的处理器（如果存在），防止重复记录
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close() # 关闭处理器

    logger.setLevel(level) # 在根 logger 上设置级别
    formatter = logging.Formatter(LOG_FORMAT)

    # 控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_message_parts = [f"Logging initialized. Level: {log_level_str}.", "Outputting to Console"]

    # 文件处理器 (可选)
    if log_to_file:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{log_filename_base}_{timestamp}.log"
            # 获取脚本目录，如果失败则使用当前工作目录
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError: # __file__ 未定义 (例如在交互模式下)
                script_dir = os.getcwd()
            # 将日志文件放入 'logs' 子目录
            log_dir = os.path.join(script_dir, "logs")
            if not os.path.exists(log_dir):
                 os.makedirs(log_dir)
            log_file_path = os.path.join(log_dir, log_file)

            fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            log_message_parts.append(f"and File: {log_file_path}")
        except Exception as e:
            # 如果文件日志设置失败，仅通过控制台记录错误
            ch.handle(logging.LogRecord(
                name='root', level=logging.ERROR, pathname='', lineno=0,
                msg=f"Failed to set up file logging handler to '{log_file_path}': {e}",
                args=[], exc_info=sys.exc_info(), func=''
            ))
            log_message_parts.append("(File logging failed)")

    logging.info(" ".join(log_message_parts))


# --- Configuration Loading Function ---
def load_config(filename=CONFIG_FILE):
    """从 INI 文件加载配置并返回包含类型化值的字典。"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Configuration file '{filename}' not found.")

    config = configparser.ConfigParser(interpolation=None) # interpolation=None 防止 % 问题
    config.read(filename, encoding='utf-8') # 指定编码
    logging.info(f"Configuration loaded from {filename}")

    typed_config = {}
    # 定义预期的节、键及其类型
    expected_types = {
        'General': {'sp500_list_url': str, 'output_excel_file': str, 'log_level': str, 'log_to_file': bool}, # 添加 log_to_file
        'Data': {'db_name': str, 'years_of_annual_data': int, 'incremental_download': bool, 'download_delay': float},
        'Screening': {'enable_screening': bool, 'max_debt_to_equity': float, 'min_interest_coverage': float},
        'Scoring_Weights': {'w_cagr': float, 'w_accel': float, 'w_growth_rev': float, 'w_growth_eps': float, 'w_profitability': float, 'w_efficiency': float, 'w_fcf': float,'w_eps_turnaround': float},
        'Calculation_Params': {'min_eps_for_cagr': float, 'eps_qoq_denominator_handling': str, 'eps_qoq_zero_value': float},
        'Methodology': {'ranking_method': str}
    }

    # 遍历预期的配置结构
    for section, keys_types in expected_types.items():
        if section not in config:
            # 处理可能缺失的可选节
            if section == 'Methodology':
                 logging.warning(f"Config section [{section}] not found. Using default methodology.")
                 typed_config[section] = {'ranking_method': 'overall'} # 提供默认值
                 continue
            elif section == 'General' and 'log_to_file' in keys_types: # 特别处理 log_to_file
                 typed_config.setdefault('General', {})['log_to_file'] = False # 如果 General 节存在但键缺失，则默认为 False
                 logging.warning("Config key 'log_to_file' not found in [General]. Defaulting to False.")
                 continue
            else:
                 raise ValueError(f"Missing required section [{section}] in config file '{filename}'.")

        typed_config[section] = {}
        for key, expected_type in keys_types.items():
            if not config.has_option(section, key):
                 # 处理可能缺失的可选键
                 if section == 'Methodology' and key == 'ranking_method':
                     logging.warning(f"Config key '{key}' not found in [{section}]. Defaulting to 'overall'.")
                     typed_config[section][key] = 'overall'
                     continue
                 elif section == 'General' and key == 'log_to_file':
                      typed_config[section][key] = False # 如果键缺失，则默认为 False
                      logging.warning(f"Config key '{key}' not found in section [{section}]. Defaulting to False.")
                      continue
                 else:
                     raise ValueError(f"Missing required key '{key}' in section [{section}] in config file '{filename}'.")

            try:
                # 使用 configparser 内置的类型转换方法
                if expected_type == int:
                    typed_config[section][key] = config.getint(section, key)
                elif expected_type == float:
                    # 处理可能的 'NAN' 字符串
                    raw_value = config.get(section, key).strip().upper()
                    typed_config[section][key] = np.nan if raw_value == 'NAN' else config.getfloat(section, key)
                elif expected_type == bool:
                    typed_config[section][key] = config.getboolean(section, key)
                else: # 假定为字符串类型
                    value = config.get(section, key)
                    # 如果需要，对特定键应用字符串转换（例如，小写）
                    if section == 'Methodology' and key == 'ranking_method':
                        typed_config[section][key] = value.lower()
                    elif section == 'Calculation_Params' and key == 'eps_qoq_denominator_handling':
                        typed_config[section][key] = value.lower()
                    else:
                        typed_config[section][key] = value # 否则保持原样
            except ValueError as e:
                raise ValueError(f"Error converting key '{key}' in section [{section}] to {expected_type.__name__}. Value read: '{config.get(section, key)}'. Error: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error processing config key '{key}' in section [{section}]: {e}") from e

    # --- 后加载验证 ---
    # 确保 Methodology 节存在（如果应用了默认值）
    typed_config.setdefault('Methodology', {})['ranking_method'] = typed_config.get('Methodology', {}).get('ranking_method', 'overall')

    # 验证 ranking_method 的值
    valid_ranking_methods = ['overall', 'industry']
    current_ranking_method = typed_config['Methodology']['ranking_method']
    if current_ranking_method not in valid_ranking_methods:
        logging.warning(f"Invalid value '{current_ranking_method}' for 'ranking_method'. Valid: {valid_ranking_methods}. Defaulting to 'overall'.")
        typed_config['Methodology']['ranking_method'] = 'overall'

    # 验证 eps_qoq_denominator_handling 的值
    valid_eps_handling = ['zero', 'nan'] # 允许 'nan' 作为一个选项
    current_eps_handling = typed_config.get('Calculation_Params', {}).get('eps_qoq_denominator_handling', 'zero')
    if current_eps_handling not in valid_eps_handling:
        logging.warning(f"Invalid value '{current_eps_handling}' for 'eps_qoq_denominator_handling'. Valid: {valid_eps_handling}. Defaulting to 'zero'.")
        typed_config.setdefault('Calculation_Params', {})['eps_qoq_denominator_handling'] = 'zero'

    # 验证评分权重总和约等于 1.0
    weights_cfg = typed_config.get('Scoring_Weights', {})
    fusion_weights_keys = ['w_cagr', 'w_accel']
    final_weights_keys = ['w_growth_rev', 'w_growth_eps', 'w_profitability', 'w_efficiency', 'w_fcf']

    # 检查融合权重
    if all(k in weights_cfg for k in fusion_weights_keys):
        fusion_weights_sum = weights_cfg['w_cagr'] + weights_cfg['w_accel']
        if not np.isclose(fusion_weights_sum, 1.0):
             logging.warning(f"Fusion weights (w_cagr, w_accel) sum to {fusion_weights_sum:.3f}, not 1.0. Check config.")
    else:
        logging.warning("One or more fusion weights (w_cagr, w_accel) missing in config.")

    # 检查最终维度权重
    if all(k in weights_cfg for k in final_weights_keys):
        final_weights_sum = sum(weights_cfg[w] for w in final_weights_keys)
        if not np.isclose(final_weights_sum, 1.0):
            final_weights_values = {w: weights_cfg[w] for w in final_weights_keys}
            logging.warning(f"Final dimension weights {final_weights_values} sum to {final_weights_sum:.3f}, not 1.0. Check config.")
    else:
        logging.warning("One or more final dimension weights missing in config.")

    logging.info("Configuration successfully loaded and validated.")
    return typed_config


# --- Data Fetching Functions ---
def get_sp500_tickers_and_industries(url):
    """从维基百科抓取 S&P 500 股票代码及其 GICS 行业分类。"""
    logging.info(f"Fetching S&P 500 tickers and industries from {url}")
    tickers_industries = {}
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', {'id': 'constituents'})
        if table is None:
             logging.warning("Could not find table with id='constituents'. Trying the first wikitable.")
             table = soup.find('table', {'class': 'wikitable'})
        if table is None:
             logging.error("Could not find a suitable constituents table.")
             return tickers_industries

        header_row = table.find('tr')
        if not header_row:
             logging.error("Could not find header row in the table.")
             return tickers_industries

        headers_list = [th.text.strip().lower() for th in header_row.find_all(['th', 'td'])]
        logging.debug(f"Found table headers: {headers_list}")

        symbol_col_index = -1
        sector_col_index = -1
        possible_symbol_headers = ['symbol']
        possible_sector_headers = ['gics sector', 'sector']

        for i, h in enumerate(headers_list):
            if h in possible_symbol_headers: symbol_col_index = i
            if h in possible_sector_headers: sector_col_index = i
        if symbol_col_index == -1:
            logging.error(f"Could not find 'Symbol' column in headers: {headers_list}")
            return tickers_industries
        if sector_col_index == -1:
            logging.warning(f"Could not find 'GICS Sector'/'Sector' column. Proceeding without industry info.")

        data_rows = table.find_all('tr')[1:]
        for row in data_rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) > symbol_col_index:
                ticker_raw = cells[symbol_col_index].text.strip()
                ticker = ticker_raw.replace('.', '-')
                industry = "Unknown"
                if sector_col_index != -1 and len(cells) > sector_col_index:
                    industry_raw = cells[sector_col_index].text.strip()
                    if industry_raw: industry = industry_raw
                if ticker: tickers_industries[ticker] = industry
                else: logging.warning(f"Skipping row due to empty ticker: {row.text.strip()}")
            else: logging.warning(f"Skipping row due to insufficient cells: {row.text.strip()}")

        if not tickers_industries: logging.warning("No tickers extracted.")
        else: logging.info(f"Successfully fetched {len(tickers_industries)} tickers and industries.")
        return tickers_industries
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error fetching S&P 500 list: {e}")
    except Exception as e:
        logging.error(f"Error parsing S&P 500 list: {e}", exc_info=True)
    return tickers_industries


# --- Database Functions ---
def create_db_connection(db_file):
    """创建到 SQLite 数据库的连接。"""
    conn = None
    try:
        db_dir = os.path.dirname(db_file)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logging.info(f"Created database directory: {db_dir}")
        conn = sqlite3.connect(db_file, timeout=10)
        logging.info(f"Database connection established to {db_file}")
    except (sqlite3.Error, OSError) as e:
        logging.error(f"Database connection/directory error for '{db_file}': {e}")
    return conn

def create_tables(conn):
    """如果表不存在，则创建它们。"""
    cursor = conn.cursor()
    try:
        annual_sql = """
        CREATE TABLE IF NOT EXISTS annual_financials (
            ticker TEXT NOT NULL, period TEXT NOT NULL, revenue REAL, net_income REAL,
            eps REAL, op_income REAL, equity REAL, total_debt REAL, ocf REAL,
            capex REAL, ebit REAL, interest_exp REAL, PRIMARY KEY (ticker, period)
        );"""
        quarterly_sql = """
        CREATE TABLE IF NOT EXISTS quarterly_financials (
            ticker TEXT NOT NULL, period TEXT NOT NULL, revenue REAL, net_income REAL,
            eps REAL, op_income REAL, equity REAL, total_debt REAL, ocf REAL,
            capex REAL, ebit REAL, interest_exp REAL, PRIMARY KEY (ticker, period)
        );"""
        cursor.execute(annual_sql)
        cursor.execute(quarterly_sql)
        conn.commit()
        logging.info("Database tables ensured.")
    except sqlite3.Error as e:
        logging.error(f"Error creating/checking tables: {e}")
        conn.rollback()

def get_last_period(conn, ticker, table_name):
    """获取指定 ticker 在表中的最新 period。"""
    cursor = conn.cursor()
    last_p = None
    try:
        cursor.execute(f"SELECT MAX(period) FROM {table_name} WHERE ticker = ?", (ticker,))
        result = cursor.fetchone()
        if result and result[0] is not None: last_p = result[0]
        logging.debug(f"Last period for {ticker} in {table_name}: {last_p}")
    except sqlite3.Error as e:
        logging.error(f"Error getting last period for {ticker}, {table_name}: {e}")
    return last_p


def save_financial_data(conn, ticker, data, table_name):
    """将财务数据 DataFrame 保存到指定的数据库表中。"""
    if data is None or data.empty:
        logging.warning(f"No data provided to save for {ticker} to {table_name}.")
        return 0

    cursor = conn.cursor()
    rename_map = {
        'Total Revenue': 'revenue', 'TotalRevenue': 'revenue',
        'Net Income Common Stockholders': 'net_income', 'Net Income': 'net_income', 'NetIncome': 'net_income',
        'Diluted EPS': 'eps', 'DilutedEPS': 'eps',
        'Operating Income': 'op_income', 'OperatingIncome': 'op_income', 'Total Operating Income As Reported': 'op_income',
        'EBIT': 'ebit',
        'Interest Expense': 'interest_exp', 'InterestExpense': 'interest_exp',
        'Stockholders Equity': 'equity', 'Total Stockholder Equity': 'equity', 'ShareholderEquity': 'equity', 'TotalEquityGrossMinorityInterest': 'equity',
        'Total Debt': 'total_debt',
        'Operating Cash Flow': 'ocf', 'OperatingCashFlow': 'ocf', 'CashFlowFromContinuingOperatingActivities': 'ocf',
        'Capital Expenditure': 'capex', 'CapitalExpenditures': 'capex',
    }
    data_processed = data.copy().rename(columns=rename_map)

    if data_processed.columns.has_duplicates:
        duplicates = data_processed.columns[data_processed.columns.duplicated()].unique()
        logging.warning(f"Duplicate columns after rename for {ticker}, {table_name}: {duplicates.tolist()}. Keeping first.")
        data_processed = data_processed.loc[:, ~data_processed.columns.duplicated(keep='first')]

    required_db_cols = ['revenue', 'net_income', 'eps', 'op_income', 'equity', 'total_debt', 'ocf', 'capex', 'ebit', 'interest_exp']
    missing_cols = [col for col in required_db_cols if col not in data_processed.columns]
    if missing_cols:
        logging.warning(f"Missing expected DB columns for {ticker}, {table_name}: {missing_cols}. Will be NULL.")
        for col in missing_cols: data_processed[col] = np.nan

    data_processed['ticker'] = ticker
    period_assigned = False
    if isinstance(data_processed.index, pd.DatetimeIndex):
        try:
            if table_name == 'annual_financials':
                 data_processed['period'] = data_processed.index.year.astype('Int64').astype(str)
                 data_processed.dropna(subset=['period'], inplace=True)
            else:
                 data_processed['period'] = data_processed.index.strftime('%Y-%m-%d')
            period_assigned = True
        except Exception as e: logging.error(f"Error formatting DatetimeIndex period ({ticker}, {table_name}): {e}")
    else:
        try:
            data_processed['period'] = data_processed.index.astype(str)
            period_assigned = True
        except Exception as e: logging.error(f"Error converting index to string period ({ticker}, {table_name}): {e}")

    if not period_assigned or 'period' not in data_processed.columns:
        logging.error(f"Could not assign 'period' for {ticker}, {table_name}. Skipping save.")
        return 0

    final_sql_cols = ['ticker', 'period'] + required_db_cols
    try:
        data_to_save = data_processed[final_sql_cols].copy()
    except KeyError as e:
        logging.error(f"KeyError selecting final columns for SQL ({ticker}, {table_name}): {e}. Available: {data_processed.columns.tolist()}")
        return 0

    for col in required_db_cols:
        if not pd.api.types.is_numeric_dtype(data_to_save[col]):
             data_to_save[col] = pd.to_numeric(data_to_save[col], errors='coerce')

    try:
        data_tuples = [tuple(None if pd.isna(x) else x for x in row) for row in data_to_save.itertuples(index=False)]
    except Exception as e:
        logging.error(f"Error converting to tuples for SQL ({ticker}, {table_name}): {e}")
        return 0

    if not data_tuples:
         logging.warning(f"No data tuples to save for {ticker}, {table_name}")
         return 0

    placeholders = ', '.join(['?'] * len(final_sql_cols))
    cols_str = ', '.join([f'"{c}"' for c in final_sql_cols])
    sql = f"INSERT OR REPLACE INTO {table_name} ({cols_str}) VALUES ({placeholders})"

    rows_affected = 0
    try:
        cursor.executemany(sql, data_tuples)
        conn.commit()
        rows_affected = cursor.rowcount
        logging.debug(f"Saved/Replaced {len(data_tuples)} records for {ticker} to {table_name} (rowcount: {rows_affected}).")
    except sqlite3.Error as e:
        logging.error(f"SQL Error saving data for {ticker}, {table_name}: {e}")
        conn.rollback()
    except Exception as e:
        logging.exception(f"Unexpected error during SQL execution for {ticker}, {table_name}: {e}")
        conn.rollback()

    return len(data_tuples)


# --- Data Downloading ---
RETRIES = 2
SLEEP_SEC = 0.2

def _safe_yfinance_getter(tkr_object, attribute_name):
    """安全地访问 yfinance 属性，带重试机制，失败时返回空 DataFrame。"""
    ticker_symbol = getattr(tkr_object, 'ticker', 'UNKNOWN')
    for attempt in range(1, RETRIES + 1):
        try:
            data = getattr(tkr_object, attribute_name)
            if isinstance(data, pd.DataFrame):
                logging.debug(f"Getter success for {attribute_name} ({ticker_symbol}, shape {data.shape}) attempt {attempt}.")
                return data
            else:
                logging.warning(f"Getter for {attribute_name} ({ticker_symbol}) returned {type(data)} attempt {attempt}.")
        except AttributeError:
            logging.warning(f"Attribute '{attribute_name}' not found for {ticker_symbol} attempt {attempt}.")
        except Exception as exc:
            logging.warning(f"Getter failed for {attribute_name} ({ticker_symbol}) attempt {attempt}/{RETRIES}: {exc}")

        if attempt == RETRIES:
            logging.error(f"Getter failed for '{attribute_name}' ({ticker_symbol}) after {RETRIES} retries.")
            break
        time.sleep(SLEEP_SEC * attempt)
    return pd.DataFrame()


def download_data_for_ticker(ticker, config, conn):
    """Disabled download for ticker."""
    logging.info(f"[download_data_for_ticker] Skipped download for {ticker} – using existing DB data.")


def fetch_data_from_db(conn):
    """从数据库获取年度和季度数据。"""
    logging.info("Fetching data from database...")
    try:
        annual_df = pd.read_sql_query("SELECT * FROM annual_financials", conn)
        annual_df['period_num'] = pd.to_numeric(annual_df['period'], errors='coerce')
        annual_df.dropna(subset=['period_num'], inplace=True)
        annual_df['period_num'] = annual_df['period_num'].astype(int)
        annual_df.sort_values(by=['ticker', 'period_num'], inplace=True)

        quarterly_df = pd.read_sql_query("SELECT * FROM quarterly_financials", conn, parse_dates=['period'])
        quarterly_df.dropna(subset=['period'], inplace=True)
        quarterly_df.sort_values(by=['ticker', 'period'], inplace=True)

        logging.info(f"Fetched {len(annual_df)} annual & {len(quarterly_df)} quarterly records.")
        return annual_df, quarterly_df
    except (pd.errors.DatabaseError, Exception) as e:
        logging.error(f"Error fetching data from DB: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()

# --- Calculation Functions ---
# MODIFICATION: Added ticker parameter for logging
def calculate_cagr(series, years=None, min_start_value=0.01, ticker="Unknown"):
    """
    Calculates Compound Annual Growth Rate for a pandas Series.
    Returns NaN on failure or insufficient data.
    'years' argument is optional (calculated from data points if None).

    Args:
        series (pd.Series): Time series data.
        years (int, optional): The number of years requested for CAGR.
        min_start_value (float): Minimum allowed starting value.
        ticker (str): The ticker symbol for logging purposes. Defaults to "Unknown".

    Returns:
        float or np.nan: Calculated CAGR or NaN.
    """
    series_numeric = pd.to_numeric(series, errors='coerce').dropna()
    num_points = len(series_numeric)

    # Add ticker to debug log
    logging.debug(f"Input for CAGR ({ticker}): {num_points} valid points. Requesting {years}-year CAGR.")

    if num_points < 2:
        logging.debug(f"Returning NaN CAGR ({ticker}): Less than 2 data points.")
        return np.nan

    start_value = series_numeric.iloc[0]
    end_value = series_numeric.iloc[-1]
    actual_periods = num_points - 1
    calc_years = actual_periods # Use actual periods for calculation

    # Log a warning if the actual period differs from the requested period
    if years is not None and years != actual_periods and actual_periods > 0:
        # MODIFICATION: Include ticker in the warning message
        logging.warning(f"CAGR calculation ({ticker}): Provided 'years' ({years}) differs from data points ({num_points}). Using actual_periods ({actual_periods} years).")
    elif actual_periods <= 0:
         logging.debug(f"Returning NaN CAGR ({ticker}): Calculated actual_periods <= 0.")
         return np.nan

    # Validate start value
    if start_value <= min_start_value:
        logging.debug(f"Returning NaN CAGR ({ticker}): Start value ({start_value:.4f}) <= min_start_value ({min_start_value}).")
        return np.nan

    try:
        ratio = end_value / start_value
        logging.debug(f"CAGR calc ({ticker}): End={end_value}, Start={start_value}, Ratio={ratio}, Actual_Years={actual_periods}")

        if ratio < 0:
            logging.debug(f"Returning NaN CAGR ({ticker}): Negative ratio ({ratio:.4f}) indicates sign change.")
            return np.nan
        if np.isclose(end_value, 0):
             logging.debug(f"CAGR calculation ({ticker}): End value is zero, CAGR is -1.0 (-100%).")
             return -1.0

        cagr = (ratio) ** (1 / calc_years) - 1

        if not np.isfinite(cagr):
             logging.warning(f"CAGR calculation ({ticker}) resulted in non-finite value ({cagr}). Returning NaN.")
             return np.nan

        logging.debug(f"Calculated CAGR ({ticker}) over {actual_periods} years: {cagr:.4f}")
        return cagr
    except (ValueError, OverflowError, ZeroDivisionError, TypeError) as e:
        logging.warning(f"CAGR calculation error ({ticker}): {e}. Returning NaN.")
        return np.nan

def calculate_slope(y_series):
    """Calculates the slope of a linear regression for a pandas Series. Returns NaN on failure."""
    y_numeric = pd.to_numeric(y_series, errors='coerce').dropna()
    y = y_numeric.values
    num_points = len(y)
    logging.debug(f"Calculating slope from {num_points} valid points (Original series length: {len(y_series)}). Data: {y}")

    if num_points < 2:
        logging.debug(f"Returning NaN slope due to insufficient points ({num_points} < 2).")
        return np.nan

    x = np.arange(num_points)
    try:
        if not np.all(np.isfinite(y)):
             logging.warning(f"Non-finite values found in series AFTER dropna for slope: {y}. Returning NaN.")
             return np.nan
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        logging.debug(f"Slope result: slope={slope:.4f}, intercept={intercept:.4f}, r={r_value:.4f}, p={p_value:.4f}")
        if not np.isfinite(slope):
             logging.warning(f"Slope calculation resulted in non-finite value ({slope}). Returning NaN.")
             return np.nan
        return slope
    except ValueError as e:
        logging.warning(f"Linregress error for data {y}: {e}. Returning NaN.")
        return np.nan
    except Exception as e_linreg:
        logging.error(f"Unexpected error during linregress for data {y}: {e_linreg}", exc_info=True)
        return np.nan

# DEBUGGING HELPER FUNCTION for CAGR
def calculate_cagr_with_logging(group, column_name, years, min_start_value=0.01):
    """Helper function to log data passed to calculate_cagr, especially for specific tickers."""
    ticker = group.name
    series_to_pass = group[column_name]
    # Log detailed info specifically for ABNB or any other ticker causing issues
    # MODIFY THIS LIST IF YOU NEED TO DEBUG OTHER TICKERS
    debug_tickers = ['ABNB', 'EMR', 'WMT', 'NVDA'] # Added NVDA
    if ticker in debug_tickers:
        logging.info(f"--- DEBUG CAGR ({ticker}, {column_name}) ---")
        logging.info(f"Data passed to calculate_cagr (from tail({years+1})):\n{series_to_pass.to_string()}")
        # Print points before and after dropna
        series_numeric_before_drop = pd.to_numeric(series_to_pass, errors='coerce')
        series_numeric_after_drop = series_numeric_before_drop.dropna()
        logging.info(f"Points before dropna: {len(series_numeric_before_drop)}, Points after dropna: {len(series_numeric_after_drop)}")
        logging.info(f"NaN count in original series: {series_numeric_before_drop.isna().sum()}")
        logging.info(f"-------------------------------------")
    # Call the original calculate_cagr function
    return calculate_cagr(series_to_pass, years=years, min_start_value=min_start_value, ticker=ticker)


# --- NEW HELPER FUNCTION for Turnaround Bonus ---
def check_eps_turnaround(series, min_start_eps):
    """
    Checks if the EPS series shows a turnaround from non-positive to positive.

    Args:
        series (pd.Series): The EPS series (should be already cleaned of NaNs).
        min_start_eps (float): The threshold below which starting EPS is considered non-positive.

    Returns:
        int: 1 if turnaround condition is met, 0 otherwise.
    """
    if len(series) < 2:
        return 0 # Not enough data to determine turnaround

    start_eps = series.iloc[0]
    end_eps = series.iloc[-1]

    # Check if start is non-positive and end is positive
    if start_eps <= min_start_eps and end_eps > 0:
        return 1 # Turnaround detected
    else:
        return 0 # No turnaround
# --- Calculation Functions ---
# (calculate_cagr, calculate_slope, calculate_cagr_with_logging remain unchanged)
def calculate_indicators(annual_df, quarterly_df, config):
    """
    Calculates all required financial indicators, including the EPS Turnaround Flag.
    Returns a DataFrame indexed by ticker with calculated indicators.
    """
    logging.info("Calculating financial indicators...")
    if annual_df.empty or 'ticker' not in annual_df.columns:
        logging.warning("Annual DataFrame is empty or missing 'ticker'. Some indicators might be unavailable.")
    if quarterly_df.empty or 'ticker' not in quarterly_df.columns:
        logging.warning("Quarterly DataFrame is empty or missing 'ticker'. Some indicators might be unavailable.")

    all_tickers = pd.unique(pd.concat([annual_df['ticker'], quarterly_df['ticker']]).dropna())
    if len(all_tickers) == 0:
        logging.warning("No tickers found. Cannot calculate indicators.")
        return pd.DataFrame()

    try:
        data_cfg = config['Data']
        calc_cfg = config['Calculation_Params']
        N_years = data_cfg.get('years_of_annual_data', 3) # This is N for N-year CAGR
        min_eps_cagr = calc_cfg.get('min_eps_for_cagr', 0.01)
        eps_qoq_handling = calc_cfg.get('eps_qoq_denominator_handling', 'nan').lower()
        eps_qoq_zero_val = calc_cfg.get('eps_qoq_zero_value', 0.0)
    except KeyError as e:
        logging.error(f"Missing config section/key for calculations: {e}. Using defaults.")
        N_years, min_eps_cagr, eps_qoq_handling, eps_qoq_zero_val = 3, 0.01, 'nan', 0.0

    essential_annual_cols = ['revenue', 'eps', 'ocf', 'capex']
    essential_quarterly_cols = ['revenue', 'eps', 'op_income', 'net_income', 'ebit', 'interest_exp', 'equity', 'total_debt']
    for df, cols, name in [(annual_df, essential_annual_cols, "Annual"), (quarterly_df, essential_quarterly_cols, "Quarterly")]:
        if df.empty: continue
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
            elif col not in ['ticker', 'period', 'period_num']:
                 if not pd.api.types.is_numeric_dtype(df[col]):
                      df[col] = pd.to_numeric(df[col], errors='coerce')

    results_df = pd.DataFrame(index=all_tickers)
    results_df.index.name = 'ticker'

    # --- 1. CAGR & EPS Turnaround Flag ---
    try:
        if not annual_df.empty:
            annual_grouped = annual_df.groupby('ticker')
            # Get last N+1 years for N-year CAGR calculation & Turnaround Check
            annual_n_plus_1_years = annual_grouped.tail(N_years + 1)

            if not annual_n_plus_1_years.empty:
                 annual_n_plus_1_years_grouped = annual_n_plus_1_years.groupby('ticker')
                 logging.debug(f"Calculating Annual CAGRs using last up to {N_years + 1} data points (for {N_years}-year CAGR).")

                 results_df['Revenue_CAGR'] = annual_n_plus_1_years_grouped.apply(
                     lambda group: calculate_cagr_with_logging(group, 'revenue', N_years),
                     include_groups=False
                 )
                 results_df['EPS_CAGR'] = annual_n_plus_1_years_grouped.apply(
                     lambda group: calculate_cagr_with_logging(group, 'eps', N_years, min_eps_cagr),
                     include_groups=False
                 )
                 # --- Calculate EPS Turnaround Flag ---
                 logging.debug("Calculating EPS Turnaround Flag...")
                 results_df['EPS_Turnaround_Flag'] = annual_n_plus_1_years_grouped.apply(
                     # Pass the cleaned EPS series from the N+1 year data
                     lambda group: check_eps_turnaround(
                         pd.to_numeric(group['eps'], errors='coerce').dropna(), # Pass cleaned series
                         min_eps_cagr
                     ),
                     include_groups=False
                 )
                 turnaround_count = results_df['EPS_Turnaround_Flag'].sum()
                 logging.info(f"Identified {turnaround_count} companies with potential EPS turnaround.")
                 # --- End Turnaround Calculation ---

            else:
                 logging.warning(f"No data available after selecting last {N_years + 1} years for CAGR.")
                 results_df['EPS_Turnaround_Flag'] = 0 # Default to 0 if no data
        else:
            logging.warning("Annual data is empty, skipping CAGR and Turnaround calculations.")
            results_df['EPS_Turnaround_Flag'] = 0 # Default to 0 if no data
    except Exception as e_cagr:
        logging.exception(f"Error calculating annual CAGRs or Turnaround Flag: {e_cagr}")
        results_df['EPS_Turnaround_Flag'] = 0 # Default to 0 on error

    # --- 2. FCF Slope ---
    try:
        if not annual_df.empty:
            if 'fcf' not in annual_df.columns:
                if 'ocf' in annual_df.columns and 'capex' in annual_df.columns:
                    annual_df['fcf'] = annual_df['ocf'].fillna(0) - annual_df['capex'].fillna(0)
                else: annual_df['fcf'] = np.nan
            if 'fcf' in annual_df.columns:
                annual_grouped_fcf = annual_df.groupby('ticker')
                results_df['Annual_FCF_Growth_Slope'] = annual_grouped_fcf.apply(
                    lambda x: calculate_slope(x.sort_values('period_num')['fcf'].tail(N_years + 1)),
                    include_groups=False
                )
                logging.debug("Applied FCF slope calculation using N+1 points.")
            else: logging.warning("OCF or CapEx missing, cannot calculate FCF slope.")
        else: logging.warning("Annual data empty, skipping FCF slope.")
    except Exception as e_fcf: logging.exception(f"Error calculating FCF slope: {e_fcf}")

    # --- 3. Quarterly Slopes ---
    try:
        if not quarterly_df.empty:
            quarterly_grouped = quarterly_df.groupby('ticker')
            quarterly_df['SeqGrowth_Rev'] = quarterly_grouped['revenue'].pct_change()
            quarterly_df['eps_shifted'] = quarterly_grouped['eps'].shift(1)
            quarterly_df['SeqGrowth_EPS'] = np.nan
            mask_valid_denom = quarterly_df['eps_shifted'] > 0
            quarterly_df.loc[mask_valid_denom, 'SeqGrowth_EPS'] = (quarterly_df['eps'] - quarterly_df['eps_shifted']) / quarterly_df['eps_shifted']
            mask_zero_neg_denom = quarterly_df['eps_shifted'] <= 0
            if eps_qoq_handling == 'zero': quarterly_df.loc[mask_zero_neg_denom, 'SeqGrowth_EPS'] = eps_qoq_zero_val
            group_counts = quarterly_grouped.cumcount()
            quarterly_df.loc[group_counts == 0, ['SeqGrowth_Rev', 'SeqGrowth_EPS']] = np.nan
            quarterly_df['SeqGrowth_Rev'].replace([np.inf, -np.inf], np.nan, inplace=True)
            quarterly_df['SeqGrowth_EPS'].replace([np.inf, -np.inf], np.nan, inplace=True)
            def last_4_qoq_slope(group, col):
                growth_rates = group.sort_values('period')[col].dropna().tail(4)
                return calculate_slope(growth_rates) if len(growth_rates) >= 2 else np.nan
            results_df['Slope_Revenue'] = quarterly_grouped.apply(lambda x: last_4_qoq_slope(x, 'SeqGrowth_Rev'), include_groups=False)
            results_df['Slope_EPS'] = quarterly_grouped.apply(lambda x: last_4_qoq_slope(x, 'SeqGrowth_EPS'), include_groups=False)
        else: logging.warning("Quarterly data empty, skipping QoQ slopes.")
    except Exception as e_slope: logging.exception(f"Error calculating quarterly slopes: {e_slope}")

    # --- 4. Profitability & Efficiency (TTM Quarterly Data) ---
    latest_qtr_data = pd.DataFrame() # Initialize to avoid UnboundLocalError
    try:
        if not quarterly_df.empty:
            ttm_source_cols = ['revenue', 'op_income', 'net_income', 'ebit', 'interest_exp']
            if all(col in quarterly_df.columns for col in ttm_source_cols):
                quarterly_grouped = quarterly_df.groupby('ticker')
                # Add debug logs for specific tickers if needed
                # ... (TTM debug logs as before) ...
                logging.debug("Calculating TTM sums...")
                for col in ttm_source_cols:
                    quarterly_df[f'TTM_{col}'] = quarterly_grouped[col].transform(lambda x: x.rolling(window=4, min_periods=4).sum())
                if 'equity' in quarterly_df.columns: quarterly_df['Avg_Equity_5Q'] = quarterly_grouped['equity'].transform(lambda x: x.rolling(window=5, min_periods=5).mean())
                else: quarterly_df['Avg_Equity_5Q'] = np.nan

                # Get the latest quarter data for each ticker
                latest_qtr_data = quarterly_df.sort_values('period').drop_duplicates(subset='ticker', keep='last').set_index('ticker')

                if not latest_qtr_data.empty:
                    ttm_revenue = latest_qtr_data.get('TTM_revenue')
                    ttm_op_income = latest_qtr_data.get('TTM_op_income')
                    ttm_net_income = latest_qtr_data.get('TTM_net_income')
                    avg_equity_5q = latest_qtr_data.get('Avg_Equity_5Q')
                    # Calculate TTM Op Margin
                    op_margin_series = pd.Series(np.nan, index=latest_qtr_data.index)
                    valid_margin_mask = pd.notna(ttm_revenue) & (ttm_revenue != 0) & pd.notna(ttm_op_income)
                    op_margin_series.loc[valid_margin_mask] = ttm_op_income.loc[valid_margin_mask] / ttm_revenue.loc[valid_margin_mask]
                    results_df['TTM_OpMargin_Level'] = op_margin_series
                    # Calculate TTM ROE
                    roe_series = pd.Series(np.nan, index=latest_qtr_data.index)
                    valid_roe_mask = pd.notna(avg_equity_5q) & (avg_equity_5q != 0) & pd.notna(ttm_net_income)
                    roe_series.loc[valid_roe_mask] = ttm_net_income.loc[valid_roe_mask] / avg_equity_5q.loc[valid_roe_mask]
                    results_df['TTM_ROE_Level'] = roe_series
                    logging.debug("Calculated TTM OpMargin and ROE levels.")
                else: logging.warning("Latest qtr data empty after TTM calc.")
            else: logging.warning(f"Missing TTM source columns. Skipping TTM Profitability/Efficiency.")
        else: logging.warning("Quarterly data empty, skipping TTM.")
    except Exception as e_ttm: logging.exception(f"Error calculating TTM metrics: {e_ttm}")

    # --- 5. Health Metrics ---
    try:
        if not latest_qtr_data.empty:
            latest_equity = latest_qtr_data.get('equity')
            latest_total_debt = latest_qtr_data.get('total_debt')
            ttm_ebit = latest_qtr_data.get('TTM_ebit')
            ttm_interest_exp = latest_qtr_data.get('TTM_interest_exp')

            # --- D/E Ratio Calculation with Debug Log ---
            if latest_equity is not None and latest_total_debt is not None:
                # Calculate the division result separately for debugging
                division_result = latest_total_debt.fillna(0).astype(float) / latest_equity.astype(float)
                # Define the condition
                condition = pd.notna(latest_equity) & (latest_equity != 0)


                # ***** MODIFIED ASSIGNMENT *****
                # Calculate the final D/E Series based on the condition
                de_ratio_final = pd.Series(np.nan, index=latest_equity.index) # Initialize with NaNs, aligned to latest_equity index
                de_ratio_final.loc[condition] = division_result.loc[condition] # Assign division result only where condition is True

                # Assign the calculated Series to the results_df column
                # Pandas will align based on the index (ticker)
                results_df['Debt_to_Equity_Ratio'] = de_ratio_final
                # ***** END MODIFIED ASSIGNMENT *****


            else:
                logging.warning("Missing equity or total_debt data for D/E ratio calculation.")
                results_df['Debt_to_Equity_Ratio'] = np.nan # Ensure column exists

            # --- ICR Calculation ---
            if ttm_ebit is not None and ttm_interest_exp is not None:
                icr = pd.Series(np.nan, index=latest_qtr_data.index, dtype=float)
                mask_pos_int = pd.notna(ttm_interest_exp) & (ttm_interest_exp > 0); icr.loc[mask_pos_int] = ttm_ebit.loc[mask_pos_int].fillna(0) / ttm_interest_exp.loc[mask_pos_int]
                mask_zero_int = pd.notna(ttm_interest_exp) & np.isclose(ttm_interest_exp, 0); icr.loc[mask_zero_int & pd.notna(ttm_ebit) & (ttm_ebit >= 0)] = np.inf; icr.loc[mask_zero_int & pd.notna(ttm_ebit) & (ttm_ebit < 0)] = -np.inf; icr.loc[mask_zero_int & ttm_ebit.isna()] = np.nan
                mask_neg_int = pd.notna(ttm_interest_exp) & (ttm_interest_exp < 0); icr.loc[mask_neg_int] = np.nan
                results_df['Interest_Coverage_Ratio'] = icr
            else:
                logging.warning("Missing TTM EBIT or Interest Expense data for ICR calculation.")
                results_df['Interest_Coverage_Ratio'] = np.nan # Ensure column exists
        else:
            logging.warning("Latest qtr data unavailable, skipping Health metrics.")
            results_df['Debt_to_Equity_Ratio'] = np.nan
            results_df['Interest_Coverage_Ratio'] = np.nan
    except Exception as e_health:
        logging.exception(f"Error calculating health metrics: {e_health}")
        results_df['Debt_to_Equity_Ratio'] = np.nan
        results_df['Interest_Coverage_Ratio'] = np.nan

    # --- Final Cleanup & Return ---
    all_indicator_cols = [
        'Revenue_CAGR', 'EPS_CAGR', 'Annual_FCF_Growth_Slope',
        'Slope_Revenue', 'Slope_EPS', 'TTM_OpMargin_Level', 'TTM_ROE_Level',
        'Debt_to_Equity_Ratio', 'Interest_Coverage_Ratio',
        'EPS_Turnaround_Flag' # Ensure the new flag column is included
    ]
    for col in all_indicator_cols:
        if col not in results_df.columns:
            # Default flag to 0, others to NaN if missing
            results_df[col] = 0 if col == 'EPS_Turnaround_Flag' else np.nan

    inf_count = np.isinf(results_df.select_dtypes(include=np.number)).sum().sum()
    if inf_count > 0:
        results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        logging.debug(f"Replaced {inf_count} inf values.")

    logging.info("Finished calculating indicators.")
    final_indicators = results_df.reset_index() # Make ticker a column again
    return final_indicators


# --- Scoring Functions ---
# (apply_financial_screening, standardize_metrics, calculate_final_score remain unchanged)
def apply_financial_screening(df, config):
    """Applies financial health screening filters based on config."""

    try:
        screen_cfg = config['Screening']
        if not screen_cfg.get('enable_screening', False): # Default to False if key missing
            logging.info("Financial screening disabled by config.")
            df['Screened_Out'] = False # Add column indicating none failed
            return df
    except KeyError:
        logging.warning("Missing [Screening] section in config. Assuming screening is disabled.")
        df['Screened_Out'] = False
        return df

    logging.info("Applying financial screening...")
    max_de = screen_cfg.get('max_debt_to_equity', 2.0) # Default values
    min_icr = screen_cfg.get('min_interest_coverage', 3.0)

    # Start assuming all pass
    df['passed_screening'] = True
    reasons = [] # Store reasons for failure


    # Check Debt-to-Equity
    if 'Debt_to_Equity_Ratio' in df.columns:
        # Fail if D/E is NaN OR exceeds threshold
        # Ensure the column is numeric before comparison to avoid type errors
        de_col_numeric = pd.to_numeric(df['Debt_to_Equity_Ratio'], errors='coerce')
        de_fails = de_col_numeric.isna() | (de_col_numeric > max_de)
        df.loc[de_fails, 'passed_screening'] = False
        reasons.append(f"D/E > {max_de} or NaN ({de_fails.sum()} failed)")
    else:
        logging.warning("Debt_to_Equity_Ratio column not found for screening. All companies will fail this check if enabled.")
        df['passed_screening'] = False # Fail all if column missing
        reasons.append("D/E Missing")

    # Check Interest Coverage Ratio (Ignoring NaN as requested)
    if 'Interest_Coverage_Ratio' in df.columns:
        icr_numeric = pd.to_numeric(df['Interest_Coverage_Ratio'], errors='coerce')
        # Fail only if ICR is NOT NaN AND is below the threshold
        icr_fails = pd.notna(icr_numeric) & (icr_numeric < min_icr)
        df.loc[icr_fails, 'passed_screening'] = False # Use regular assignment, |= not needed if starting with True
        num_failed_icr = icr_fails.sum()
        if num_failed_icr > 0 : reasons.append(f"ICR < {min_icr} ({num_failed_icr} failed)")
        num_nan_icr = icr_numeric.isna().sum()
        if num_nan_icr > 0: logging.debug(f"ICR check: Ignored {num_nan_icr} NaN values.")
    else:
        logging.warning("Interest_Coverage_Ratio column not found for screening. All companies will fail this check if enabled.")
        df['passed_screening'] = False # Fail all if column missing
        reasons.append("ICR Missing")


    # Final 'Screened_Out' column is the inverse of 'passed_screening'
    df['Screened_Out'] = ~df['passed_screening']
    df.drop(columns=['passed_screening'], inplace=True) # Remove intermediate column

    screened_out_count = df['Screened_Out'].sum()
    logging.info(f"Screening complete. Reasons: {'; '.join(reasons)}. Total screened out: {screened_out_count}.")
    return df


def standardize_metrics(df, config):
    """
    Calculates percentile rank scores (0-100) for metrics based on config methodology.
    Handles NaNs and merges scores back carefully.
    """
    try:
        ranking_method = config.get('Methodology', {}).get('ranking_method', 'overall')
    except KeyError:
        logging.warning("Missing [Methodology] section or 'ranking_method' key. Defaulting to 'overall'.")
        ranking_method = 'overall'
    logging.info(f"Standardizing metrics using '{ranking_method}' percentile ranking...")

    metrics_to_standardize = [
        'Revenue_CAGR', 'EPS_CAGR', 'Annual_FCF_Growth_Slope',
        'TTM_OpMargin_Level', 'TTM_ROE_Level',
        'Slope_Revenue', 'Slope_EPS'
    ]

    if 'ticker' not in df.columns:
         logging.error("Input DataFrame to standardize_metrics is missing 'ticker' column!")
         for metric in metrics_to_standardize:
             df[f'Score_{metric}'] = np.nan
         return df
    if ranking_method == 'industry' and 'Industry' not in df.columns:
         logging.error("Ranking method 'industry' selected, but 'Industry' column is missing! Falling back to 'overall'.")
         ranking_method = 'overall'
    if 'Screened_Out' not in df.columns:
         logging.warning("'Screened_Out' column not found in standardize_metrics input. Assuming all companies passed screening for ranking.")
         df['Screened_Out'] = False

    df_to_rank = df[~df['Screened_Out']].copy()

    if df_to_rank.empty:
        logging.warning("No companies passed screening. Cannot standardize metrics. Scores will be NaN.")
        for metric in metrics_to_standardize:
            df[f'Score_{metric}'] = np.nan
        return df

    calculated_scores = {}
    logging.debug(f"Calculating percentile ranks ({ranking_method}) for {len(df_to_rank)} screened-in companies...")

    if ranking_method == 'industry':
        if 'Industry' not in df_to_rank.columns:
             logging.error("Critical: 'Industry' column still missing despite check. Aborting industry ranking.")
             ranking_method = 'overall'
        else:
             df_to_rank['Industry'].fillna('Unknown', inplace=True)
             grouped_by_industry = df_to_rank.groupby('Industry')

    for metric in metrics_to_standardize:
        score_col_name = f'Score_{metric}'
        if metric in df_to_rank.columns:
            metric_series = df_to_rank[metric]
            nan_count_before = metric_series.isna().sum()
            logging.debug(f"  Ranking '{metric}' ({nan_count_before} NaNs)...")

            if ranking_method == 'industry':
                try:
                    ranks = grouped_by_industry[metric].transform(
                        lambda x: x.rank(pct=True, na_option='keep')
                    ) * 100
                    calculated_scores[score_col_name] = ranks
                except Exception as e_rank_ind:
                     logging.error(f"Error calculating industry rank for '{metric}': {e_rank_ind}", exc_info=True)
                     calculated_scores[score_col_name] = pd.Series(np.nan, index=df_to_rank.index)
            else: # 'overall' ranking
                ranks = metric_series.rank(pct=True, na_option='keep') * 100
                calculated_scores[score_col_name] = ranks

            nan_count_after = calculated_scores[score_col_name].isna().sum()
            logging.debug(f"    -> Score '{score_col_name}' calculated ({nan_count_after} NaNs).")
        else:
            logging.warning(f"Metric '{metric}' not found in DataFrame. Skipping score calculation for it.")
            calculated_scores[score_col_name] = pd.Series(np.nan, index=df_to_rank.index)

    df_scores = pd.DataFrame(calculated_scores)
    df_result = df.merge(df_scores, left_index=True, right_index=True, how='left')

    logging.info(f"Finished standardizing metrics (using '{ranking_method}' method).")
    logging.debug(f"Columns after standardization merge: {df_result.columns.tolist()}")
    return df_result


# --- MODIFIED calculate_final_score Function ---
def calculate_final_score(df, config):
    """Calculates the fused growth scores and the final overall score, imputing missing scores with 50 and adding turnaround bonus."""
    logging.info("Calculating final composite scores...")
    try:
        weights = config['Scoring_Weights']
        # Get the turnaround bonus weight, default to 0 if not specified
        w_eps_turnaround = weights.get('w_eps_turnaround', 0.0)
        # Ensure it's a float
        try:
            w_eps_turnaround = float(w_eps_turnaround)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert w_eps_turnaround '{weights.get('w_eps_turnaround')}' to float. Defaulting to 0.0.")
            w_eps_turnaround = 0.0

        if w_eps_turnaround != 0: # Log only if bonus is active
            logging.info(f"Applying EPS Turnaround Bonus with weight: {w_eps_turnaround}")
        else:
            logging.info("EPS Turnaround Bonus weight is 0 or not configured. Bonus will not be applied.")
    except KeyError:
        logging.error("Missing [Scoring_Weights] section in config. Cannot calculate final scores.")
        # Ensure necessary columns exist even if calculation fails
        df['Score_Growth_Revenue'] = np.nan
        df['Score_Growth_EPS'] = np.nan
        df['Overall_Score'] = np.nan
        return df

    # Define which standardized scores are needed for the base calculation
    score_cols_needed = [
        'Score_Revenue_CAGR', 'Score_Slope_Revenue',
        'Score_EPS_CAGR', 'Score_Slope_EPS',
        'Score_TTM_OpMargin_Level', 'Score_TTM_ROE_Level',
        'Score_Annual_FCF_Growth_Slope'
    ]
    imputed_scores = {}

    logging.debug("Imputing NaN scores with 50 before final calculation:")
    for col in score_cols_needed:
        if col in df.columns:
            original_nans = df[col].isna().sum()
            # Impute NaN with 50.0 - using .fillna() is generally safer
            imputed_scores[col] = df[col].fillna(50.0)
            if original_nans > 0:
                 logging.debug(f"  Score column '{col}': Imputed {original_nans} NaNs with 50.")
        else:
            logging.warning(f"Required score column '{col}' missing. Using default 50 for calculations.")
            # Create a Series of 50s aligned with the DataFrame's index
            imputed_scores[col] = pd.Series(50.0, index=df.index)

    # Retrieve imputed scores safely using .get() on the dictionary
    score_rev_cagr = imputed_scores.get('Score_Revenue_CAGR', pd.Series(50.0, index=df.index))
    score_rev_accel = imputed_scores.get('Score_Slope_Revenue', pd.Series(50.0, index=df.index))
    score_eps_cagr = imputed_scores.get('Score_EPS_CAGR', pd.Series(50.0, index=df.index))
    score_eps_accel = imputed_scores.get('Score_Slope_EPS', pd.Series(50.0, index=df.index))
    score_profitability = imputed_scores.get('Score_TTM_OpMargin_Level', pd.Series(50.0, index=df.index))
    score_efficiency = imputed_scores.get('Score_TTM_ROE_Level', pd.Series(50.0, index=df.index))
    score_fcf = imputed_scores.get('Score_Annual_FCF_Growth_Slope', pd.Series(50.0, index=df.index))

    # Fuse Growth Scores
    w_cagr = weights.get('w_cagr', 0.6) # Use .get for safety
    w_accel = weights.get('w_accel', 0.4)
    df['Score_Growth_Revenue'] = (w_cagr * score_rev_cagr) + (w_accel * score_rev_accel)
    df['Score_Growth_EPS'] = (w_cagr * score_eps_cagr) + (w_accel * score_eps_accel)
    logging.debug("Calculated fused Score_Growth_Revenue and Score_Growth_EPS.")

    # Calculate Final Overall Score including Turnaround Bonus
    w_growth_rev = weights.get('w_growth_rev', 0.25)
    w_growth_eps = weights.get('w_growth_eps', 0.25)
    w_profitability = weights.get('w_profitability', 0.15)
    w_efficiency = weights.get('w_efficiency', 0.15)
    w_fcf = weights.get('w_fcf', 0.20)

    # Get the turnaround flag (default to 0 if column missing or contains NaN)
    # Ensure it's numeric and fillna before calculation
    eps_turnaround_flag = pd.to_numeric(df.get('EPS_Turnaround_Flag', 0), errors='coerce').fillna(0)

    # Calculate base score from weighted dimensions
    base_overall_score = (
        (w_growth_rev * df['Score_Growth_Revenue']) +
        (w_growth_eps * df['Score_Growth_EPS']) +
        (w_profitability * score_profitability) +
        (w_efficiency * score_efficiency) +
        (w_fcf * score_fcf)
    )

    # Add the turnaround bonus directly (scaled by 100 if flag is 1)
    # Multiply the flag (0 or 1) by the weight * 100 to represent the bonus points
    turnaround_bonus_points = eps_turnaround_flag * (w_eps_turnaround * 100)


    df['Overall_Score'] = base_overall_score + turnaround_bonus_points
    logging.debug("Calculated Overall_Score including potential EPS turnaround bonus.")


    # Log companies receiving the bonus
    # Ensure index is accessible or use ticker column
    if 'ticker' in df.columns:
        bonus_tickers = df.loc[df['EPS_Turnaround_Flag'] == 1, 'ticker'].tolist()
    else: # Fallback to index if ticker column is missing (less likely now)
        bonus_tickers = df[df['EPS_Turnaround_Flag'] == 1].index.tolist()

    if bonus_tickers:
        logging.info(f"Applied EPS turnaround bonus (Weight: {w_eps_turnaround}) to: {', '.join(map(str, bonus_tickers))}")


    # Set scores to NaN for companies screened out AFTER calculation
    if 'Screened_Out' in df.columns:
        screened_mask = df['Screened_Out']
        # Include the turnaround flag in columns to NaN for screened out companies? Optional.
        # Let's keep the flag visible, but NaN out the scores.
        cols_to_nan_for_screened = ['Overall_Score', 'Score_Growth_Revenue', 'Score_Growth_EPS'] + score_cols_needed
        for col in cols_to_nan_for_screened:
            if col in df.columns:
                 df.loc[screened_mask, col] = np.nan
        logging.debug(f"Set scores to NaN for {screened_mask.sum()} screened-out companies.")
    else:
        logging.warning("'Screened_Out' column not found before final score NaN assignment.")

    logging.info("Finished calculating final scores.")
    return df
# --- Main Execution Function (Modified) ---
# MODIFIED: Added update_data parameter with default False and db_path override
def compute_growth_score(update_data=False, db_path: str | None = None):
    """
    Main function to orchestrate the growth score calculation process.

    Args:
        update_data (bool): If True, run the data download phase.
                            If False, skip the data download phase.
                            Defaults to True.
    """
    run_start_time = time.time()
    try:
        config = load_config()
        log_level_setting = config.get('General', {}).get('log_level', 'INFO')
        setup_logging(log_level_str=log_level_setting, log_to_file=config.get('General', {}).get('log_to_file', False))
        logging.info(f"--- Growth Score Script Starting --- Update Data Flag: {update_data} ---")

    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        print(f"CRITICAL CONFIGURATION ERROR: {e}")
        logging.critical(f"CRITICAL CONFIGURATION ERROR: {e}", exc_info=True)
        return False

    data_cfg = config.get('Data', {})
    inc_download = data_cfg.get('incremental_download', True)

    # --- Database Connection & Setup ---
    conn = None
    try:
        db_name = db_path or data_cfg.get('db_name') or _get_finance_db()
        conn = create_db_connection(db_name)
        if not conn:
            logging.critical("Failed to establish database connection. Exiting.")
            return False
        create_tables(conn)

        # Delete old data if full update requested
        if update_data and not inc_download:
            if os.path.exists(db_name):
                try:
                    # Example: Clear tables instead of deleting file
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM annual_financials;")
                    cursor.execute("DELETE FROM quarterly_financials;")
                    conn.commit()
                    logging.info(f"Cleared tables in {db_name} for full reload.")
                except (sqlite3.Error, PermissionError) as e:
                    logging.error(f"Failed to clear tables in {db_name}: {e}")
                    if conn:
                        conn.close()
                    return False  # Stop if clearing fails

    except (KeyError, OSError, sqlite3.Error) as e:
         logging.critical(f"Database setup/clearing error: {e}. Exiting.")
         if conn:
             conn.close()
         return False

    # --- Get Tickers and Industries ---
    tickers_industries_dict = {}
    industry_df = pd.DataFrame() # Initialize
    try:
        url = config.get('General', {}).get('sp500_list_url')
        if not url:
             raise ValueError("Missing 'sp500_list_url' in [General] config section.")
        tickers_industries_dict = get_sp500_tickers_and_industries(url)
        if not tickers_industries_dict:
            logging.error("Could not retrieve S&P 500 ticker list with industries. Proceeding without industry data.")
            # Create an empty df with expected columns if needed later, or handle absence
            industry_df = pd.DataFrame(columns=['ticker', 'Industry'])
        else:
            tickers = sorted(list(tickers_industries_dict.keys()))
            industry_df = pd.DataFrame(list(tickers_industries_dict.items()), columns=['ticker', 'Industry'])
            logging.debug(f"Industry data fetched. Example:\n{industry_df.head().to_string()}")
    except (KeyError, ValueError, requests.exceptions.RequestException) as e:
        logging.critical(f"Error getting ticker list/URL: {e}. Exiting.")
        if conn:
            conn.close()
        return False

    # --- Data Download Phase (Conditional) ---
    if update_data:
        logging.info("--- Starting Data Download Phase ---")
        # Use tickers from the fetched list if available, otherwise maybe skip?
        tickers_to_download = tickers if tickers_industries_dict else []
        if not tickers_to_download:
             logging.warning("No tickers available to download data for.")
        else:
            successful_downloads = 0
            total_tickers = len(tickers_to_download)
            for i, ticker in enumerate(tickers_to_download):
                logging.info(f"Downloading data for {ticker} ({i+1}/{total_tickers})...")
                try:
                    download_data_for_ticker(ticker, config, conn)
                    successful_downloads += 1
                except Exception as e_download:
                     logging.error(f"Error during download/save process for ticker {ticker}: {e_download}", exc_info=True)
            logging.info(f"--- Finished Data Download Phase (Attempted: {total_tickers}, Successful Fetches (approx): {successful_downloads}) ---")
    else:
        logging.info("--- Skipping Data Download Phase based on 'update_data=False' ---")

    # --- Data Processing and Scoring Phase ---
    logging.info("--- Starting Data Processing and Scoring Phase ---")
    annual_df, quarterly_df = fetch_data_from_db(conn)

    if annual_df.empty and quarterly_df.empty:
        logging.error("Failed to fetch ANY data from database. Cannot proceed with scoring. Exiting.")
        if conn:
            conn.close()
        return False
    elif annual_df.empty:
        logging.warning("Annual data DataFrame is empty after fetching from DB.")
    elif quarterly_df.empty:
        logging.warning("Quarterly data DataFrame is empty after fetching from DB.")

    # Calculate indicators
    # Ensure calculate_indicators returns df with ticker as a column
    indicator_df = calculate_indicators(annual_df, quarterly_df, config)

    if not isinstance(indicator_df, pd.DataFrame) or indicator_df.empty:
        logging.error("Indicator calculation returned an empty or invalid DataFrame. Exiting.")
        if conn:
            conn.close()
        return False
    if 'ticker' not in indicator_df.columns:
        # If calculate_indicators returns ticker as index, reset it here
        if indicator_df.index.name == 'ticker':
             indicator_df.reset_index(inplace=True)
             logging.debug("Reset index 'ticker' to column in indicator_df.")
        else:
             logging.error("Indicator DataFrame is missing 'ticker' column/index after calculation. Exiting.")
             if conn:
                 conn.close()
             return False

    # Ensure ticker column is string type for reliable merging BEFORE merge
    indicator_df['ticker'] = indicator_df['ticker'].astype(str)


    # Merge industry data
    if not industry_df.empty:
        logging.debug(f"Merging industry data. Indicator shape before: {indicator_df.shape}")
        # Ensure ticker in industry_df is also string
        industry_df['ticker'] = industry_df['ticker'].astype(str)
        # Perform merge
        # Use validate='one_to_one' or 'one_to_many' if appropriate to catch key issues
        try:
            indicator_df_merged = pd.merge(indicator_df, industry_df, on='ticker', how='left', validate="one_to_one") # Added validate
            logging.debug(f"Merge successful. Indicator shape after merging industry: {indicator_df_merged.shape}")
        except pd.errors.MergeError as me:
             logging.error(f"Merge validation failed: {me}. Trying merge without validation.")
             # Fallback or handle error - here we try without validation
             indicator_df_merged = pd.merge(indicator_df, industry_df, on='ticker', how='left')
        except Exception as e_merge:
             logging.error(f"Unexpected error during merge: {e_merge}", exc_info=True)
             # Decide how to proceed - maybe keep unmerged df?
             indicator_df_merged = indicator_df.copy() # Keep original if merge fails badly
             if 'Industry' not in indicator_df_merged.columns: indicator_df_merged['Industry'] = 'Unknown'

        # Check for missing industries after merge
        missing_industry_count = indicator_df_merged['Industry'].isna().sum()
        if missing_industry_count > 0:
             logging.warning(f"{missing_industry_count} companies have missing industry information after merge.")
             indicator_df_merged['Industry'].fillna('Unknown', inplace=True) # Fill NaNs
        # Assign back to indicator_df
        indicator_df = indicator_df_merged
    else:
        logging.warning("Industry DataFrame is empty, cannot merge industry information.")
        indicator_df['Industry'] = 'Unknown' # Add column if missing


    # Apply screening, standardization, and final scoring
    scored_df = apply_financial_screening(indicator_df, config)
    scored_df = standardize_metrics(scored_df, config)
    scored_df = calculate_final_score(scored_df, config)

    # Add Rank column
    if 'Overall_Score' in scored_df.columns:
        scored_df.sort_values(by='Overall_Score', ascending=False, na_position='last', inplace=True)
        scored_df['Rank'] = np.nan
        valid_score_mask = scored_df['Overall_Score'].notna()
        scored_df.loc[valid_score_mask, 'Rank'] = range(1, valid_score_mask.sum() + 1)
        scored_df['Rank'] = scored_df['Rank'].astype('Int64')
    else:
        logging.error("'Overall_Score' column missing, cannot sort or rank.")
        scored_df['Rank'] = pd.NA

    # --- Final Output ---
    logging.info("--- Preparing Final Output ---")
    output_cols = [ # Define desired output columns
        'ticker', 'Industry', 'Overall_Score', 'Rank', 'Screened_Out',
        'Score_Growth_Revenue', 'Score_Growth_EPS',
        'Score_TTM_OpMargin_Level', 'Score_TTM_ROE_Level', 'Score_Annual_FCF_Growth_Slope',
        'Score_Revenue_CAGR', 'Score_Slope_Revenue', 'Score_EPS_CAGR', 'Score_Slope_EPS',
        'Revenue_CAGR', 'EPS_CAGR', 'Slope_Revenue', 'Slope_EPS',
        'TTM_OpMargin_Level', 'TTM_ROE_Level', 'Annual_FCF_Growth_Slope',
        'Debt_to_Equity_Ratio', 'Interest_Coverage_Ratio'
    ]
    output_cols_present = [col for col in output_cols if col in scored_df.columns]
    # ... (rest of output and saving logic) ...
    final_output_df = scored_df[output_cols_present].copy()
    try:
        output_file = config.get('General', {}).get('output_excel_file', 'sp500_fundamental_scores.xlsx')
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        if not output_dir:
             try: script_dir = os.path.dirname(os.path.abspath(__file__))
             except NameError: script_dir = os.getcwd()
             output_file = os.path.join(script_dir, output_file)

        logging.info(f"Saving results to {output_file}...")
        numeric_cols = final_output_df.select_dtypes(include=np.number).columns
        final_output_df[numeric_cols] = final_output_df[numeric_cols].round(4)
        final_output_df.to_excel(output_file, index=False, engine='openpyxl')
        logging.info(f"Successfully saved results to {output_file}")
    except (KeyError, PermissionError, Exception) as e:
        logging.error(f"Failed to save results to Excel: {e}", exc_info=True)


    # --- Cleanup ---
    if conn:
        try:
            conn.close()
            logging.info("Database connection closed.")
        except Exception as e_close:
             logging.error(f"Error closing database connection: {e_close}")

    run_end_time = time.time()
    logging.info(f"--- Growth Score Script Finished --- Duration: {run_end_time - run_start_time:.2f} seconds ---")
    return True

# --- Entry Point ---
if __name__ == "__main__":
    # This allows running the growth score script directly
    # By default, it will update data unless controlled otherwise (e.g., by args if added)
    print("Running compute_growth_score script directly...")
    # Example: Add argparse here if you want command-line control when running directly
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--no-update', action='store_true', help='Skip data update when run directly')
    # args = parser.parse_args()
    # compute_growth_score(update_data=not args.no_update)
    compute_growth_score(False) # Example: Default to NOT update when run directly
