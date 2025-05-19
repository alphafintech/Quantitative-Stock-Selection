#!/usr/bin/env python3
"""
Script to download daily price and volume data for all S&P 500 stocks from Yahoo Finance.
The script reads the desired date range from config.ini, checks/updates the local SQLite database,
and only downloads the incremental missing data on subsequent runs.
"""

import os
import configparser
import sqlite3
import datetime
import pandas as pd
import yfinance as yf
import time


def _get_price_db(cfg_path: str = "config.ini") -> str:
    """Return price DB path from config or default."""
    cfg = configparser.ConfigParser()
    if os.path.exists(cfg_path):
        cfg.read(cfg_path)
    return cfg.get("database", "price_db", fallback="SP500_price_data.db")


# Default database storing historical price data
DEFAULT_DB_FILE = _get_price_db()
config_file = "config_trend.ini"


# ----------------- Configuration ----------------- #
def read_config(config_file="config.ini"):
    """
    Reads configuration values from config.ini under section [data_download].
    If start_date is missing, defaults to '1900-01-01' (forcing full history download).
    If end_date is missing, defaults to today's date.
    Returns the tuple (start_date, end_date).
    """
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
        print(f"Reading configuration from {config_file}...")
    else:
        print(f"Config file {config_file} not found. Using default date values.")

    # Read the dates from config and default if missing.
    start_date = None
    end_date = None
    if 'data_download' in config:
        start_date = config['data_download'].get('start_date', None)
        end_date = config['data_download'].get('end_date', None)

    if not start_date:
        # Set to a date that forces Yahoo Finance to return all available history.
        start_date = '1900-01-01'
        print("No start_date in config; defaulting to earliest possible data (1900-01-01).")
    else:
        print(f"Using start_date from config: {start_date}")

    if not end_date:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
        print(f"No end_date in config; defaulting to today: {end_date}")
    else:
        print(f"Using end_date from config: {end_date}")

    return start_date, end_date

# ----------------- Database Setup ----------------- #
def init_db(db_file=DEFAULT_DB_FILE):
    """
    Initialize (or connect to) the local SQLite database and create the required table if it does not exist.
    Returns the connection and cursor.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
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
    """)
    conn.commit()
    print(f"Connected to database '{db_file}'.")
    return conn, cursor

# ----------------- Retrieve S&P 500 Tickers ----------------- #
def get_sp500_tickers():
    """
    Retrieves the current list of S&P 500 tickers by scraping the Wikipedia page.
    Adjusts ticker symbols if necessary (e.g., changing '.' to '-' for Yahoo Finance compatibility).
    Returns a list of ticker symbols.
    """
    print("Retrieving S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    # Adjust tickers: replace '.' with '-' for Yahoo Finance compatibility.
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    print(f"Retrieved {len(tickers)} ticker symbols.")
    tickers.append("SPY")   # 用于 RS 计算
    return tickers

# ----------------- Data Insertion ----------------- #
def insert_data(cursor, conn, df, ticker):
    """
    Inserts downloaded data into the database.
    Uses INSERT OR IGNORE to handle duplicates.
    """
    # Reset index to bring the date from the index into a column.
    df.reset_index(inplace=True)

    # Handle date column: if the column is named 'Date' (or 'index') rename to 'date'.
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'date'}, inplace=True)

    # Rename other columns to match our DB fields.
    rename_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    }
    df.rename(columns=rename_mapping, inplace=True)
    
    # Clean up column names:
    # If a column name is a tuple (from a MultiIndex), take the first element.
    df.columns = [
        (col[0].strip().lower() if isinstance(col, tuple) else col.strip().lower())
        for col in df.columns
    ]
    
    # Ensure the date column is converted to datetime.
    df['date'] = pd.to_datetime(df['date'])
    
    # Check for the 'adj_close' column. If it doesn't exist, create it using the 'close' column.
    if 'adj_close' not in df.columns:
        print(f"Ticker {ticker}: 'adj_close' column not found. Using 'close' values as fallback.")
        df['adj_close'] = df['close']
        
    # Add the ticker column.
    df['ticker'] = ticker

    # Remove any duplicate columns if they exist.
    df = df.loc[:, ~df.columns.duplicated()]

    # Reorder columns: ticker, date, open, high, low, close, adj_close, volume.
    df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]

    # Iterate over rows using itertuples, then convert each row to a dictionary.
    for row in df.itertuples(index=False, name='StockRow'):
        row_dict = row._asdict()  # Convert the namedtuple to a dictionary.
        if 'date' not in row_dict:
            print(f"Error: 'date' field not found in row for ticker {ticker}. Available keys: {list(row_dict.keys())}")
            continue
        try:
            date_str = row_dict['date'].strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error converting date for ticker {ticker}: {e}")
            continue

        try:
            ticker_val = str(row_dict['ticker'])
            open_val = float(row_dict['open'])
            high_val = float(row_dict['high'])
            low_val = float(row_dict['low'])
            close_val = float(row_dict['close'])
            adj_close_val = float(row_dict['adj_close'])
            volume_val = int(row_dict['volume'])
        except Exception as e:
            print(f"Error converting row values for ticker {ticker} on {date_str}: {e}")
            continue

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO stock_data (ticker, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticker_val, date_str, open_val, high_val, low_val, close_val, adj_close_val, volume_val))
        except Exception as e:
            print(f"Error inserting data for ticker {ticker} on {date_str}: {e}")

    conn.commit()
    print(f"Data inserted for ticker {ticker}.")

# ----------------- Data Download and Merge ----------------- #
def update_ticker_data(ticker, config_start, config_end, cursor, conn):
    """
    For a given ticker:
      - Check if data exists in the database and determine its date range.
      - If no data exists, perform a full download from config_start to config_end.
      - If data exists, download any missing data before the earliest date (backfill) or
        after the latest date (incremental update).
    """
    print(f"\nProcessing ticker: {ticker}")
    # Query for the minimum and maximum dates present for this ticker.
    cursor.execute("SELECT MIN(date), MAX(date) FROM stock_data WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    existing_min, existing_max = row if row is not None else (None, None)

    # Prepare an empty DataFrame for new data.
    new_data = pd.DataFrame()

    # Helper for converting string to date object.
    def to_date(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    # First-time download (no existing data).
    if existing_min is None or existing_max is None:
        print(f"No existing data for {ticker}; performing full download.")
        try:
            # Use the full history if the config start date is the default "1900-01-01".
            if config_start == '1900-01-01':
                df = yf.download(ticker, period="max", progress=False)
            else:
                df = yf.download(ticker, start=config_start, end=config_end, progress=False)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            return
        if not df.empty:
            new_data = df.copy()
        else:
            print(f"No data returned for {ticker}.")
    else:
        # Convert existing_min and existing_max to date objects for comparison.
        existing_min_date = to_date(existing_min)
        existing_max_date = to_date(existing_max)
        config_start_date = to_date(config_start)
        config_end_date = to_date(config_end)

        # Check if backfill is needed (if DB starts later than desired config start).
        if config_start_date < existing_min_date:
            backfill_start = config_start
            # Download until one day before the currently stored minimum date.
            backfill_end = (existing_min_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Backfilling for {ticker} from {backfill_start} to {backfill_end}.")
            try:
                df_backfill = yf.download(ticker, start=backfill_start, end=backfill_end, progress=False)
                if not df_backfill.empty:
                    new_data = pd.concat([new_data, df_backfill])
                else:
                    print(f"No backfill data returned for {ticker}.")
            except Exception as e:
                print(f"Error backfilling data for {ticker}: {e}")

        # Check if incremental update is needed (if DB ends before desired config end).
        if config_end_date > existing_max_date:
            incremental_start = (existing_max_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Incremental update for {ticker} from {incremental_start} to {config_end}.")
            try:
                df_update = yf.download(ticker, start=incremental_start, end=config_end, progress=False)
                if not df_update.empty:
                    new_data = pd.concat([new_data, df_update])
                else:
                    print(f"No incremental data returned for {ticker}.")
            except Exception as e:
                print(f"Error incrementally updating data for {ticker}: {e}")

    # Insert new data into the database if any new records were downloaded.
    if not new_data.empty:
        print(f"Inserting new data for {ticker} into the database...")
        insert_data(cursor, conn, new_data, ticker)
    else:
        print(f"No new data to insert for {ticker}.")

# ----------------- Main Execution ----------------- #
def Update_DB(db_file, *_, **__):
    """Disabled data download step."""
    print("[Update_DB] Download step disabled – using existing database.")


# ----------------- Calculate and Store Moving Averages ----------------- #
def calculate_and_store_moving_averages(db_path):
    """
    Calculates and stores moving averages for each stock and date in the database.
    Computes 10-day, 20-day, 50-day, 150-day, and 200-day moving averages
    based on the adjusted closing prices, and updates the database accordingly.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ensure moving average columns exist.
        required_columns = ['ma_10', 'ma_20', 'ma_50', 'ma_150', 'ma_200']
        cursor.execute("PRAGMA table_info(stock_data)")
        existing_info = cursor.fetchall()
        existing_columns = [info[1] for info in existing_info]
        for col in required_columns:
            if col not in existing_columns:
                cursor.execute(f"ALTER TABLE stock_data ADD COLUMN {col} REAL")
                print(f"Added column {col} to stock_data table.")

        # Load stock data.
        df = pd.read_sql_query("SELECT * FROM stock_data", conn, parse_dates=['date'])
        if df.empty:
            print("No data found in the database.")
            conn.close()
            return

        df.sort_values(by=['ticker', 'date'], inplace=True)

        # Process each ticker group.
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values(by='date')
            group['ma_10'] = group['adj_close'].rolling(window=10, min_periods=10).mean()
            group['ma_20'] = group['adj_close'].rolling(window=20, min_periods=20).mean()
            group['ma_50'] = group['adj_close'].rolling(window=50, min_periods=50).mean()
            group['ma_150'] = group['adj_close'].rolling(window=150, min_periods=150).mean()
            group['ma_200'] = group['adj_close'].rolling(window=200, min_periods=200).mean()

            # Update rows in the database.
            for index, row in group.iterrows():
                cursor.execute("""
                    UPDATE stock_data
                    SET ma_10 = ?, ma_20 = ?, ma_50 = ?, ma_150 = ?, ma_200 = ?
                    WHERE ticker = ? AND date = ?
                """, (
                    None if pd.isna(row['ma_10']) else float(row['ma_10']),
                    None if pd.isna(row['ma_20']) else float(row['ma_20']),
                    None if pd.isna(row['ma_50']) else float(row['ma_50']),
                    None if pd.isna(row['ma_150']) else float(row['ma_150']),
                    None if pd.isna(row['ma_200']) else float(row['ma_200']),
                    ticker,
                    row['date'].strftime('%Y-%m-%d')
                ))
        conn.commit()
        print("Moving averages calculated and stored successfully.")
    except Exception as e:
        print("Error in calculating and storing moving averages:", e)
    finally:
        conn.close()


# ----------------- Calculate and Store Technical Indicators ----------------- #
def calculate_and_store_other_indicators(db_path):
    """
    Calculates and stores technical indicators for each stock and date in the database.
    
    Indicators computed (using adjusted close for price):
      - MACD, MACD Signal, and MACD Histogram, using 12-day and 26-day EMAs and a 9-day signal.
      - RSI (Relative Strength Index) using a 14-day lookback period.
      - Bollinger Bands (Upper, Middle, Lower) based on a 20-day SMA and ±2 standard deviations.
      - Volume Moving Averages: 10-day and 20-day averages of daily volume.
    
    The function alters the stock_data table to add indicator columns if they don't already exist,
    computes indicator values for each ticker, and then updates each row accordingly.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Define the required indicator columns.
        indicator_columns = [
            'macd', 'macd_signal', 'macd_hist',
            'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'volume_ma_10', 'volume_ma_20'
        ]
        cursor.execute("PRAGMA table_info(stock_data)")
        existing_info = cursor.fetchall()
        existing_columns = [info[1] for info in existing_info]
        for col in indicator_columns:
            if col not in existing_columns:
                cursor.execute(f"ALTER TABLE stock_data ADD COLUMN {col} REAL")
                print(f"Added column {col} to stock_data table.")
        conn.commit()

        # Load all stock data with dates parsed.
        df = pd.read_sql_query("SELECT * FROM stock_data", conn, parse_dates=['date'])
        if df.empty:
            print("No data found in the database.")
            conn.close()
            return

        df.sort_values(by=['ticker', 'date'], inplace=True)

        # Process data for each ticker.
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values(by='date')
            # --- MACD Calculations ---
            ema_12 = group['adj_close'].ewm(span=12, adjust=False).mean()
            ema_26 = group['adj_close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal

            # --- RSI Calculation (14-day) ---
            delta = group['adj_close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # --- Bollinger Bands (20-day) ---
            sma_20 = group['adj_close'].rolling(window=20, min_periods=20).mean()
            std_20 = group['adj_close'].rolling(window=20, min_periods=20).std()
            bb_middle = sma_20
            bb_upper = sma_20 + 2 * std_20
            bb_lower = sma_20 - 2 * std_20

            # --- Volume Moving Averages ---
            vol_ma_10 = group['volume'].rolling(window=10, min_periods=10).mean()
            vol_ma_20 = group['volume'].rolling(window=20, min_periods=20).mean()

            # Assign computed indicator columns.
            group['macd'] = macd
            group['macd_signal'] = macd_signal
            group['macd_hist'] = macd_hist
            group['rsi'] = rsi
            group['bb_upper'] = bb_upper
            group['bb_middle'] = bb_middle
            group['bb_lower'] = bb_lower
            group['volume_ma_10'] = vol_ma_10
            group['volume_ma_20'] = vol_ma_20

            # Update each row in the database.
            for index, row in group.iterrows():
                cursor.execute("""
                    UPDATE stock_data
                    SET macd = ?, macd_signal = ?, macd_hist = ?,
                        rsi = ?,
                        bb_upper = ?, bb_middle = ?, bb_lower = ?,
                        volume_ma_10 = ?, volume_ma_20 = ?
                    WHERE ticker = ? AND date = ?
                """, (
                    None if pd.isna(row['macd']) else float(row['macd']),
                    None if pd.isna(row['macd_signal']) else float(row['macd_signal']),
                    None if pd.isna(row['macd_hist']) else float(row['macd_hist']),
                    None if pd.isna(row['rsi']) else float(row['rsi']),
                    None if pd.isna(row['bb_upper']) else float(row['bb_upper']),
                    None if pd.isna(row['bb_middle']) else float(row['bb_middle']),
                    None if pd.isna(row['bb_lower']) else float(row['bb_lower']),
                    None if pd.isna(row['volume_ma_10']) else float(row['volume_ma_10']),
                    None if pd.isna(row['volume_ma_20']) else float(row['volume_ma_20']),
                    ticker,
                    row['date'].strftime('%Y-%m-%d')
                ))
            conn.commit()

        print("Technical indicators calculated and stored successfully.")
    except Exception as e:
        print("Error in calculating and storing technical indicators:", e)
    finally:
        conn.close()

# ----------------- Calculate and Store Technical Indicators ----------------- #
def calculate_and_store_technical_indicators(db_path):
    """
    Calculates and stores technical indicators for each stock and date in the database.
    
    This function computes:
      - Stochastic Oscillator (%K, %D) using a rolling period (default 14 days for %K 
        and a 3-day SMA of %K for %D),
      - On-Balance Volume (OBV),
    
    For each ticker, it then examines only the latest trading day (using a configurable lookback window,
    default 30 days) and computes:
      - Support level: minimum low over the lookback window,
      - Resistance level: maximum high over the lookback window,
      - Overbought/Oversold status: based on the latest day’s %K compared to thresholds (default >80 for overbought, <20 for oversold).
    
    The computed Stochastic Oscillator and OBV values are updated into the existing 'stock_data' table,
    and the support/resistance and overbought/oversold status are stored in a new table 'latest_indicators'.
    
    Configuration parameters are read from config.ini under the [indicator] section.
    """

    # Read configuration values from config.ini.
    config = configparser.ConfigParser()
    config.read(config_file)
    indicator_conf = config["indicator"] if "indicator" in config else {}

    try:
        stoch_k_period = int(indicator_conf.get("stoch_k_period", 14))
    except:
        stoch_k_period = 14
    try:
        stoch_d_period = int(indicator_conf.get("stoch_d_period", 3))
    except:
        stoch_d_period = 3
    try:
        stoch_overbought = float(indicator_conf.get("stoch_overbought", 80))
    except:
        stoch_overbought = 80.0
    try:
        stoch_oversold = float(indicator_conf.get("stoch_oversold", 20))
    except:
        stoch_oversold = 20.0
    try:
        # Although RSI thresholds are provided, this function bases overbought/oversold solely on stoch %K.
        rsi_overbought = float(indicator_conf.get("rsi_overbought", 70))
    except:
        rsi_overbought = 70.0
    try:
        rsi_oversold = float(indicator_conf.get("rsi_oversold", 30))
    except:
        rsi_oversold = 30.0
    try:
        support_resistance_lookback = int(indicator_conf.get("support_resistance_lookback", 30))
    except:
        support_resistance_lookback = 30

    # Connect to the database.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Ensure the 'stock_data' table has columns for our new daily indicators ---
    indicator_columns = ["stoch_k", "stoch_d", "obv"]
    cursor.execute("PRAGMA table_info(stock_data)")
    existing_info = cursor.fetchall()
    existing_columns = [info[1] for info in existing_info]
    for col in indicator_columns:
        if col not in existing_columns:
            cursor.execute(f"ALTER TABLE stock_data ADD COLUMN {col} REAL")
            print(f"Added column {col} to stock_data table.")
    conn.commit()

    # --- Load all stock data ---
    df = pd.read_sql_query("SELECT * FROM stock_data", conn, parse_dates=['date'])
    if df.empty:
        print("No stock data found in the database.")
        conn.close()
        return

    # Ensure data is sorted per ticker and chronologically.
    df.sort_values(by=['ticker', 'date'], inplace=True)

    # Process each ticker individually.
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values(by='date')
        # --- Compute Stochastic Oscillator ---
        lowest_low = group['low'].rolling(window=stoch_k_period, min_periods=stoch_k_period).min()
        highest_high = group['high'].rolling(window=stoch_k_period, min_periods=stoch_k_period).max()
        stoch_k = 100 * (group['close'] - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=stoch_d_period, min_periods=stoch_d_period).mean()

        # --- Compute OBV ---
        obv = [0]
        closes = group['close'].tolist()
        volumes = group['volume'].tolist()
        for i in range(1, len(group)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[i - 1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[i - 1] - volumes[i])
            else:
                obv.append(obv[i - 1])
        obv_series = pd.Series(obv, index=group.index)

        # Update the group DataFrame with computed indicator columns.
        group['stoch_k'] = stoch_k
        group['stoch_d'] = stoch_d
        group['obv'] = obv_series

        # --- Update the daily indicator columns in the stock_data table ---
        for idx, row in group.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            cursor.execute("""
                UPDATE stock_data
                SET stoch_k = ?, stoch_d = ?, obv = ?
                WHERE ticker = ? AND date = ?
            """, (
                None if pd.isna(row['stoch_k']) else float(row['stoch_k']),
                None if pd.isna(row['stoch_d']) else float(row['stoch_d']),
                None if pd.isna(row['obv']) else float(row['obv']),
                ticker,
                date_str
            ))
        conn.commit()

        # --- For the latest date only: Compute Support/Resistance and Overbought/Oversold ---
        latest_row = group.iloc[-1]
        # Look back over the configurable period.
        lookback_df = group.tail(support_resistance_lookback)
        support_level = lookback_df['low'].min() if not lookback_df.empty else None
        resistance_level = lookback_df['high'].max() if not lookback_df.empty else None

        # Determine overbought/oversold status based on the latest stoch_k.
        latest_stoch_k = latest_row.get('stoch_k', None)
        if latest_stoch_k is None or pd.isna(latest_stoch_k):
            status = "Neutral"
        elif latest_stoch_k > stoch_overbought:
            status = "Overbought"
        elif latest_stoch_k < stoch_oversold:
            status = "Oversold"
        else:
            status = "Neutral"

        # --- Ensure the new table 'latest_indicators' exists ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latest_indicators (
                ticker TEXT PRIMARY KEY,
                latest_support_level REAL,
                latest_resistance_level REAL,
                overbought_oversold_status TEXT
            )
        """)
        conn.commit()

        # Insert or replace the latest indicator values for this ticker.
        cursor.execute("""
            INSERT OR REPLACE INTO latest_indicators 
            (ticker, latest_support_level, latest_resistance_level, overbought_oversold_status)
            VALUES (?, ?, ?, ?)
        """, (ticker, support_level, resistance_level, status))
        conn.commit()

    print("Technical indicators (Stochastic, OBV, Support/Resistance, Overbought/Oversold) calculated and stored successfully.")
    conn.close()

def calculate_trend_indicators(db_path):
    """
    Calculates three trend indicators from the existing daily data and stores them appropriately.
    
    1. ADX (Average Directional Index):
       - Uses Wilder's smoothing method with a configurable period (default: 14).
       - For each (ticker, date), ADX is calculated using the smoothed True Range (TR),
         smoothed +DM, and smoothed -DM.
       - The computed ADX is stored in a new column 'ADX' in the stock_data table.
       
    2. 52-Week High/Low:
       - For each (ticker, date), the highest high and lowest low over a configurable lookback
         (default: 252 trading days, or derived from 52 weeks) are computed.
       - The results are stored in new columns 'High_52w' and 'Low_52w' in the stock_data table.
    
    3. Rate of Change (ROC):
       - Computes ROC over a configurable period (default: 14 days) as:
             ROC = ((Close_today - Close_N_days_ago) / Close_N_days_ago) * 100.
       - ROC values are stored in a separate table 'roc_indicators', keyed by (ticker, date).
    
    Configuration values are read from config.ini under the [indicator] section:
    
        [indicator]
        adx_period = 14
        roc_period = 14
        weeks_lookback_52 = 52   ; or alternatively days_lookback_52 = 252
    
    For repeated runs, existing rows will be updated rather than duplicated.
    """
    import configparser
    import sqlite3
    import pandas as pd
    import numpy as np

    # --- Read Configuration ---
    config = configparser.ConfigParser()
    config.read(config_file)
    indicator_conf = config["indicator"] if "indicator" in config else {}

    try:
        adx_period = int(indicator_conf.get("adx_period", 14))
    except:
        adx_period = 14
    try:
        roc_period = int(indicator_conf.get("roc_period", 14))
    except:
        roc_period = 14
    # Use days_lookback_52 if provided; else use weeks_lookback_52 (approximate trading days ~5 per week)
    try:
        days_lookback_52 = int(indicator_conf.get("days_lookback_52", None))
    except:
        days_lookback_52 = None
    if days_lookback_52 is None:
        try:
            weeks_lookback_52 = int(indicator_conf.get("weeks_lookback_52", 52))
        except:
            weeks_lookback_52 = 52
        days_lookback_52 = weeks_lookback_52 * 5

    # --- Connect to Database ---
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Ensure necessary columns exist in the daily data table (stock_data) ---
    required_daily_cols = ["ADX", "High_52w", "Low_52w"]
    cursor.execute("PRAGMA table_info(stock_data)")
    cols_info = cursor.fetchall()
    existing_cols = [info[1] for info in cols_info]
    for col in required_daily_cols:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE stock_data ADD COLUMN {col} REAL")
            print(f"Added column {col} to stock_data table.")
    conn.commit()

    # --- Ensure ROC table exists ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS roc_indicators (
            ticker TEXT,
            date TEXT,
            ROC REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()

    # --- Load daily data ---
    df = pd.read_sql_query("SELECT * FROM stock_data", conn, parse_dates=["date"])
    if df.empty:
        print("No data found in stock_data table.")
        conn.close()
        return
    df.sort_values(by=["ticker", "date"], inplace=True)

    # Prepare a list to collect ROC records.
    roc_records = []

    # Process each ticker group separately.
    for ticker in df["ticker"].unique():
        df_t = df[df["ticker"] == ticker].copy().reset_index(drop=True)
        df_t.sort_values(by="date", inplace=True)
        length = len(df_t)
        
        # Initialize arrays for True Range (TR), plusDM, minusDM.
        TR = np.zeros(length)
        plusDM = np.zeros(length)
        minusDM = np.zeros(length)
        
        # Compute TR, plusDM, minusDM for each row (starting from index 1).
        for i in range(1, length):
            current_high = df_t.loc[i, "high"]
            current_low = df_t.loc[i, "low"]
            previous_close = df_t.loc[i-1, "close"]
            previous_high = df_t.loc[i-1, "high"]
            previous_low = df_t.loc[i-1, "low"]
            TR[i] = max(current_high - current_low, 
                        abs(current_high - previous_close), 
                        abs(current_low - previous_close))
            delta_high = current_high - previous_high
            delta_low = previous_low - current_low
            plusDM[i] = delta_high if (delta_high > delta_low and delta_high > 0) else 0
            minusDM[i] = delta_low if (delta_low > delta_high and delta_low > 0) else 0
        
        # --- Apply Wilder's Smoothing for TR, +DM, and -DM ---
        smoothed_TR = np.empty(length)
        smoothed_plusDM = np.empty(length)
        smoothed_minusDM = np.empty(length)
        smoothed_TR[:adx_period] = np.nan
        smoothed_plusDM[:adx_period] = np.nan
        smoothed_minusDM[:adx_period] = np.nan
        
        # First smoothed values: simple sum over first adx_period days.
        initial_TR_sum = np.nansum(TR[1:adx_period+1])  # starting at index 1 since index 0 is undefined in DM calculations.
        initial_plusDM_sum = np.nansum(plusDM[1:adx_period+1])
        initial_minusDM_sum = np.nansum(minusDM[1:adx_period+1])
        smoothed_TR[adx_period-1] = initial_TR_sum
        smoothed_plusDM[adx_period-1] = initial_plusDM_sum
        smoothed_minusDM[adx_period-1] = initial_minusDM_sum
        
        # For subsequent values use Wilder's smoothing:
        for i in range(adx_period, length):
            smoothed_TR[i] = (smoothed_TR[i-1]*(adx_period-1) + TR[i]) / adx_period
            smoothed_plusDM[i] = (smoothed_plusDM[i-1]*(adx_period-1) + plusDM[i]) / adx_period
            smoothed_minusDM[i] = (smoothed_minusDM[i-1]*(adx_period-1) + minusDM[i]) / adx_period

        # Compute +DI and -DI for indices >= adx_period-1.
        DI_plus = np.full(length, np.nan)
        DI_minus = np.full(length, np.nan)
        for i in range(adx_period-1, length):
            if smoothed_TR[i] != 0:
                DI_plus[i] = 100 * (smoothed_plusDM[i] / smoothed_TR[i])
                DI_minus[i] = 100 * (smoothed_minusDM[i] / smoothed_TR[i])
            else:
                DI_plus[i] = np.nan
                DI_minus[i] = np.nan

        # Compute DX for indices >= adx_period-1.
        DX = np.full(length, np.nan)
        for i in range(adx_period-1, length):
            sum_DI = DI_plus[i] + DI_minus[i]
            if sum_DI > 0:
                DX[i] = 100 * abs(DI_plus[i] - DI_minus[i]) / sum_DI
            else:
                DX[i] = np.nan

        # Compute ADX using Wilder's method:
        ADX = np.full(length, np.nan)
        # The first ADX value is the average of DX from index (adx_period-1) to index (2*adx_period-2)
        if length >= 2*adx_period - 1:
            first_ADX = np.nanmean(DX[adx_period-1:2*adx_period-1])
            ADX[2*adx_period-2] = first_ADX
            # For subsequent days, use the Wilder's smoothing formula:
            for i in range(2*adx_period-1, length):
                ADX[i] = ((ADX[i-1]*(adx_period-1)) + DX[i]) / adx_period

        # Add the computed ADX to the dataframe.
        df_t["ADX"] = ADX
        
        # --- Calculate 52-Week High/Low ---
        df_t["High_52w"] = df_t["high"].rolling(window=days_lookback_52, min_periods=days_lookback_52).max()
        df_t["Low_52w"] = df_t["low"].rolling(window=days_lookback_52, min_periods=days_lookback_52).min()
        
        # --- Calculate Rate of Change (ROC) ---
        # ROC = ((Close_today - Close_N_days_ago) / Close_N_days_ago) * 100
        df_t["ROC_calc"] = (df_t["close"] - df_t["close"].shift(roc_period)) / df_t["close"].shift(roc_period) * 100
        
        # Collect ROC records for rows where ROC is computed.
        for idx, row in df_t.iterrows():
            if idx >= roc_period and pd.notna(row["ROC_calc"]):
                roc_record = {
                    "ticker": ticker,
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "ROC": float(row["ROC_calc"])
                }
                roc_records.append(roc_record)
        
        # --- Update daily table (stock_data) with ADX, High_52w, and Low_52w for this ticker ---
        for idx, row in df_t.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            cursor.execute("""
                UPDATE stock_data
                SET ADX = ?,
                    High_52w = ?,
                    Low_52w = ?
                WHERE ticker = ? AND date = ?
            """, (
                None if pd.isna(row["ADX"]) else float(row["ADX"]),
                None if pd.isna(row["High_52w"]) else float(row["High_52w"]),
                None if pd.isna(row["Low_52w"]) else float(row["Low_52w"]),
                ticker,
                date_str
            ))
        conn.commit()
    
    # --- Insert ROC records into roc_indicators table ---
    for record in roc_records:
        cursor.execute("""
            INSERT OR REPLACE INTO roc_indicators (ticker, date, ROC)
            VALUES (?, ?, ?)
        """, (record["ticker"], record["date"], record["ROC"]))
    conn.commit()
    
    print("Trend indicators (ADX, 52-Week High/Low, ROC) calculated and stored successfully.")
    conn.close()



# ----------------- 导出趋势分榜单 ----------------- #
# ----------------- 导出趋势分榜单 ----------------- #
def export_trend_excel(df: pd.DataFrame, out_path: str | None = None):
    """
    导出趋势评分榜单到 Excel / CSV。

    选取顺序：
    1. 显式传入 out_path                           → 优先。
    2. config_trend.ini → [trend_score] → excel_file_name
    3. 若两者皆无，则使用固定 'trend_scores.xlsx'
    """
    from pathlib import Path
    import configparser, logging

    logger = logging.getLogger("TrendExport")

    # -- 读取配置 --
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).with_name("config_trend.ini"))
    cfg_out = None
    if cfg.has_section("trend_score") and cfg.has_option("trend_score", "excel_file_name"):
        cfg_out = cfg["trend_score"]["excel_file_name"].strip()

    # -- 决定文件名 --
    if out_path:           # ① 函数参数
        out_file = Path(out_path)
    elif cfg_out:          # ② 配置文件
        out_file = Path(cfg_out)
    else:                  # ③ 默认
        out_file = Path("trend_scores.xlsx")

    # -- 导出 --
    try:
        df.sort_values("TotalScore", ascending=False).to_excel(out_file, index=False)
        logger.info("[TrendScore] Exported → %s", out_file)
    except Exception as e:                    # Excel 写失败则回退 CSV
        alt = out_file.with_suffix(".csv")
        df.to_csv(alt, index=False)
        logger.warning("[TrendScore] Excel failed (%s) – wrote CSV %s", e, alt)


# ----------------- Calculate, Store & Export Trend Scores ----------------- #
def calculate_and_store_trend_scores(db_path, config_path="config.ini"):
    """
    为数据库中所有股票计算 Trend Score，存入表 trend_scores，
    并将结果按得分降序导出为 Excel（文件名可在 config.ini 的 [trend_score]
    区段通过 excel_file_name 参数自定义，默认 trend_scores.xlsx）。

    本实现全部使用 NumPy 计算线性回归斜率，不依赖 scikit‑learn。
    """
    # ---------- 依赖 ----------
    import os, math, datetime, sqlite3, configparser
    import pandas as pd, numpy as np

    # ---------- 小工具：用 numpy.polyfit 计算斜率 ----------
    def _slope_np(arr: np.ndarray) -> float:
        """
        输入一维等间隔序列，返回线性回归斜率 k。
        若有效样本 <2 或全为 NaN，则返回 0。
        """
        arr = np.asarray(arr, dtype=float)
        mask = ~np.isnan(arr)
        if mask.sum() < 2:
            return 0.0
        y = arr[mask]
        x = np.arange(len(arr))[mask].astype(float)
        k, _ = np.polyfit(x, y, 1)  # 一阶多项式系数 [k, b]
        return float(k)

    # ---------- 读取配置 ----------
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    tc = cfg["trend_score"] if "trend_score" in cfg else {}

    slope_window = int(tc.get("slope_window", 20))
    vol_ma_long  = int(tc.get("vol_ma_long", 120))
    rs_window    = int(tc.get("rs_window", 63))   # ← 就放这里
    excel_name   = tc.get("excel_file_name", "trend_scores.xlsx")

    # 权重（总和≈100，若不同自动归一）
    w = {
        "ma_structure": float(tc.get("w_ma_structure", 15)),
        "sma_slope"   : float(tc.get("w_sma_slope"  , 10)),
        "adx"         : float(tc.get("w_adx"        , 15)),
        "roc"         : float(tc.get("w_roc"        , 10)),
        "rsi"         : float(tc.get("w_rsi"        ,  5)),
        "macd_slope"  : float(tc.get("w_macd_slope" , 10)),
        "rs"          : float(tc.get("w_rs"         ,  10)),   # 默认 0，未实现 RS
        "obv_slope"   : float(tc.get("w_obv_slope"  , 10)),
        "pct_high"    : float(tc.get("w_pct_high"   , 10)),
        "vol_ratio"   : float(tc.get("w_vol_ratio"  ,  5))
    }

    tot_w = sum(w.values())
    if not math.isclose(tot_w, 100.0):
        w = {k: v / tot_w * 100 for k, v in w.items()}

    # ---------- 连接数据库 ----------
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 若无 volume_ma_120 列则添加
    cursor.execute("PRAGMA table_info(stock_data)")
    existing_cols = [c[1] for c in cursor.fetchall()]
    if "volume_ma_120" not in existing_cols:
        cursor.execute("ALTER TABLE stock_data ADD COLUMN volume_ma_120 REAL")
        conn.commit()

    # 建立 trend_scores 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trend_scores (
            ticker TEXT PRIMARY KEY,
            score  REAL,
            asof   TEXT
        )
    """)
    conn.commit()

    # ---------- 读取 stock_data ----------
    df = pd.read_sql_query("SELECT * FROM stock_data", conn, parse_dates=["date"])
    if df.empty:
        print("[TrendScore] stock_data 为空，终止计算。")
        conn.close()
        return
    df.sort_values(["ticker", "date"], inplace=True)
    # --- 准备 SPY 收益序列 ---
    spx_ret = pd.Series(dtype=float)
    if "SPY" in df["ticker"].unique():
        spx = df[df["ticker"] == "SPY"].copy()
        spx.sort_values("date", inplace=True)
        spx["ret_rs"] = spx["adj_close"].pct_change(rs_window)
        spx_ret = spx.set_index("date")["ret_rs"]
    else:
        print("[RS] 未找到 SPY 数据，RS 得分将置 0；如需启用，请将 'SPY' 添加到下载列表。")

    # 填补 volume_ma_120（如果缺失）
    upd_rows = []
    for tk, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        vol120 = grp["volume"].rolling(vol_ma_long, min_periods=vol_ma_long).mean()
        missing = grp["volume_ma_120"].isna() & vol120.notna()
        if missing.any():
            df.loc[grp[missing].index, "volume_ma_120"] = vol120[missing]
            for idx in grp[missing].index:
                row = df.loc[idx]
                upd_rows.append((float(row["volume_ma_120"]),
                                 row["ticker"],
                                 row["date"].strftime("%Y-%m-%d")))
    if upd_rows:
        cursor.executemany("UPDATE stock_data SET volume_ma_120=? WHERE ticker=? AND date=?", upd_rows)
        conn.commit()

    # 读取 ROC 表（若存在）
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    roc_df = (pd.read_sql_query("SELECT * FROM roc_indicators", conn, parse_dates=["date"])
              if "roc_indicators" in tables["name"].tolist()
              else pd.DataFrame(columns=["ticker", "date", "ROC"]))

    # ---------- 逐股票计算 ----------
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    score_records = []

    for tk, grp in df.groupby("ticker"):
        grp = grp.sort_values("date")
        if len(grp) < slope_window:
            continue
        recent = grp.tail(slope_window)
        last   = recent.iloc[-1]

        # 1) 均线多头排列
        ma_score = 0
        if all(pd.notna(last[["ma_10", "ma_20", "ma_50", "ma_150"]])):
            conds = [last["ma_10"] > last["ma_20"],
                     last["ma_20"] > last["ma_50"],
                     last["ma_50"] > last["ma_150"]]
            ma_score = conds.count(True) / 3 * 100

        # 2) 50日SMA斜率
        if grp["ma_50"].notna().tail(slope_window).all():
            slope50 = _slope_np(grp["ma_50"].tail(slope_window).values)
            sma_slope_score = np.clip((slope50 / last["ma_50"]) * 1e4, -1, 1)
            sma_slope_score = (sma_slope_score + 1) / 2 * 100
        else:
            sma_slope_score = 0

        # 3) ADX
        adx_score = np.clip((last["ADX"] - 15) / 25 * 100, 0, 100) if pd.notna(last["ADX"]) else 0

        # 4) ROC
        roc_val = roc_df.loc[(roc_df["ticker"] == tk) &
                             (roc_df["date"] == last["date"]), "ROC"]
        roc_val = roc_val.iloc[0] if not roc_val.empty else np.nan
        roc_score = np.clip((roc_val / 20) * 50 + 50, 0, 100) if pd.notna(roc_val) else 0

        # 5) RSI
        rsi_score = np.clip(100 - abs(last["rsi"] - 50) * 2, 0, 100) if pd.notna(last["rsi"]) else 0

        # 6) MACD_hist 斜率
        if grp["macd_hist"].notna().tail(slope_window).all():
            macd_s = _slope_np(grp["macd_hist"].tail(slope_window).values)
            macd_slope_score = np.clip((macd_s / (abs(grp["macd_hist"].tail(slope_window).std()) + 1e-9)) * 50 + 50,
                                       0, 100)
        else:
            macd_slope_score = 0

        # 7) RS：过去 rs_window 天超额收益
        if not spx_ret.empty and len(grp) > rs_window:
            stock_ret = grp.set_index("date")["adj_close"].pct_change(rs_window).iloc[-1]
            idx_ret = spx_ret.reindex([last["date"]]).iloc[0] if last["date"] in spx_ret.index else np.nan
            if pd.notna(stock_ret) and pd.notna(idx_ret):
                rs_val = stock_ret - idx_ret
                rs_score = np.clip((rs_val / 0.20) * 50 + 50, 0, 100)
            else:
                rs_score = 0
        else:
            rs_score = 0

        # 8) OBV 斜率
        if grp["obv"].notna().tail(slope_window).all():
            obv_s = _slope_np(grp["obv"].tail(slope_window).values)
            obv_slope_score = np.clip((obv_s / (abs(grp["obv"].tail(slope_window).std()) + 1e-9)) * 50 + 50,
                                      0, 100)
        else:
            obv_slope_score = 0

        # 9) 距 52 周新高（0%→100，-10%→50，-20%→0）
        if pd.notna(last["High_52w"]) and last["High_52w"] > 0:
            pct_from_high = (last["High_52w"] - last["close"]) / last["High_52w"]  # 正值
            pct_high_score = np.clip((1 - pct_from_high / 0.20) * 100, 0, 100)
        else:
            pct_high_score = 0

        # 10) 放量倍数（安全处理 NaN）
        if pd.notna(last["volume_ma_120"]) and last["volume_ma_120"] > 0:
           vm20 = 0.0 if pd.isna(last["volume_ma_20"]) else float(last["volume_ma_20"])
           vol_ratio = vm20 / last["volume_ma_120"]
           vol_ratio_score = np.clip((np.log10(max(vol_ratio, 1e-6)) + 0.3) / 0.6 * 100, 0, 100)
        else:
           vol_ratio_score = 0
        
        # ---- 将任何 NaN 分值归零，避免 NaN 向下传播 ----
        for _v in ["ma_score", "sma_slope_score", "adx_score", "roc_score", "rsi_score",
           "macd_slope_score", "rs_score", "obv_slope_score", "pct_high_score", "vol_ratio_score"]:
            if pd.isna(locals()[_v]):
                locals()[_v] = 0.0

        # ---------- 综合得分 ----------
        total = (
            w["ma_structure"] * ma_score +
            w["sma_slope"]   * sma_slope_score +
            w["adx"]         * adx_score +
            w["roc"]         * roc_score +
            w["rsi"]         * rsi_score +
            w["macd_slope"]  * macd_slope_score +
            w["rs"]          * rs_score +
            w["obv_slope"]   * obv_slope_score +
            w["pct_high"]    * pct_high_score +
            w["vol_ratio"]   * vol_ratio_score
        ) / 100.0

        record = {
            "Ticker": tk,
            "TotalScore": round(total, 2),
            "MA_struct": round(ma_score, 2),
            "SMA50_slope": round(sma_slope_score, 2),
            "ADX14": round(adx_score, 2),
            "ROC%": round(roc_score, 2),
            "RSI14": round(rsi_score, 2),
            "MACDhist_slope": round(macd_slope_score, 2),
            "RS": round(rs_score, 2),
            "OBV_slope": round(obv_slope_score, 2),
            "Pct_to_High": round(pct_high_score, 2),
            "Vol_ratio": round(vol_ratio_score, 2),
            "AsOf": today_str
        }
        score_records.append(record)


    # ---------- 写入 trend_scores ----------
    cursor.executemany("""
        INSERT OR REPLACE INTO trend_scores (ticker, score, asof)
        VALUES (:Ticker, :TotalScore, :AsOf)
    """, score_records)
    conn.commit()

    # ---------- 导出 Excel ----------
    score_df = pd.DataFrame(score_records)
    score_df.sort_values("TotalScore", ascending=False, inplace=True)


    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), excel_name)
    try:
        export_trend_excel(score_df)        # score_df 是计算完的 DataFrame
        print(f"[TrendScore] 已按得分降序导出 Excel: {save_path}")
    except Exception as e:
        alt_path = save_path.rsplit(".", 1)[0] + ".csv"
        score_df.to_csv(alt_path, index=False)
        print(f"[TrendScore] 写 Excel 失败, 已输出 CSV: {alt_path} ({e})")

    print(f"[TrendScore] 完成：{len(score_records)} 支股票写入 trend_scores。")
    conn.close()

# ----------------- Hard Reset DB ----------------- #
def reset_database(db_path):
    """删除旧数据库文件，触发全量重建。"""
    import os
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"[reset_database] Removed database file {db_path}.")
        except Exception as e:
            print(f"[reset_database] Could not delete {db_path}: {e}")
    else:
        print(f"[reset_database] {db_path} does not exist – no need to delete.")
# ----------------- Process Control Wrapper ----------------- #
def run_process_control(stage: int, db_path: str = DEFAULT_DB_FILE):
    """
    根据 stage 参数控制流程：
      0 → 全流程(强制全量更新DB)
      1 → 仅 Update_DB
      2 → 仅指标计算（不含 TrendScore）
      3 → 仅 TrendScore 计算
    """
    if stage == 0:
        reset_database(db_path)
        Update_DB(db_path)
        calculate_and_store_moving_averages(db_path)
        calculate_and_store_other_indicators(db_path)
        calculate_and_store_technical_indicators(db_path)
        calculate_trend_indicators(db_path)
        calculate_and_store_trend_scores(db_path)
    elif stage == 1:
        Update_DB(db_path)
    elif stage == 2:
        calculate_and_store_moving_averages(db_path)
        calculate_and_store_other_indicators(db_path)
        calculate_and_store_technical_indicators(db_path)
        calculate_trend_indicators(db_path)
    elif stage == 3:
        calculate_and_store_trend_scores(db_path)
    else:
        print(f"[run_process_control] 未识别的 stage 参数: {stage}，不执行任何操作。")

if __name__ == '__main__':
    # 从 config.ini 读取 runstage，默认 0
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    runstage = int(cfg.get("run_control", "runstage", fallback="0"))
    print(f"[Main] runstage = {runstage}")
    run_process_control(runstage, _get_price_db())

    