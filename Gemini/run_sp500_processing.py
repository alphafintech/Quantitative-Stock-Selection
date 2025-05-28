# -*- coding: utf-8 -*-
import logging
import time
from datetime import datetime
import os
import sys
import traceback
import pandas as pd
import configparser
import numpy as np
import argparse # Added for command-line control
import shutil



from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE_TREND  = BASE_DIR / 'config_trend.ini'
CONFIG_FILE_GROWTH = BASE_DIR / 'config_finance.ini'
CONFIG_FILE_RUN    = BASE_DIR / 'config_run.ini'

# --- Import necessary functions from other scripts ---
try:
    # Import functions for Trend Score calculation
    from .compute_trend_score_sp500 import (
        load_configuration as load_trend_config, # Rename to avoid conflict
        create_connection,
        create_tables,
        update_stock_data,
        calculate_all_indicators,
        calculate_and_save_trend_scores,
        CONFIG as TREND_CONFIG # Import the config dict used by trend functions
    )
    # Import the main function for Growth Score calculation
    # *** IMPORTANT: Assumes you have renamed 'Compute_growth_score_S&P500.py' to 'compute_growth_score_sp500.py' ***
    from .Compute_growth_score_sp500 import compute_growth_score

    # Optional: Check for pandas_ta dependency if needed directly here (unlikely)
    try:
        import pandas_ta
    except ImportError:
        print("Warning: 'pandas_ta' library not found. Ensure it's installed if indicator calculations fail.")

except ImportError as e:
    print(f"Error: Failed to import necessary functions or variables.")
    print(f"Please ensure 'compute_trend_score_sp500.py' and 'compute_growth_score_sp500.py' are in the same directory or Python path ({sys.path}).")
    print(f"Import Error: {e}")
    print("-" * 20 + " Traceback " + "-" * 20)
    traceback.print_exc()
    print("-" * 50)
    sys.exit(1) # Exit if essential imports fail
except Exception as e_import:
    print(f"Unexpected error during imports: {e_import}")
    print("-" * 20 + " Traceback " + "-" * 20)
    traceback.print_exc()
    print("-" * 50)
    sys.exit(1)

# --- Basic Logging Setup ---
# More detailed logging might be configured within the imported functions based on their config files
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Runner] %(message)s')

# --- Configuration Loading (for this script) ---

def load_run_config(config_file=CONFIG_FILE_RUN):
    """Loads configuration from the run configuration INI file."""
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Run configuration file '{config_file}' not found.")
    config.read(config_file, encoding="utf-8")
    logging.info(f"Run configuration loaded from '{config_file}'.")
    return config

# --- Data Loading and Preprocessing (for final screening) ---

def find_column(df_columns, potential_names):
    """Finds the actual column name from a list of potential names (case-insensitive)."""
    for name in potential_names:
        for col in df_columns:
            if col.lower() == name.lower():
                return col
    # If loop finishes without finding, raise error
    raise ValueError(f"Could not find any of the columns: {potential_names} in the DataFrame columns: {df_columns}")

def load_data(filepath, ticker_col_config, score_col_config):
    """Loads data from an Excel file and prepares it for screening."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file '{filepath}' not found.")
    logging.debug(f"Loading data from: {filepath}")
    df = pd.read_excel(filepath)
    logging.debug(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    # Find the actual ticker and score column names based on config
    ticker_col_actual = find_column(df.columns, [ticker_col_config, 'ticker', 'Ticker']) # Check common variations
    score_col_actual = find_column(df.columns, [score_col_config])
    logging.debug(f"Identified Ticker column: '{ticker_col_actual}', Score column: '{score_col_actual}'")

    # Select and rename columns for consistency
    df_renamed = df[[ticker_col_actual, score_col_actual]].rename(columns={
        ticker_col_actual: 'Ticker',
        score_col_actual: 'Score'
    })

    # Standardize Ticker format (string, uppercase)
    df_renamed['Ticker'] = df_renamed['Ticker'].astype(str).str.upper()
    # Ensure Score is numeric, coercing errors to NaN
    df_renamed['Score'] = pd.to_numeric(df_renamed['Score'], errors='coerce')
    nan_scores = df_renamed['Score'].isna().sum()
    if nan_scores > 0:
        logging.warning(f"Found {nan_scores} non-numeric/NaN scores in '{score_col_actual}' from {filepath}. They will be treated as NaN.")

    logging.debug(f"Finished loading and renaming. Shape: {df_renamed.shape}")
    return df_renamed

def normalize_to_percentile(series):
    """
    Converts a pandas Series of scores into percentile ranks (0-100).
    Higher scores get higher percentile ranks. NaNs are kept as NaNs.
    """
    # Use na_option='keep' to preserve NaNs
    percentiles = series.rank(pct=True, na_option='keep') * 100
    logging.debug(f"Calculated percentiles. Original NaNs: {series.isna().sum()}, Percentile NaNs: {percentiles.isna().sum()}")
    return percentiles

# --- Screening Methods (for final screening) ---

def combined_weighted_score(df, trend_weight, fundamental_weight):
    """
    Calculates the combined weighted score based on percentile ranks.
    Handles potential NaN percentiles by treating them as 0 for weighting,
    but keeps the final Combined_Score NaN if *both* inputs were NaN.
    """
    if 'Trend_Percentile' not in df.columns or 'Fundamental_Percentile' not in df.columns:
        raise ValueError("Input DataFrame must contain 'Trend_Percentile' and 'Fundamental_Percentile' columns.")

    trend_perc = df['Trend_Percentile']
    fund_perc = df['Fundamental_Percentile']

    # Calculate combined score, filling NaNs with 0 ONLY for the calculation
    df['Combined_Score'] = (trend_perc.fillna(0) * trend_weight +
                            fund_perc.fillna(0) * fundamental_weight)

    # If both original percentiles were NaN, the Combined_Score should also be NaN
    df.loc[trend_perc.isna() & fund_perc.isna(), 'Combined_Score'] = np.nan

    # Rank based on the combined score, putting NaNs last
    df_sorted = df.sort_values(by='Combined_Score', ascending=False, na_position='last').reset_index(drop=True)
    logging.debug("Calculated and sorted by Combined_Weighted_Score.")
    return df_sorted

def dual_threshold_filter(df, min_trend_percentile, min_fundamental_percentile):
    """
    Filters the DataFrame based on minimum percentile thresholds.
    Rows must meet *both* thresholds to pass. NaNs automatically fail.
    """
    if 'Trend_Percentile' not in df.columns or 'Fundamental_Percentile' not in df.columns:
         raise ValueError("Input DataFrame must contain 'Trend_Percentile' and 'Fundamental_Percentile' columns.")

    # Filter conditions (NaNs will evaluate to False)
    condition = (
        (df['Trend_Percentile'] >= min_trend_percentile) &
        (df['Fundamental_Percentile'] >= min_fundamental_percentile)
    )
    filtered_df = df[condition].copy() # Use .copy() to avoid SettingWithCopyWarning
    logging.debug(f"Applied dual threshold filter: {len(filtered_df)} rows passed.")

    # Sort the filtered results (e.g., by fundamental percentile, then trend)
    filtered_df_sorted = filtered_df.sort_values(
        by=['Fundamental_Percentile', 'Trend_Percentile'],
        ascending=[False, False],
        na_position='last' # Keep NaNs last, though they shouldn't be present after filter
    ).reset_index(drop=True)
    logging.debug("Sorted filtered results by Fundamental, then Trend percentile.")

    return filtered_df_sorted

# --- Saving Results (for final screening) ---

def save_results(df_weighted, df_threshold, output_dir, filename):
    """Saves the screening results into an Excel file with two sheets."""
    # Ensure the output directory exists
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Failed to create output directory '{output_dir}': {e}")
            # Fallback: Save to current directory if dir creation fails
            output_dir = "."
            logging.warning(f"Attempting to save results to current directory: {os.getcwd()}")

    output_path = os.path.join(output_dir or ".", filename) # Handle case where output_dir is empty or None

    # Use ExcelWriter to save multiple sheets
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Round numeric columns before saving for better readability
            numeric_cols_w = df_weighted.select_dtypes(include=np.number).columns
            df_weighted[numeric_cols_w] = df_weighted[numeric_cols_w].round(4)
            df_weighted.to_excel(writer, sheet_name='Combined_Weighted_Score', index=False)

            numeric_cols_t = df_threshold.select_dtypes(include=np.number).columns
            df_threshold[numeric_cols_t] = df_threshold[numeric_cols_t].round(4)
            df_threshold.to_excel(writer, sheet_name='Dual_Threshold_Filter', index=False)
        logging.info(f"Screening results successfully saved to {output_path}")
        print(f"Screening results successfully saved to {output_path}")
    except PermissionError:
        logging.error(f"Permission denied writing to {output_path}. Check file permissions or if the file is open.")
        print(f"Error: Permission denied writing to {output_path}.")
    except Exception as e:
        logging.error(f"Error saving screening results to {output_path}: {e}", exc_info=True)
        print(f"Error saving screening results: {e}")

# --- Pipeline Functions ---

def run_trend_score_pipeline(config_file=CONFIG_FILE_TREND, do_calculate_indicators=True, do_calculate_trend_score=True):
    """
    Runs the indicator calculation and trend score calculation steps.
    The indicator database is recreated from ``price_db`` every run and no
    data update is performed.

    Args:
        config_file (str): Path to the configuration file for trend calculations.
        do_calculate_indicators (bool): If True, run the indicator calculation step.
        do_calculate_trend_score (bool): If True, run the trend score calculation step.

    Returns:
        bool: True if all executed steps completed without critical errors, False otherwise.
    """
    pipeline_start_time = time.time()
    logging.info(f"--- Starting Trend Score Pipeline --- Config: '{config_file}' ---")
    print(f"\n--- Starting Trend Score Pipeline ---")
    print(f"Parameters: CalcIndicators={do_calculate_indicators}, CalcTrendScore={do_calculate_trend_score}")

    conn = None
    pipeline_successful = True

    try:
        # 1. Load Trend Configuration (uses the imported function)
        # This updates the TREND_CONFIG dictionary used by the imported functions
        global TREND_CONFIG # Ensure we modify the global TREND_CONFIG
        TREND_CONFIG = load_trend_config(config_file)
        logging.info("Trend configuration loaded.")

        # Determine database paths from configuration
        db_conf = TREND_CONFIG.get('database', {})
        price_db = db_conf.get('db_file')
        indicator_db = db_conf.get('indicator_db_file', os.path.join(BASE_DIR, 'trend_analysis.db'))
        # Resolve relative indicator DB path to project root
        if not os.path.isabs(indicator_db):
            project_root = BASE_DIR.parent
            indicator_db = os.path.join(project_root, indicator_db)
        if not price_db:
            logging.critical("Database file path not found in trend configuration.")
            print("Error: Database file path not found in trend configuration.")
            pipeline_successful = False
            return
        # Resolve price DB path similar to create_connection
        price_db_path = price_db
        if not os.path.isabs(price_db_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            price_db_path = os.path.join(project_root, price_db_path)

        logging.info(f"Using indicator database file: {indicator_db}")

        if not os.path.exists(price_db_path):
            logging.error(f"Price database '{price_db_path}' does not exist.")
            pipeline_successful = False
            return

        # Recreate indicator database from the price database every run
        if os.path.exists(indicator_db):
            try:
                os.remove(indicator_db)
            except PermissionError:
                logging.error(f"无法删除数据库 {indicator_db}，文件可能被占用。")
                raise
        try:
            os.makedirs(os.path.dirname(indicator_db), exist_ok=True)
            shutil.copy2(price_db_path, indicator_db)
            logging.info(f"Copied price DB to indicator DB: {indicator_db}")
        except Exception as e:
            logging.error(f"无法创建指标数据库 '{indicator_db}': {e}")
            pipeline_successful = False
            return

        TREND_CONFIG['database']['db_file'] = indicator_db

        # 2. Create Database Connection
        conn = create_connection(indicator_db)
        if not conn:
            logging.critical("Failed to establish database connection for trend pipeline.")
            print("Error: Failed to establish database connection for trend pipeline.")
            pipeline_successful = False
            return
        logging.info("Database connection established.")

        # 3. Ensure Tables Exist
        # Uses the imported function, assumes it handles all necessary tables
        create_tables(conn)
        logging.info("Database tables checked/created.")

        # --- Step 4: Data Update --- (skipped)
        logging.info("--- Skipping Data Update (database copied from price_db) ---")
        print("--- Skipping Data Update (database copied from price_db) ---")

        # --- Step 5: Indicator Calculation ---
        if do_calculate_indicators:
            # Proceed only if previous steps were generally successful or if forced
            if pipeline_successful:
                step_start = time.time()
                logging.info("--- Running Indicator Calculation ---")
                print("--- Running Indicator Calculation ---")
                calc_success = calculate_all_indicators(conn) # Call imported function
                if not calc_success:
                    logging.warning("Indicator calculation step reported potential issues.")
                    # pipeline_successful = False
                logging.info(f"--- Indicator Calculation finished (Duration: {time.time() - step_start:.2f}s) ---")
                print(f"--- Indicator Calculation finished (Duration: {time.time() - step_start:.2f}s) ---")
            else:
                logging.warning("Skipping Indicator Calculation due to previous errors.")
                print("Skipping Indicator Calculation due to previous errors.")
        else:
            logging.info("--- Skipping Indicator Calculation ---")
            print("--- Skipping Indicator Calculation ---")

        # --- Step 6: Trend Score Calculation ---
        if do_calculate_trend_score:
            # Proceed only if previous steps were generally successful or if forced
            if pipeline_successful:
                step_start = time.time()
                logging.info("--- Running Trend Score Calculation ---")
                print("--- Running Trend Score Calculation ---")
                # Warn if indicators weren't calculated in this run
                if not do_calculate_indicators:
                    logging.warning("Calculating trend scores without running indicator calculation in the same session. Ensure indicators are up-to-date.")
                    print("Warning: Calculating trend scores based on potentially non-current indicators.")

                trend_success = calculate_and_save_trend_scores(conn) # Call imported function
                if not trend_success:
                    logging.warning("Trend score calculation step reported potential issues.")
                    # pipeline_successful = False
                logging.info(f"--- Trend Score Calculation finished (Duration: {time.time() - step_start:.2f}s) ---")
                print(f"--- Trend Score Calculation finished (Duration: {time.time() - step_start:.2f}s) ---")
            else:
                logging.warning("Skipping Trend Score Calculation due to previous errors.")
                print("Skipping Trend Score Calculation due to previous errors.")
        else:
            logging.info("--- Skipping Trend Score Calculation ---")
            print("--- Skipping Trend Score Calculation ---")

    except FileNotFoundError as e:
        logging.error(f"Configuration file error: {e}")
        print(f"Error: {e}")
        pipeline_successful = False
    except KeyError as e:
         logging.error(f"Configuration key error: Missing key {e}. Check '{config_file}'.")
         print(f"Configuration Error: Missing key {e}. Check '{config_file}'.")
         pipeline_successful = False
    except Exception as e:
        logging.error(f"Unexpected error during trend score pipeline: {e}", exc_info=True)
        print(f"Unexpected Error during trend score pipeline: {e}")
        pipeline_successful = False
    finally:
        # Ensure database connection is closed
        if conn:
            conn.close()
            logging.info("Trend pipeline database connection closed.")
        pipeline_duration = time.time() - pipeline_start_time
        logging.info(f"--- Trend Score Pipeline Finished --- Duration: {pipeline_duration:.2f}s --- Status: {'Success' if pipeline_successful else 'Failed'} ---")
        print(f"--- Trend Score Pipeline Finished --- Duration: {pipeline_duration:.2f}s ---")
        return pipeline_successful

# MODIFIED: compute_trend_score now accepts parameters for finer control
def compute_trend_score(calculate_indicators=True, calculate_score=True):
    """
    Wrapper function to configure and run the trend score pipeline.
    The indicator database is always recreated from ``price_db`` so there is no
    separate data update step.

    Args:
        calculate_indicators (bool): If True, run the indicator calculation step.
        calculate_score (bool): If True, run the trend score calculation step.

    Returns:
        bool: True if the pipeline completed successfully, False otherwise.
    """
    logging.info(">>> Initiating Trend Score Computation <<<")
    print(">>> Initiating Trend Score Computation <<<")

    # Pass the control parameters down to the pipeline runner
    success = run_trend_score_pipeline(
        config_file=CONFIG_FILE_TREND,
        do_calculate_indicators=calculate_indicators,
        do_calculate_trend_score=calculate_score
    )
    if not success:
        logging.error("Trend score computation pipeline failed.")
        print("ERROR: Trend score computation pipeline failed.")
    else:
        logging.info(">>> Trend Score Computation Completed <<<")
        print(">>> Trend Score Computation Completed <<<")
    return success


def final_screen():
    """
    Performs the final screening by loading trend and fundamental scores,
    applying weighting and threshold methods, and saving the results.
    """
    logging.info(">>> Initiating Final Screening Process <<<")
    print("\n--- Starting Final Screening Process ---")
    screening_successful = True
    try:
        # 1. Load Run Configuration
        config = load_run_config(CONFIG_FILE_RUN) # Uses the specific run config

        # Extract parameters from config
        files_config = config['FILES']
        cols_config = config['COLUMNS']
        weights_config = config['WEIGHTED_SCORE']
        threshold_config = config['THRESHOLD_FILTER']

        trend_file = files_config.get('trend_file')
        fundamental_file = files_config.get('fundamental_file')
        output_dir = files_config.get('output_dir', 'results') # Default output dir
        output_filename = files_config.get('output_filename', 'screened_stocks.xlsx')

        trend_ticker_col = cols_config.get('trend_ticker_col', 'Ticker')
        trend_score_col = cols_config.get('trend_score_col', 'Normalized_Trend_Score')
        fundamental_ticker_col = cols_config.get('fundamental_ticker_col', 'ticker')
        fundamental_score_col = cols_config.get('fundamental_score_col', 'Overall_Score')

        trend_weight = config.getfloat('WEIGHTED_SCORE', 'trend_weight', fallback=0.5)
        fundamental_weight = config.getfloat('WEIGHTED_SCORE', 'fundamental_weight', fallback=0.5)
        min_trend_percentile = config.getfloat('THRESHOLD_FILTER', 'min_trend_percentile', fallback=0)
        min_fundamental_percentile = config.getfloat('THRESHOLD_FILTER', 'min_fundamental_percentile', fallback=0)

        # Validate weights sum to 1
        if not np.isclose(trend_weight + fundamental_weight, 1.0):
            logging.warning("Weights in [WEIGHTED_SCORE] do not sum to 1. Normalizing...")
            print("Warning: Weights do not sum to 1. Normalizing...")
            total_weight = trend_weight + fundamental_weight
            if total_weight > 1e-6: # Avoid division by zero
                    trend_weight /= total_weight
                    fundamental_weight /= total_weight
            else:
                    trend_weight = 0.5
                    fundamental_weight = 0.5
            logging.info(f"Using normalized weights: Trend={trend_weight:.2f}, Fundamental={fundamental_weight:.2f}")
            print(f"Using normalized weights: Trend={trend_weight:.2f}, Fundamental={fundamental_weight:.2f}")

        # 2. Load Data Files
        logging.info(f"Loading trend data from: {trend_file}")
        print(f"Loading trend data from: {trend_file}")
        df_trend = load_data(trend_file, trend_ticker_col, trend_score_col)
        df_trend = df_trend.rename(columns={'Score': 'Trend_Score'})
        logging.info(f"Loaded {len(df_trend)} trend records.")
        print(f"Loaded {len(df_trend)} trend records.")

        logging.info(f"Loading fundamental data from: {fundamental_file}")
        print(f"Loading fundamental data from: {fundamental_file}")
        df_fundamental = load_data(fundamental_file, fundamental_ticker_col, fundamental_score_col)
        df_fundamental = df_fundamental.rename(columns={'Score': 'Fundamental_Score'})
        logging.info(f"Loaded {len(df_fundamental)} fundamental records.")
        print(f"Loaded {len(df_fundamental)} fundamental records.")

        # 3. Merge Data
        logging.info("Merging trend and fundamental data...")
        print("Merging trend and fundamental data...")
        # Use inner merge to keep only tickers present in both files
        df_merged = pd.merge(df_trend, df_fundamental, on='Ticker', how='inner')
        logging.info(f"Merged data contains {len(df_merged)} tickers found in both files.")
        print(f"Merged data contains {len(df_merged)} tickers found in both files.")

        if df_merged.empty:
            logging.error("No common tickers found between the trend and fundamental files. Cannot proceed with screening.")
            print("Error: No common tickers found between the two files. Cannot proceed.")
            screening_successful = False
        else:
            # 4. Normalize Scores to Percentiles (0-100)
            logging.info("Normalizing scores to percentile ranks...")
            print("Normalizing scores to percentile ranks...")
            df_merged['Trend_Percentile'] = normalize_to_percentile(df_merged['Trend_Score'])
            df_merged['Fundamental_Percentile'] = normalize_to_percentile(df_merged['Fundamental_Score'])

            # Select and reorder columns for clarity before applying methods
            df_processed = df_merged[['Ticker', 'Trend_Score', 'Fundamental_Score', 'Trend_Percentile', 'Fundamental_Percentile']].copy()

            # 5. Apply Screening Methods
            logging.info("Applying Combined Weighted Score method...")
            print("Applying Combined Weighted Score method...")
            df_weighted_results = combined_weighted_score(df_processed.copy(), trend_weight, fundamental_weight)

            logging.info("Building Dual Threshold (rank‑based) list …")
            print("Building Dual Threshold (rank‑based) list …")

            # ---------- new rank parameters ----------
            rank_cfg = config["RANK_FILTER"] if config.has_section("RANK_FILTER") else {}
            top_num_trend  = int(rank_cfg.get("top_num_trend", 50))
            top_num_growth = int(rank_cfg.get("top_num_growth", 50))

            # ---------- top‑N by Trend & Fundamental ----------
            # Prefer 'Normalized_Trend_Score' if present; fallback to Trend_Score
            trend_sort_col = "Normalized_Trend_Score" if "Normalized_Trend_Score" in df_weighted_results.columns else "Trend_Score"
            # --- top‑N by Trend & include ties ---
            trend_sorted = df_weighted_results.sort_values(
                trend_sort_col, ascending=False, na_position="last"
            ).reset_index(drop=True)

            if len(trend_sorted) >= top_num_trend:
                threshold_trend = trend_sorted.loc[top_num_trend - 1, trend_sort_col]
            else:
                threshold_trend = trend_sorted[trend_sort_col].min()

            top_trend_tickers = (
                trend_sorted[trend_sorted[trend_sort_col] >= threshold_trend]["Ticker"]
                .tolist()
            )

            # Prefer 'Overall_Score' if present; fallback to Fundamental_Score
            fund_sort_col = "Overall_Score" if "Overall_Score" in df_weighted_results.columns else "Fundamental_Score"
            # --- top‑N by Fundamental & include ties ---
            fund_sorted = df_weighted_results.sort_values(
                fund_sort_col, ascending=False, na_position="last"
            ).reset_index(drop=True)

            if len(fund_sorted) >= top_num_growth:
                threshold_fund = fund_sorted.loc[top_num_growth - 1, fund_sort_col]
            else:
                threshold_fund = fund_sorted[fund_sort_col].min()

            top_fund_tickers = (
                fund_sorted[fund_sorted[fund_sort_col] >= threshold_fund]["Ticker"]
                .tolist()
            )

            dual_tickers = set(top_trend_tickers).intersection(top_fund_tickers)
            df_threshold_results = (
                df_weighted_results[df_weighted_results["Ticker"].isin(dual_tickers)]
                .copy()
                .sort_values("Fundamental_Score", ascending=False, na_position="last")
                .reset_index(drop=True)
                [["Ticker", "Trend_Score", "Fundamental_Score", "Combined_Score"]]
            )

            # 6. Save Results
            logging.info("Saving screening results...")
            print("Saving screening results...")
            save_results(df_weighted_results, df_threshold_results, output_dir, output_filename)

    except FileNotFoundError as e:
        logging.error(f"File not found during screening: {e}")
        print(f"Error: {e}")
        screening_successful = False
    except ValueError as e:
        logging.error(f"Value error during screening (e.g., missing column): {e}")
        print(f"Error: {e}")
        screening_successful = False
    except configparser.Error as e:
        logging.error(f"Configuration error in '{CONFIG_FILE_RUN}': {e}")
        print(f"Configuration Error: {e}")
        screening_successful = False
    except Exception as e:
        logging.error(f"An unexpected error occurred during final screening: {e}", exc_info=True)
        print(f"An unexpected error occurred during final screening: {e}")
        screening_successful = False

    if screening_successful:
        logging.info(">>> Final Screening Process Completed Successfully <<<")
        print("--- Final Screening Process Completed Successfully ---")
    else:
        logging.error(">>> Final Screening Process Failed <<<")
        print("--- Final Screening Process Failed ---")
    return screening_successful


# CORRECTED: Main pipeline function encapsulating the steps
def main_pipeline(run_final_screening=True):
    """
    Orchestrates the entire S&P 500 processing pipeline.

    Args:
        run_final_screening (bool): Controls if the final screening step runs.
    """

    main_start_time = time.time()
    print("=============================================")
    print("=== Starting S&P 500 Processing Script ===")
    print("=============================================")
    print(f"Run Control: Run Screening = {run_final_screening}")

    overall_success = True

    # Step 1: Compute Trend Score
    # Indicators and scores are always calculated using the latest price data
    print("\nSTEP 1: Computing Trend Score (Indicators, Score)...")
    trend_success = compute_trend_score(
        calculate_indicators=True,  # Always calculate indicators
        calculate_score=True        # Always calculate score
    )

    if not trend_success:
        print("\nERROR: Trend score computation failed. Aborting further steps.")
        overall_success = False
        # sys.exit(1) # Optional: exit immediately

    # Step 2: Compute Growth Score (only if trend step was ok)
    if overall_success:
        print("\nSTEP 2: Computing Growth Score...")
        try:
            # Call the imported function and capture success status
            growth_success = compute_growth_score()
            print("Growth score computation function executed.")
            if not growth_success:
                print("Growth score computation reported failure.")
                overall_success = False
        except Exception as e_growth:
            print(f"\nERROR: Growth score computation failed: {e_growth}")
            traceback.print_exc()
            growth_success = False
            overall_success = False
            # sys.exit(1) # Optional: exit immediately

    # Step 3: Final Screening (only if previous steps succeeded and flag is True)
    if overall_success and run_final_screening:
        print("\nSTEP 3: Performing Final Screening...")
        screen_success = final_screen()
        if not screen_success:
             print("\nERROR: Final screening step failed.")
             overall_success = False
             # sys.exit(1) # Optional: exit immediately
    elif not overall_success:
        print("\nSkipping Final Screening due to errors in previous steps.")
    else: # overall_success is True but run_final_screening is False
        print("\nSkipping Final Screening as requested.")


    main_end_time = time.time()
    total_script_duration = main_end_time - main_start_time
    print("\n=============================================")
    print(f"=== Script Execution Finished ===")
    print(f"Total Duration: {total_script_duration:.2f} seconds")
    print(f"Overall Status: {'Success' if overall_success else 'Failed'}")
    print("=============================================")
    # Return status or exit code if needed
    # return 0 if overall_success else 1

def test_main():
    main_pipeline(
        True
    )
#test_main()

# # --- Main Execution Block ---
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run S&P 500 processing pipeline.")
    parser.add_argument(
        '--skip-screening',
        action='store_true',
        help="Skip the final screening step."
    )
    # Example: Add a flag to skip the entire trend calculation if needed
    # parser.add_argument(
    #     '--skip-trend-calculation',
    #     action='store_true',
    #     help="Skip the entire trend score calculation (data update, indicators, and score)."
    # )
    args = parser.parse_args()

    # Determine run parameters based on arguments
    run_screen = not args.skip_screening
    # run_trend_calculation = not args.skip_trend_calculation # If using the extra flag

    # Call the main pipeline function with determined parameters
    # If you add --skip-trend-calculation, you would add an outer if condition here:
    # if run_trend_calculation:
    #      main_pipeline(...)
    # else:
    #      # Call main_pipeline, but maybe skip trend step entirely or handle differently
    #      print("Skipping entire Trend Calculation Step")
    #      # Need to decide how the rest of the pipeline behaves if trend is skipped

    # Current implementation based on existing flags:
    main_pipeline(
        run_final_screening=run_screen
    )
