# INI Configuration Reference

This document explains all parameters found in the various `*.ini` files.

## Root `config.ini`

### `[data_download]`
- `start_price_date` – first day of price history loaded by `yahoo_downloader.py`. Defaults to `1900-01-01` if empty.
- `end_price_date` – last day of price history. Defaults to today when blank.

### `[prompt_common]`
Common placeholders used when building prompts in `Run_complete_program.py`:
- `start_date` / `end_date` – date range for news. When missing, `end_date` becomes today and `start_date` is derived from `previous_days`.
- `previous_days` – number of days prior to `end_date` used when `start_date` is absent.
- `investment_horizon` – holding period in days for prompt generation.
- `mini_num_stocks` / `maxi_num_stocks` – minimum/maximum count of tickers in generated prompts.
- `base_prompt` – path to the base text template.
- `trend_scoring_algorithm_overview_GPT`, `fundamental_scoring_algorithm_overview_GPT` – explanation snippets inserted into the prompt.
- `trend_scoring_algorithm_overview_Gemini`, `fundamental_scoring_algorithm_overview_Gemini` – same as above for Gemini pipeline.

### `[ScriptSettings]`
Controls the helper scripts launched by `scripts/run_*.sh`:
- `Gemini_python_script_name` – file executed for the Gemini pipeline.
- `GPT_python_script_name` – file executed for the GPT pipeline.
- `Gemini_work_dir` / `GPT_work_dir` – working directories used when running the two pipelines.

### `[Arguments_run_sp500_processing]`
Flags forwarded to `Gemini/run_sp500_processing.py`:
- `Gemini_skip_screening` – when `true`, skip the final screening step.

### `[Arguments_run_complete_process]`
Flags forwarded to `GPT/run_complete_process.py`:
- `Gemini_arg_update_trend_db` – `true` updates the trend database.
- `Gemini_arg_no_recalc` – skip recomputation of scores.
- `Gemini_arg_no_select` – skip composite selection.
- `Gemini_arg_cfg_path` – path to the GPT run configuration file.

### `[FILES]`
File locations used across the project:
- `screened_stocks_file_Gemini` – Gemini screening result Excel.
- `screened_stocks_file_GPT` – GPT composite selection result file.
- `output_dir` – directory where prompts and HTML reports are written.
- `current_holdings_file_Gemini` / `current_holdings_file_GPT` – holdings list used for prompt placeholders.
- `AI_selection_Prompt_Gemini` / `AI_selection_Prompt_GPT` – output prompt filenames.

### `[COLUMNS]`
Column names for reading screening results:
- `Gemini_ticker_col`, `Gemini_trend_score_col`, `Gemini_fundamental_col`, `Gemini_final_score_col` – column names for Gemini Excel sheets.
- `Gemini_sheet_Combined_Weighted_Score`, `Gemini_sheet_Dual_Threshold_Filter` – sheet names used by `Run_complete_program.py`.
- `GPT_ticker_col`, `GPT_trend_score_col`, `GPT_fundamental_col`, `GPT_final_score_col` – column names for the GPT Excel file.

### `[database]`
Paths to various SQLite databases. Most scripts read these values via `configparser`:
- `price_db` – raw price history database used by both pipelines.
- `finance_db` – Gemini finance database.
- `gpt_price_db` – copy of price history for the GPT pipeline.
- `GPT_finance_db` – finance database for the GPT pipeline.
- `raw_stage_db` – staging DB for raw fundamental statements.

### `[completeness]`
Thresholds applied when filtering raw fundamental data in `yahoo_downloader.py`:
- `latest_qtr_max_gap` – allowed fraction of missing values in the most recent quarter.
- `after_filter_max_gap` – allowed fraction after basic filtering.
- `history_quarters` – minimum number of quarters required.

## `GPT/config_run.ini`

### `[run_control]`
`runstage` selects which part of the GPT pipeline runs:
0 – rebuild database and run every step; 1 – update price data only; 2 – compute indicators only; 3 – compute trend score only.

### `[selection]`
Controls the composite selection produced by `GPT/run_complete_process.py`:
- `output_name` – output Excel file.
- `trend_file` / `fund_file` – pre‑calculated trend and fundamental score files.
- `top_num_trend` / `top_num_growth` – number of rows taken from each ranking list.
- `trend_thresh`, `fund_thresh`, `growth_thresh` – score thresholds used when filtering.
- `w_core` / `w_growth` – weights used to combine trend and growth scores.

## `GPT/config_finance.ini`

Used by `compute_high_growth_score_SP500_GPT.py`.

### `[data]`
- `start_date` / `end_date` – date range of financial data to download.
- `update_mode` – `full` rebuilds the DB, `incremental` appends new data only.

### `[export]`
`excel_file_name` – destination of the exported fundamental score table.

### `[database]`
- `engine` – database backend (SQLite by default).
- `db_name` – path to the GPT finance database file.

### `[weights]`
Weights for each metric pillar when computing the overall score. Should sum to 1.

### `[metric_parameters]`
- `winsor_min` / `winsor_max` – bounds used when winsorising metrics.
- `percentile_scope` – percentile benchmark scope (`industry` or `all`).
- `fy_years` – how many fiscal years to consider for growth calculations.
- `fy_calc` – method for FY growth (`cagr` or `chain`).
- `min_industry_size` – minimum industry size to use industry percentiles.

### `[rating_thresholds]`
Cutoffs for star ratings applied in reports: `five_star`, `four_star`, `three_star`, `two_star`.

### `[combo_weights]`
Weights for combining FY and trailing‑twelve‑month growth views.

## `GPT/config_trend.ini`

### `[data_download]`
`start_date` and `end_date` define the price window used to compute indicators.

### `[trend_score]`
`excel_file_name` – export path for trend score results.

### `[indicator]`
Parameters passed to indicator calculations in `Compute_Trend_score_SP500_GPT.py`:
- `stoch_k_period`, `stoch_d_period` – stochastic oscillator periods.
- `stoch_overbought`, `stoch_oversold` – overbought/oversold thresholds.
- `support_resistance_lookback` – window for support/resistance analysis.
- `adx_period` – period for Average Directional Index.
- `roc_period` – rate of change period.
- `weeks_lookback_52` – lookback weeks for 52‑week high/low.

### `[run_control]`
Same semantics as the GPT `run_control` section.

## `Gemini/config_finance.ini`
Used by `Gemini/Compute_growth_score_sp500.py`.

### `[General]`
- `sp500_list_url` – URL for obtaining the list of tickers.
- `output_excel_file` – path for the fundamental score output.
- `log_level` – logging verbosity.
- `log_to_file` – whether to log to a file.

### `[Data]`
- `db_name` – SQLite database name for fundamental data.
- `years_of_annual_data` – required history length for CAGR.
- `incremental_download` – only fetch missing statements when `True`.
- `download_delay` – delay between ticker downloads.

### `[Screening]`
Financial health filter parameters:
- `enable_screening` – enable the screening step.
- `max_debt_to_equity` – upper bound for debt‑to‑equity ratio.
- `min_interest_coverage` – lower bound for interest coverage ratio.

### `[Scoring_Weights]`
Weights used when fusing CAGR and acceleration as well as final dimension weights. They must each sum to 1.

### `[Calculation_Params]`
Advanced calculation options:
- `min_eps_for_cagr` – minimum EPS for CAGR calculation.
- `eps_qoq_denominator_handling` – how to handle zero/negative EPS in QoQ growth (`zero` or `clip_large`).
- `eps_qoq_zero_value` – value used when denominator is zero.
- `default_cagr` – value assigned when CAGR calculation fails.
- `default_slope` – value used when regression slope fails.

### `[Methodology]`
`ranking_method` selects whether percentiles are computed overall or per industry.

## `Gemini/config_trend.ini`

### `[data_download]`
Start and end dates for price history used by the Gemini trend pipeline.

### `[database]`
- `db_file` – source price database.
- `indicator_db_file` – database where indicators are stored.
- `main_table` / `latest_analysis_table` – table names for history and latest analysis results.

### `[logging]`
`log_file` and `log_level` configure logging output for the trend pipeline.

### `[calculation_params]`
Indicator calculation settings used by `compute_trend_score_sp500.py`:
- `ma_windows`, `volume_ma_windows` – moving average windows.
- `days_lookback_52` – trading days for 52‑week high/low.
- `macd_fast`, `macd_slow`, `macd_signal` – MACD parameters.
- `rsi_length`, `rsi_overbought`, `rsi_oversold` – RSI settings.
- `bbands_length`, `bbands_std` – Bollinger Bands settings.
- `stoch_k_period`, `stoch_d_period`, `stoch_smooth_k` – stochastic oscillator parameters.
- `stoch_overbought`, `stoch_oversold` – stochastic thresholds.
- `adx_period` – ADX calculation period.
- `roc_period` – ROC calculation period.
- `support_resistance_lookback` – lookback window for support/resistance.
- `obv_sma_period` – SMA period used with OBV.

### `[Calculate_trend]`
Weights for each component of the Gemini trend score and the output file name.

### `[performance]`
- `update_batch_size` – number of rows per database batch when storing data.
- `download_delay` – delay between ticker downloads.

## `Gemini/config_run.ini`
Used by `Gemini/run_sp500_processing.py`.

### `[FILES]`
Input/output locations for screening:
- `trend_file` – normalized trend score file.
- `fundamental_file` – fundamental score file.
- `output_dir` – directory for the final Excel.
- `output_filename` – name of the output Excel file.

### `[COLUMNS]`
Column names in the input Excel files:
- `trend_ticker_col` / `trend_score_col` – columns for tickers and trend scores.
- `fundamental_ticker_col` / `fundamental_score_col` – columns for fundamental scores.

### `[WEIGHTED_SCORE]`
`trend_weight` and `fundamental_weight` – weights used when combining percentile ranks. They are normalized to sum to 1.

### `[THRESHOLD_FILTER]`
Minimum percentile thresholds applied by `dual_threshold_filter()`.

### `[RANK_FILTER]`
Number of rows kept from each ranking table before merging.

