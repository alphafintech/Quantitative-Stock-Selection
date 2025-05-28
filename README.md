# Quantitative-Stock-Selection
Quantitative Stock Selection

## Configuration

Database locations are defined in `config.ini` under the `[database]` section:

```
[database]
price_db = SP500_price_data.db
finance_db = SP500_finance_data.db
GPT_finance_db = GPT/SP500_finance_data_GPT.db
```
Both paths are relative to the project root unless absolute paths are provided.
Scripts that download or process data will automatically use these settings.

All configuration files and helper scripts expect the finance database to be named `SP500_finance_data.db`.

### Initializing the databases

The pipeline relies on two SQLite databases (price and finance). On the first
run simply execute the script so that both databases are populated with price
and fundamental data:

```bash
python Run_complete_program.py
```

After this initial download you may skip the update steps by supplying the
`--skip-update-price-data` and `--skip-update-finance-data` options. Use
`--skip-Gemini-pipeline` or `--skip-GPT-pipeline` to disable either pipeline
when not needed.

### Working Directory

Pipeline scripts such as `Run_complete_program.py` expect to be executed
from the repository root so that `config.ini` and the database files can
be located correctly. Running them from inside subdirectories (for
example `GPT/`) will cause new, empty databases to be created in that
subfolder, leading to errors like `no such table: stock_data`. If you
must run a script from elsewhere, supply the path to `config.ini` via the
appropriate command‑line option or change to the project root first.

## Gemini screening results

`Run_complete_program.py` expects a screening results file for the Gemini
pipeline located at `Gemini/results/screened_stocks.xlsx` (path configurable via
`config.ini`). You can generate this file by running the Gemini processing
pipeline:

```bash
python Gemini/run_sp500_processing.py
```

Ensure the file exists before attempting to generate Gemini prompts.

Trend data indicators are always derived from existing price records. The
previous `Gemini_skip_trend_data_update` configuration option has been removed.

### Finance database migration

Historical versions stored raw statements in the table `raw_financials`. If your
database only contains this table, run the migration script once to populate the
new `annual_financials` and `quarterly_financials` summaries:

```bash
python -m Gemini.finance_db_migrate --config config.ini
```

The script does nothing when the target tables already contain data.

## Running the complete workflow

Use `Run_complete_program.py` to update data, run both the Gemini and GPT pipelines and generate the final prompts. The script writes the resulting HTML reports and prompt files into the directory defined by `output_dir` (default `result_output`):

```bash
python Run_complete_program.py [options]
```

Supported options:

* `--skip-update-price-data` – reuse existing price history.
* `--skip-update-finance-data` – reuse existing fundamental data.
* `--skip-Gemini-pipeline` – do not run the Gemini steps.
* `--skip-GPT-pipeline` – do not run the GPT steps.

Convenience scripts in `scripts/` mirror these options for Linux (`.sh`) and Windows (`.bat`).

### Prompt and HTML outputs

After each run the utility exports the screening results to HTML with `export_gemini_screened_to_html()` and `export_gpt_screened_to_html()` and generates two prompt files for Gemini and GPT. Check the `Output/` directory for the generated `AI_selection_Prompt_Gemini.txt` and `AI_selection_Prompt_GPT.txt`.

### Portfolio performance helper

`portfolio_yield.py` reads `Portfolio/portfolio_performance.xlsx`, calculates cumulative returns for Gemini, ChatGPT and the S&P 500 and saves an updated Excel file and comparison chart in `result_output/`.

### Running tests

Install the packages from `requirements.txt` and run:

```bash
pytest -q
```

