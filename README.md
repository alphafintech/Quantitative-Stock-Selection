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

The pipeline relies on two SQLite databases (price and finance). Before you
skip the download phases, ensure these databases contain data. On the first run,
invoke the main script with the update flags enabled so that both databases are
populated:

```bash
python Run_complete_program.py --update-price-data --update-finance-data
```

After this initial download you can run the script without the update flags and
use `--skip-Gemini` or `--skip-GPT` as needed.

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
