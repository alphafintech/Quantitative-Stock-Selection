# Quantitative-Stock-Selection
Quantitative Stock Selection

## Configuration

Database locations are defined in `config.ini` under the `[database]` section:

```
[database]
price_db = SP500_price_data.db
finance_db = SP500_finance_data.db
```
Both paths are relative to the project root unless absolute paths are provided.
Scripts that download or process data will automatically use these settings.

All configuration files and helper scripts expect the finance database to be named `SP500_finance_data.db`.

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
