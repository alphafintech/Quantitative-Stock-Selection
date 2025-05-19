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

## Gemini screening results

`Run_complete_program.py` expects a screening results file for the Gemini
pipeline located at `Gemini/results/screened_stocks.xlsx` (path configurable via
`config.ini`). You can generate this file by running the Gemini processing
pipeline:

```bash
python Gemini/run_sp500_processing.py
```

Ensure the file exists before attempting to generate Gemini prompts.
