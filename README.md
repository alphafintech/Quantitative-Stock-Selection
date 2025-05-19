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
