import configparser
import sqlite3
from pathlib import Path
import pandas as pd
import traceback
from typing import Dict

def export_ticker_financials_to_excel(ticker: str,
                                      cfg_path: str | Path | None = None,
                                      raw_table: str = "raw_financials",
                                      metrics_table: str = "derived_metrics") -> Path:
    """
    Export all financial records for a ticker to Excel and print three tables
    on the console.

    Parameters
    ----------
    ticker : str
        Target ticker symbol (case insensitive, converted to upper case)
    cfg_path : str | Path | None
        Path to ``config_finance.ini``. ``None`` defaults to ``/GPT/config_finance.ini``.
    raw_table / metrics_table : str
        Table names to query; change if you customized them.

    Returns
    -------
    Path
        Path to the generated Excel file.
    """
    # Resolve cfg_path to an absolute path (defaults to /GPT/config_finance.ini)
    if cfg_path is None:
        cfg_path = Path(__file__).parent / "config_finance.ini"
    else:
        cfg_path = Path(cfg_path)
        if not cfg_path.is_absolute():
            cfg_path = Path(__file__).parent / cfg_path
    # Read the DB path
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.read(str(cfg_path), encoding="utf-8")
    db_name = cfg["database"].get("db_name", "SP500_finance_data.db")
    db_path = cfg_path.parent / db_name

    if not db_path.exists():
        raise FileNotFoundError(f"Database file does not exist: {db_path}")

    # Connect to DB and dynamically read all tables containing a ticker column
    sheet_dfs: Dict[str, pd.DataFrame] = {}

    with sqlite3.connect(db_path) as conn:
        # Find all user tables
        table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()

        for tbl in table_names:
            # Only process tables containing a ticker column
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
            if not any(c.lower() == "ticker" for c in cols):
                continue

            try:
                df = pd.read_sql(f'SELECT * FROM "{tbl}" WHERE UPPER(ticker)=UPPER(?)', conn, params=(ticker,))
                if not df.empty:
                    sheet_dfs[tbl] = df.sort_values(df.columns[0], ignore_index=True)
            except Exception:
                traceback.print_exc()
                continue

    # Print to console for a quick look
    if not sheet_dfs:
        raise ValueError(f"No data for {ticker} found in the database.")

    for tbl, df in sheet_dfs.items():
        print(f"\n========== {tbl.upper()} =========")
        print(df.head())

    # Convert common date columns to strings
    for df in sheet_dfs.values():
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    # Output to Excel, one sheet per table
    out_name = f"{ticker.upper()}_financials.xlsx"
    out_path = Path(__file__).with_name(out_name)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for tbl, df in sheet_dfs.items():
            writer.book.create_sheet(tbl[:31])   # sheet name max 31 chars
            df.to_excel(writer, sheet_name=tbl[:31], index=False)

    print(f"[INFO] Exported to {out_path.resolve()}")
    return out_path

# ========= Example usage =========
if __name__ == "__main__":
    export_ticker_financials_to_excel("NVDA")

