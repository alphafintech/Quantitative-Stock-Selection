import unittest
import sqlite3
from pathlib import Path
import configparser
from importlib import util
import sys

export_ticker = "BAC"  # Example ticker for export
# Ensure repo root is on module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pandas_spec = util.find_spec("pandas")
openpyxl_spec = util.find_spec("openpyxl")

if pandas_spec and openpyxl_spec:
    import pandas as pd
    from Gemini.finance_db_migrate import _load_staged_json_to_df
else:
    pd = None


@unittest.skipUnless(pandas_spec and openpyxl_spec, "requires pandas and openpyxl")
class GptAndRawDbExportTest(unittest.TestCase):
    """Export a ticker's data from GPT and raw finance databases."""

    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = configparser.ConfigParser()
        cfg.read(repo_root / "config.ini")
        gpt_name = cfg.get("database", "GPT_finance_db", fallback="GPT/SP500_finance_data_GPT.db")
        raw_name = cfg.get("database", "raw_stage_db", fallback="data/SP500_raw_stage.db")
        self.gpt_db_path = repo_root / gpt_name
        self.raw_db_path = repo_root / raw_name
        self.ticker = export_ticker

    def test_export_from_both_dbs(self):
        if not self.gpt_db_path.exists() or not self.raw_db_path.exists():
            self.skipTest(f"Database not found: {self.gpt_db_path} or {self.raw_db_path}")

        # --------------- GPT finance DB export -----------------
        gpt_xlsx = Path(__file__).with_name(f"{self.ticker}_GPT_finance.xlsx")
        with pd.ExcelWriter(gpt_xlsx, engine="openpyxl") as writer, \
             sqlite3.connect(self.gpt_db_path) as conn:
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )]
            for tbl in tables:
                # Fetch only rows for selected ticker when table has a 'ticker' column
                cols = [c[1] for c in conn.execute(f"PRAGMA table_info('{tbl}')")]
                if "ticker" in cols:
                    df = pd.read_sql(
                        f"SELECT * FROM {tbl} WHERE UPPER(ticker)=UPPER(?)",
                        conn,
                        params=(self.ticker,),
                    )
                else:
                    df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
                if not df.empty:
                    df.to_excel(writer, sheet_name=tbl[:31], index=False)

        # --------------- raw staging DB export -----------------
        raw_xlsx = Path(__file__).with_name(f"{self.ticker}_raw_stage.xlsx")
        with pd.ExcelWriter(raw_xlsx, engine="openpyxl") as writer, \
             sqlite3.connect(self.raw_db_path) as conn:
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'stg_%'"
            )]
            cur = conn.cursor()
            for tbl in tables:
                df = _load_staged_json_to_df(cur, self.ticker, tbl)
                if not df.empty:
                    df.to_excel(writer, sheet_name=tbl[:31], index=False)

        # Assert files created
        self.assertTrue(gpt_xlsx.exists() and raw_xlsx.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
