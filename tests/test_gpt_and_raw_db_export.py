import unittest
import sqlite3
from pathlib import Path
import configparser
from importlib import util
import sys

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
        raw_name = cfg.get("database", "raw_stage_db", fallback="SP500_raw_finance.db")
        self.gpt_db_path = repo_root / gpt_name
        self.raw_db_path = repo_root / raw_name
        self.ticker = "AAPL"
        self.output_path = Path(__file__).with_name(f"{self.ticker}_gpt_and_raw.xlsx")

    def tearDown(self):
        # Keep output file for manual inspection
        pass

    def test_export_from_both_dbs(self):
        if not self.gpt_db_path.exists() or not self.raw_db_path.exists():
            self.skipTest(f"Database not found: {self.gpt_db_path} or {self.raw_db_path}")

        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            # GPT finance DB
            with sqlite3.connect(self.gpt_db_path) as conn:
                gpt_df = pd.read_sql(
                    "SELECT * FROM raw_financials WHERE UPPER(ticker)=UPPER(?)",
                    conn,
                    params=(self.ticker,),
                )
                if not gpt_df.empty:
                    gpt_df.to_excel(writer, sheet_name="GPT_raw_financials", index=False)

            # Raw staging DB
            with sqlite3.connect(self.raw_db_path) as conn:
                tables = [row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'stg_%'"
                )]
                cur = conn.cursor()
                for tbl in tables:
                    df = _load_staged_json_to_df(cur, self.ticker, tbl)
                    if not df.empty:
                        df.to_excel(writer, sheet_name=tbl[:31])

        self.assertTrue(self.output_path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
