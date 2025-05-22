import unittest
import sqlite3
import configparser
from pathlib import Path
import importlib.util
import sys

# Ensure the repository root is on the module search path so that
# modules can be imported when tests are executed from the 'tests'
# directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pandas_spec = importlib.util.find_spec("pandas")
openpyxl_spec = importlib.util.find_spec("openpyxl")

if pandas_spec and openpyxl_spec:
    import pandas as pd
    from Gemini.finance_db_migrate import _load_staged_json_to_df
else:
    pd = None


@unittest.skipUnless(pandas_spec and openpyxl_spec, "requires pandas and openpyxl")
class RawStageDbExportTest(unittest.TestCase):
    """Export raw staging data for one ticker if DB is available."""

    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = configparser.ConfigParser()
        cfg.read(repo_root / "config.ini")
        db_name = cfg.get("database", "raw_stage_db", fallback="SP500_raw_finance.db")
        self.db_path = repo_root / db_name
        self.ticker = "AAPL"
        self.output_path = Path(__file__).with_name(f"{self.ticker}_raw_stage.xlsx")

    def tearDown(self):
        # keep output file for inspection
        pass

    def test_export_raw_ticker_data(self):
        if not self.db_path.exists():
            self.skipTest(f"Database not found: {self.db_path}")

        with sqlite3.connect(self.db_path) as conn:
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'stg_%'"
            )]
            cur = conn.cursor()
            with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
                for tbl in tables:
                    df = _load_staged_json_to_df(cur, self.ticker, tbl)
                    if not df.empty:
                        df.to_excel(writer, sheet_name=tbl[:31])

        self.assertTrue(self.output_path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
