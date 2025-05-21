import unittest
import sqlite3
from pathlib import Path
import configparser
from importlib import util

pandas_spec = util.find_spec("pandas")
openpyxl_spec = util.find_spec("openpyxl")

if pandas_spec and openpyxl_spec:
    import pandas as pd
else:
    pd = None


@unittest.skipUnless(pandas_spec and openpyxl_spec, "requires pandas and openpyxl")
class RealFinanceDbExportTest(unittest.TestCase):
    """Export one ticker's financial data from the real DB if available."""

    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = configparser.ConfigParser()
        cfg.read(repo_root / "config.ini")
        db_name = cfg.get("database", "finance_db", fallback="SP500_finance_data.db")
        self.db_path = repo_root / db_name
        self.ticker = "AAPL"
        self.output_path = Path(__file__).with_name(f"{self.ticker}_financials.xlsx")

    def tearDown(self):
        if self.output_path.exists():
            self.output_path.unlink()

    def test_export_ticker_financials(self):
        if not self.db_path.exists():
            self.skipTest(f"Database not found: {self.db_path}")

        with sqlite3.connect(self.db_path) as conn:
            annual = pd.read_sql(
                "SELECT * FROM annual_financials WHERE UPPER(ticker)=UPPER(?)",
                conn,
                params=(self.ticker,),
            )
            quarterly = pd.read_sql(
                "SELECT * FROM quarterly_financials WHERE UPPER(ticker)=UPPER(?)",
                conn,
                params=(self.ticker,),
            )

        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            if not annual.empty:
                annual.to_excel(writer, sheet_name="Annual_Data", index=False)
            if not quarterly.empty:
                quarterly.to_excel(writer, sheet_name="Quarterly_Data", index=False)

        self.assertTrue(self.output_path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

