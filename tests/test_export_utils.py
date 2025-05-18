import unittest
import sqlite3
from pathlib import Path
import tempfile
import importlib.util

# Skip entire module if dependencies are missing
pandas_spec = importlib.util.find_spec("pandas")
openpyxl_spec = importlib.util.find_spec("openpyxl")
if pandas_spec and openpyxl_spec:
    import pandas as pd
    from GPT.finance_export import export_ticker_financials_to_excel
else:
    pd = None

@unittest.skipUnless(pandas_spec and openpyxl_spec, "requires pandas and openpyxl")
class ExportFinancialsTest(unittest.TestCase):
    def test_export_to_excel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"
            cfg_path = tmpdir / "config.ini"

            # create database
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE annual_financials (ticker TEXT, value INTEGER, report_date TEXT)")
            conn.execute("CREATE TABLE quarterly_financials (ticker TEXT, value INTEGER, report_date TEXT)")
            conn.execute("INSERT INTO annual_financials VALUES ('AAA', 1, '2023-12-31')")
            conn.execute("INSERT INTO quarterly_financials VALUES ('AAA', 2, '2023-09-30')")
            conn.commit()
            conn.close()

            cfg_path.write_text("[database]\ndb_name = test.db\n", encoding="utf-8")

            out_path = export_ticker_financials_to_excel("AAA", cfg_path=str(cfg_path))
            self.assertTrue(out_path.exists())

            xls = pd.ExcelFile(out_path)
            self.assertEqual(set(xls.sheet_names), {"annual_financials", "quarterly_financials"})

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
