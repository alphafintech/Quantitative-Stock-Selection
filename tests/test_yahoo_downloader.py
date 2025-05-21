import sqlite3
import unittest
import importlib.util
import sys, types
from pathlib import Path

# Ensure the repository root is on the module search path so that
# 'yahoo_downloader.py' can be imported when tests are executed from
# the 'tests' directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pandas_spec = importlib.util.find_spec("pandas")
numpy_spec = importlib.util.find_spec("numpy")
yf_spec = importlib.util.find_spec("yfinance")

if pandas_spec and numpy_spec and yf_spec:
    import pandas as pd
    from yahoo_downloader import (
        _ensure_price_schema,
        _insert_price_df,
        _ensure_fin_schema,
        _save_batches,
    )
else:
    pd = None
    # Create dummy modules so importing yahoo_downloader doesn't fail when skipped
    for name in [n for n, spec in [("numpy", numpy_spec), ("yfinance", yf_spec)] if spec is None]:
        sys.modules.setdefault(name, types.ModuleType(name))

@unittest.skipUnless(pandas_spec and numpy_spec and yf_spec, "requires pandas, numpy, yfinance")
class InsertPriceDfMissingAdjCloseTest(unittest.TestCase):
    def test_missing_adj_close_uses_close(self):
        # Create dummy dataframe without 'Adj Close'
        df = pd.DataFrame({
            'Date': ['2023-01-01'],
            'Open': [1.0],
            'High': [2.0],
            'Low': [0.5],
            'Close': [1.5],
            'Volume': [100]
        }).set_index('Date')

        conn = sqlite3.connect(':memory:')
        _ensure_price_schema(conn)
        cur = conn.cursor()

        _insert_price_df(cur, df, 'AAA')
        conn.commit()

        cur.execute('SELECT close, adj_close FROM stock_data WHERE ticker=?', ('AAA',))
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1.5)
        self.assertEqual(row[1], 1.5)

        conn.close()


@unittest.skipUnless(pandas_spec and numpy_spec and yf_spec, "requires pandas, numpy, yfinance")
class InsertPriceDfTimestampTest(unittest.TestCase):
    def test_timestamp_index_converted_to_str(self):
        df = pd.DataFrame(
            {
                "Open": [1.0],
                "High": [2.0],
                "Low": [0.5],
                "Close": [1.5],
                "Adj Close": [1.6],
                "Volume": [100],
            },
            index=pd.to_datetime(["2023-01-02"]),
        )

        conn = sqlite3.connect(":memory:")
        _ensure_price_schema(conn)
        cur = conn.cursor()

        _insert_price_df(cur, df, "BBB")
        conn.commit()

        cur.execute("SELECT date FROM stock_data WHERE ticker=?", ("BBB",))
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "2023-01-02")

        conn.close()


@unittest.skipUnless(pandas_spec and numpy_spec and yf_spec, "requires pandas, numpy, yfinance")
class SaveBatchesIndexAlignTest(unittest.TestCase):
    def test_misaligned_quarterly_frames(self):
        inc = pd.DataFrame(
            {"Total Revenue": [1, 2]},
            index=pd.to_datetime(["2023-03-31", "2022-12-31"]),
        )
        bal = pd.DataFrame(
            {"Total Assets": [3]},
            index=pd.to_datetime(["2023-03-31"]),
        )
        cf = pd.DataFrame(
            {"Operating Cash Flow": [4, 5, 6]},
            index=pd.to_datetime(["2023-06-30", "2023-03-31", "2022-12-31"]),
        )

        conn = sqlite3.connect(":memory:")
        _ensure_fin_schema(conn)

        try:
            _save_batches(conn, "CCC", [("Q", inc, bal, cf)])
        except Exception as e:
            self.fail(f"_save_batches raised {e!r}")

        conn.close()

if __name__ == '__main__':
    unittest.main()
