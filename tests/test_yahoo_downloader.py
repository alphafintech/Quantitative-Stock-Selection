import sqlite3
import unittest
import importlib.util
import sys, types

pandas_spec = importlib.util.find_spec("pandas")
numpy_spec = importlib.util.find_spec("numpy")
yf_spec = importlib.util.find_spec("yfinance")

if pandas_spec and numpy_spec and yf_spec:
    import pandas as pd
    from yahoo_downloader import _ensure_price_schema, _insert_price_df
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

if __name__ == '__main__':
    unittest.main()
