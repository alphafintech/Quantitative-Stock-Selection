import unittest
import sys, types
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch
import sqlite3

# Stub heavy deps so module can be imported without them installed
sys.modules.setdefault('yfinance', types.ModuleType('yfinance'))
requests_stub = types.ModuleType('requests')
requests_stub.exceptions = types.SimpleNamespace(RequestException=Exception)
sys.modules.setdefault('requests', requests_stub)

# Provide minimal pandas stub
pd_stub = types.ModuleType('pandas')
pd_stub.DataFrame = type('DataFrame', (), {})
pd_stub.read_html = lambda *a, **k: []
sys.modules['pandas'] = pd_stub
sys.modules.pop('GPT.Compute_Trend_score_SP500_GPT', None)
import GPT.Compute_Trend_score_SP500_GPT as mod

class GetSp500TickersFallbackTest(unittest.TestCase):
    def test_load_from_fallback_file(self):
        def failing_read_html(*a, **k):
            raise URLError('fail')
        with patch.object(mod.pd, 'read_html', side_effect=failing_read_html):
            tickers = mod.get_sp500_tickers()
        self.assertIn('AAPL', tickers)
        self.assertIn('SPY', tickers)

    def test_empty_when_no_file(self):
        def failing_read_html(*a, **k):
            raise URLError('fail')
        with patch.object(mod.pd, 'read_html', side_effect=failing_read_html):
            with patch('pathlib.Path.exists', return_value=False):
                tickers = mod.get_sp500_tickers()
        self.assertEqual(tickers, [])

class UpdateDbGracefulTest(unittest.TestCase):
    def test_update_db_no_tickers(self):
        with patch.object(mod, 'get_sp500_tickers', return_value=[]):
            mod.yf.download = lambda *a, **k: None
            def fake_init_db(db_file):
                conn = sqlite3.connect(':memory:')
                return conn, conn.cursor()
            mod.init_db = fake_init_db
            try:
                mod.Update_DB(':memory:')
            except Exception as e:
                self.fail(f'Update_DB raised {e}')

if __name__ == '__main__':
    unittest.main()
