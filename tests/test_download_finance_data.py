import unittest
import sys, types
from pathlib import Path

# Stub trend module used by download_data
trend_mod = types.ModuleType('GPT.Compute_Trend_score_SP500_GPT')
trend_mod.Update_DB = lambda *a, **k: None
sys.modules['GPT.Compute_Trend_score_SP500_GPT'] = trend_mod

# Prepare dummy Gemini finance module
gem_mod = types.ModuleType('Gemini.Compute_growth_score_sp500')
call_flags = {
    'load_cfg': False,
    'conn': False,
    'tables': False,
    'tickers': False,
    'download': [],
}

def load_config():
    call_flags['load_cfg'] = True
    return {'General': {'sp500_list_url': 'http://example.com'}, 'Data': {'db_name': 'orig.db'}}

def create_db_connection(db_file):
    call_flags['conn'] = db_file
    class Dummy:
        def close(self):
            pass
    return Dummy()

def create_tables(conn):
    call_flags['tables'] = True

def get_sp500_tickers_and_industries(url):
    call_flags['tickers'] = True
    return {'AAA': 'Tech'}

def download_data_for_ticker(ticker, config, conn):
    call_flags['download'].append(ticker)

gem_mod.load_config = load_config
gem_mod.create_db_connection = create_db_connection
gem_mod.create_tables = create_tables
gem_mod.get_sp500_tickers_and_industries = get_sp500_tickers_and_industries
gem_mod.download_data_for_ticker = download_data_for_ticker

sys.modules['Gemini.Compute_growth_score_sp500'] = gem_mod

import importlib
import download_data
importlib.reload(download_data)

class FinanceDownloadInitTest(unittest.TestCase):
    def test_download_called(self):
        download_data.download_finance_data(Path('dummy.db'))
        self.assertTrue(call_flags['load_cfg'])
        self.assertEqual(call_flags['conn'], 'dummy.db')
        self.assertTrue(call_flags['tables'])
        self.assertTrue(call_flags['tickers'])
        self.assertEqual(call_flags['download'], ['AAA'])

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
