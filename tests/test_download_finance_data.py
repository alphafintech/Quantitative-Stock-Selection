import unittest
import sys, types
from pathlib import Path

# Stub SQLAlchemy create_engine to avoid dependency
sys.modules.setdefault('sqlalchemy', types.ModuleType('sqlalchemy'))
import sqlalchemy
sqlalchemy.create_engine = lambda *a, **k: None

# Prepare dummy finance module
fin_mod = types.ModuleType('GPT.compute_high_growth_score_SP500_GPT')
call_flags = {
    'init': False,
    'meta': False,
    'download': False,
}

def init():
    call_flags['init'] = True
    fin_mod.CFG = {'database': {'db_name': ''}}

fin_mod.initialize = init
fin_mod.CFG = None
fin_mod.DB_PATH = None
fin_mod.engine = None
fin_mod.load_sp500_meta = lambda: call_flags.__setitem__('meta', True)
fin_mod.download_all = lambda: call_flags.__setitem__('download', True)

sys.modules['GPT.compute_high_growth_score_SP500_GPT'] = fin_mod

import importlib
import download_data
importlib.reload(download_data)

class FinanceDownloadInitTest(unittest.TestCase):
    def test_initialize_called(self):
        download_data.download_finance_data(Path('dummy.db'))
        self.assertTrue(call_flags['init'])
        self.assertTrue(call_flags['meta'])
        self.assertTrue(call_flags['download'])

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
