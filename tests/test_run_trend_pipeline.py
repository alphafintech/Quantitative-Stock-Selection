import os
import sys
import types
import unittest
import tempfile
from pathlib import Path
import importlib
import sqlite3
import configparser

class TrendPipelineFailureTest(unittest.TestCase):
    def setUp(self):
        # Temporary directory for config files
        self.tmpdir = Path(tempfile.mkdtemp())
        self.repo_root = Path(__file__).resolve().parents[1]
        self.created_files = []

        # Create stub module for heavy dependency
        stub = types.ModuleType("Gemini.compute_trend_score_sp500")

        def load_configuration(cfg_file):
            parser = configparser.ConfigParser()
            if os.path.exists(cfg_file):
                parser.read(cfg_file)
            conf = {}
            if parser.has_section("database"):
                conf["database"] = dict(parser.items("database"))
            return conf

        def create_connection(path):
            return sqlite3.connect(":memory:")

        stub.load_configuration = load_configuration
        stub.create_connection = create_connection
        stub.create_tables = lambda conn: None
        stub.update_stock_data = lambda conn: True
        stub.calculate_all_indicators = lambda conn: True
        stub.calculate_and_save_trend_scores = lambda conn: True
        stub.CONFIG = {}

        # Inject stub before importing target module
        sys.modules["Gemini.compute_trend_score_sp500"] = stub
        # Stub pandas and numpy which may be missing
        sys.modules.setdefault("pandas", types.ModuleType("pandas"))
        sys.modules.setdefault("numpy", types.ModuleType("numpy"))
        # Stub growth score module to avoid heavy deps
        growth_stub = types.ModuleType("Gemini.Compute_growth_score_sp500")
        growth_stub.compute_growth_score = lambda *a, **k: True
        sys.modules["Gemini.Compute_growth_score_sp500"] = growth_stub
        self.module = importlib.import_module("Gemini.run_sp500_processing")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)
        for f in self.created_files:
            if f.exists():
                f.unlink()
        sys.modules.pop("Gemini.compute_trend_score_sp500", None)
        # Remove imported module so next tests get a clean state
        sys.modules.pop("Gemini.run_sp500_processing", None)
        sys.modules.pop("pandas", None)
        sys.modules.pop("numpy", None)
        sys.modules.pop("Gemini.Compute_growth_score_sp500", None)

    def _write_cfg(self, text: str) -> Path:
        cfg = self.tmpdir / "cfg.ini"
        cfg.write_text(text, encoding="utf-8")
        return cfg

    def test_missing_db_file_propagates_false(self):
        cfg = self._write_cfg(
            f"""[database]\ndb_file = {self.tmpdir / 'missing.db'}\nindicator_db_file = {self.tmpdir / 'indicator.db'}\n"""
        )
        res = self.module.run_trend_score_pipeline(
            config_file=str(cfg),
            do_calculate_indicators=False,
            do_calculate_trend_score=False,
        )
        self.assertFalse(res)

    def test_missing_db_key_propagates_false(self):
        cfg = self._write_cfg(
            f"""[database]\nindicator_db_file = {self.tmpdir / 'indicator.db'}\n"""
        )
        res = self.module.run_trend_score_pipeline(
            config_file=str(cfg),
            do_calculate_indicators=False,
            do_calculate_trend_score=False,
        )
        self.assertFalse(res)

    def test_relative_db_path_resolved(self):
        price_name = "rel_price.db"
        price_path = self.repo_root / price_name
        sqlite3.connect(price_path).close()
        self.created_files.append(price_path)

        ind_path = self.tmpdir / "indicator.db"
        cfg = self._write_cfg(
            f"""[database]\ndb_file = {price_name}\nindicator_db_file = {ind_path}\n"""
        )
        res = self.module.run_trend_score_pipeline(
            config_file=str(cfg),
            do_calculate_indicators=False,
            do_calculate_trend_score=False,
        )
        self.assertTrue(res)
        self.assertTrue(ind_path.exists())

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
