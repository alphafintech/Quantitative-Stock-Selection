import unittest
from pathlib import Path
import sys, types

# Stub heavy dependencies so we can import run_complete_process without
# requiring optional packages like pandas.
# Some environments may lack pandas; stub out modules that require heavy deps.
for mod_name in ["GPT.Compute_Trend_score_SP500_GPT", "GPT.Compute_Trend_socre_SP500_GPT"]:
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
    sys.modules[mod_name].run_process_control = lambda *a, **k: None

sys.modules.setdefault(
    "GPT.compute_high_growth_score_SP500_GPT",
    types.ModuleType("GPT.compute_high_growth_score_SP500_GPT"),
)
for name in ["download_all", "compute_metrics", "calc_scores", "export_excel"]:
    setattr(sys.modules["GPT.compute_high_growth_score_SP500_GPT"], name, lambda *a, **k: None)

from GPT.run_complete_process import _load_sel_cfg, _SEL_DEFAULTS


class TestLoadSelCfg(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(self._testMethodName)
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.tmpdir.iterdir():
            f.unlink()
        self.tmpdir.rmdir()

    def _write_cfg(self, content: str) -> Path:
        cfg = self.tmpdir / "cfg.ini"
        cfg.write_text(content)
        return cfg

    def test_defaults(self):
        cfg = self._write_cfg("")
        res = _load_sel_cfg(cfg)
        self.assertEqual(res, _SEL_DEFAULTS)

    def test_override(self):
        content = """[selection]\ntrend_thresh = 80\nw_core = 0.5\n"""
        cfg = self._write_cfg(content)
        res = _load_sel_cfg(cfg)
        expected = dict(_SEL_DEFAULTS)
        expected["trend_thresh"] = 80
        expected["w_core"] = 0.5
        self.assertEqual(res, expected)

    def test_ignore_empty(self):
        content = """[selection]\noutput_name =\nfund_file =\n"""
        cfg = self._write_cfg(content)
        res = _load_sel_cfg(cfg)
        self.assertEqual(res["output_name"], _SEL_DEFAULTS["output_name"])
        self.assertEqual(res["fund_file"], _SEL_DEFAULTS["fund_file"])


if __name__ == "__main__":
    unittest.main()
