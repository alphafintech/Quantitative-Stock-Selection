# -*- coding: utf-8 -*-
"""Pipeline orchestrator (v2, config driven)
============================================
Coordinate the S&P 500 trend and fundamental pipelines and output the
"high‑growth" stock list.

Key changes
-----------
* All ``selection`` parameters are configurable via ``config_run.ini``.
* ``composite_selection()`` reads defaults from that section if not
  explicitly passed.
* ``run_pipeline()`` loads the selection defaults with ``_load_sel_cfg``.
* The CLI offers ``--cfg`` to specify a custom ``config_run`` file.

Example usage
-------------
```
python orchestrator_select.py                # use default config_run.ini
python orchestrator_select.py --cfg myrun.ini --no-recalc
```
"""
from __future__ import annotations
import logging, sys, re, glob, subprocess, shutil, datetime as dt, configparser
from pathlib import Path
from typing import Optional, Dict, Any

# ------------------------------------------------------------
# Paths / config locations
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CFG_RUN      = ROOT / "config_run.ini"
CFG_TREND    = ROOT / "config_trend.ini"
CFG_FINANCE  = ROOT / "config_finance.ini"

# ------------------------------------------------------------
# Import dependent pipeline functions
# ------------------------------------------------------------
from .Compute_Trend_score_SP500_GPT import run_process_control as trend_run_ctrl
from .compute_high_growth_score_SP500_GPT import (
    compute_metrics as FIN_METRICS,
    calc_scores as FIN_SCORES,
    export_excel as FIN_EXPORT,
)

# ------------------------------------------------------------
# logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("orchestrator")

# ------------------------------------------------------------
# Helper: find the newest file
# ------------------------------------------------------------
_DATE_RE = re.compile(r"(\d{8})")

def _latest_file(pattern: str) -> Optional[Path]:
    files = sorted(Path().glob(pattern),
                   key=lambda p: dt.datetime.strptime(_DATE_RE.search(p.stem)[1], "%Y%m%d")
                   if _DATE_RE.search(p.stem) else dt.datetime.min,
                   reverse=True)
    return files[0] if files else None

# ------------------------------------------------------------
# Read selection configuration
# ------------------------------------------------------------
# Apart from thresholds/weights, explicit paths to the generated trend and
# fundamental Excel files can be given; empty values fall back to automatic
# search.
#
# [selection]
# trend_file = trend_scores.xlsx
# fund_file  = my_fundamental_scores.xlsx
# ... rest of the parameters are the same
# ------------------------------------------------------------
_SEL_DEFAULTS: Dict[str, Any] = {
    "output_name"      : "composite_selection.xlsx",
    "top_num_trend"    : 100,   # take top N by trend score
    "top_num_growth"   : 70,    # take top M by fundamental score
    "trend_file"       : "",    # explicit file can be blank
    "fund_file"        : "",
    "trend_thresh"     : 70,
    "fund_thresh"      : 70,
    "growth_thresh"    : 80,
    "w_core"           : 0.8,
    "w_growth"         : 0.2,
}

def _load_sel_cfg(cfg_path: Path) -> Dict[str, Any]:
    """Read `[selection]` from *cfg_path* and merge with ``_SEL_DEFAULTS``.

    Empty values in the config file will not override the defaults.
    """

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")

    sel = {k: v for k, v in _SEL_DEFAULTS.items()}  # clone defaults
    if cfg.has_section("selection"):
        s = cfg["selection"]
        for k in sel:
            if k in s and s[k]:  # ignore blank values
                if k.startswith("w_"):
                    sel[k] = float(s[k])
                elif k.endswith("_thresh"):
                    sel[k] = int(s[k])
                else:
                    sel[k] = s[k]
    return sel

# ------------------------------------------------------------
# Step wrappers
# ------------------------------------------------------------

def build_trend_scores(run_stage) -> None:
    """(Re)calculate trend scores if requested."""
    stage = run_stage
    log.info("[Trend] run_process_control(stage=%d)", stage)
    trend_run_ctrl(stage)



def build_finance_scores(*, recalc_scores: bool) -> None:
    if recalc_scores:
        log.info("[Finance] recomputing scores …")
        from .compute_high_growth_score_SP500_GPT import _prepare_gpt_finance_db

        db_path = _prepare_gpt_finance_db()
        metrics = FIN_METRICS(db_path=db_path)
        scores  = FIN_SCORES(metrics)
        FIN_EXPORT(scores)
    if not recalc_scores:
        log.info("[Finance] skipped")


def composite_selection(cfg_sel: Dict[str, Any]) -> Path:
    """
    Combine filtering using the new rules:
        • Ranked within *top_num_trend* in ``trend_scores.xlsx``
        • Ranked within *top_num_growth* in ``high_growth_scoring.xlsx``
      The intersection of these stocks is retained.

    Output columns: ticker | trend_score | fund_score | final_score
    where ``final_score = (trend_score + fund_score) / 2``.
    """

    import pandas as pd

    # ---------- 1) Resolve file paths ----------
    def _resolve(name_key: str, fixed: str, pattern: str) -> Optional[Path]:
        cfg_path = cfg_sel.get(name_key, "").strip()
        if cfg_path and Path(cfg_path).exists():
            return Path(cfg_path)
        root = Path()
        if Path(fixed).exists():
            return Path(fixed)
        latest = _latest_file(pattern)
        if latest:
            return latest
        return None

    trend_file = _resolve("trend_file", "trend_scores.xlsx", "trend_scores*.xlsx")
    fund_file  = _resolve("fund_file",  "high_growth_scoring.xlsx", "high_growth_scoring*.xlsx")
    if trend_file is None or fund_file is None:
        log.error("[Select] Required Excel not found.")
        raise FileNotFoundError

    # ---------- 2) Read Top‑N ----------
    top_trend  = int(cfg_sel.get("top_num_trend", 100))
    top_growth = int(cfg_sel.get("top_num_growth", 70))

    trend_df = (
        pd.read_excel(trend_file)
          .sort_values("TotalScore", ascending=False, na_position="last")
          .head(top_trend)[["Ticker", "TotalScore"]]
          .rename(columns={"Ticker":"ticker", "TotalScore":"trend_score"})
    )

    fund_df = (
        pd.read_excel(fund_file)
          .sort_values("total_score", ascending=False, na_position="last")
          .head(top_growth)[["ticker", "total_score"]]
          .rename(columns={"total_score":"fund_score"})
    )

    # ---------- 3) Intersection ----------
    merged = trend_df.merge(fund_df, on="ticker", how="inner")
    if merged.empty:
        log.warning("[Select] No overlap between top Trend and top Growth lists.")
        out_path = Path(cfg_sel["output_name"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_excel(out_path, index=False)
        return out_path

    # ---------- 4) Calculate final_score ----------
    merged["final_score"] = ((merged["trend_score"] + merged["fund_score"]) / 2).round(1)
    merged = merged.sort_values("final_score", ascending=False)

    # ---------- 5) Output ----------
    out_path = Path(cfg_sel["output_name"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["ticker", "trend_score", "fund_score", "final_score"]].to_excel(out_path, index=False)
    log.info("[Select] Exported composite_selection → %s (%d stocks)", out_path, len(merged))
    return out_path

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def run_pipeline(*,
                 trend_run_stage: int = 0,
                 recalc_scores: bool = True,
                 do_selection: bool = True,
                 cfg_run: Path = CFG_RUN) -> None:
    start = dt.datetime.now()
    log.info("========== PIPELINE START ==========")

    # 1) Trend scores
    build_trend_scores(trend_run_stage)  # 0: run all stages

    # 2) Fundamental scores
    build_finance_scores(recalc_scores=recalc_scores)

    # 3) Composite selection
    if do_selection:
        sel_cfg = _load_sel_cfg(cfg_run)
        composite_selection(sel_cfg)

    log.info("========== PIPELINE END (%.1fs) =========", (dt.datetime.now()-start).total_seconds())

# ------------------------------------------------------------
# Quick unit‑test helper
# ------------------------------------------------------------

def test_pipeline():
    """Convenience wrapper: run scoring and selection using the current
    ``config_run.ini``."""
    run_pipeline(
        recalc_scores=True,      # recompute trend and fundamental scores
        do_selection=True,
        cfg_run=CFG_RUN)

#test_pipeline()
# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
#if __name__ == "__main__":
def main():
    import argparse
    p = argparse.ArgumentParser(description="S&P500 Trend + Fundamental orchestrator")
    p.add_argument("--cfg", default=str(CFG_RUN), help="path to config_run.ini")
    p.add_argument("--no-recalc", dest="recalc", action="store_false")
    p.add_argument("--no-select", dest="select", action="store_false")
    args = p.parse_args()

    run_pipeline(recalc_scores=args.recalc,
                 do_selection=args.select,
                 cfg_run=Path(args.cfg))
