# -*- coding: utf-8 -*-
"""orchestrator_select.py  (v2 — config‑driven)
============================================================
统一调度 S&P 500 趋势分与基本面分流水线, 并输出“优质成长股”榜单。

主要变动
---------
* **selection 参数全部可在 config_run.ini 里调节**；代码预设区段：

```ini
[selection]
output_name   = composite_selection.xlsx  ; 导出文件名
trend_thresh  = 70                         ; 趋势分过滤 ≥
fund_thresh   = 70                         ; 基本面总分过滤 ≥
growth_thresh = 80                         ; Growth‑sub 过滤 ≥
w_core        = 0.8                        ; TF_core 权重
w_growth      = 0.2                        ; Growth 加成权重
```

* `composite_selection()` 若未显式传参，将从该区段自动读取；缺失项回退到硬编码默认。
* `run_pipeline()` 在执行前调用 `_load_sel_cfg()` 获取 selection 默认值。
* CLI 增加 `--cfg` 选项指定自定义 config_run 文件。

外部使用示例
-------------
```bash
# 使用默认 config_run.ini
python orchestrator_select.py

# 指定另一个配置文件并仅做筛选
python orchestrator_select.py --cfg myrun.ini --no-recalc
```
"""
from __future__ import annotations
import logging, sys, re, glob, subprocess, shutil, datetime as dt, configparser
from pathlib import Path
from typing import Optional, Dict, Any

# ------------------------------------------------------------
# 目录 / 配置文件定位
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CFG_RUN      = ROOT / "config_run.ini"
CFG_TREND    = ROOT / "config_trend.ini"
CFG_FINANCE  = ROOT / "config_finance.ini"

# ------------------------------------------------------------
# 依赖脚本函数导入
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
# 辅助：找出最新文件
# ------------------------------------------------------------
_DATE_RE = re.compile(r"(\d{8})")

def _latest_file(pattern: str) -> Optional[Path]:
    files = sorted(Path().glob(pattern),
                   key=lambda p: dt.datetime.strptime(_DATE_RE.search(p.stem)[1], "%Y%m%d")
                   if _DATE_RE.search(p.stem) else dt.datetime.min,
                   reverse=True)
    return files[0] if files else None

# ------------------------------------------------------------
# 读取 selection 配置
# ------------------------------------------------------------
# 除阈值/权重外，允许显式指定已生成的趋势 & 基本面 Excel
# 若留空则回退到自动搜索逻辑。
#
# [selection]
# trend_file = trend_scores.xlsx
# fund_file  = my_fundamental_scores.xlsx
# ... 其余参数同前
# ------------------------------------------------------------
_SEL_DEFAULTS: Dict[str, Any] = {
    "output_name"      : "composite_selection.xlsx",
    "top_num_trend"    : 100,   # 取趋势分前 N 名
    "top_num_growth"   : 70,    # 取基本面分前 M 名
    "trend_file"       : "",    # 显式指定文件可留空
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
# 步骤封装
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
    组合筛选（新规则）:
        • 在 trend_scores.xlsx 排名前 *top_num_trend*
        • 且在 high_growth_scoring.xlsx 排名前 *top_num_growth*
      的股票交集。

    输出列: ticker | trend_score | fund_score | final_score
    final_score = (trend_score + fund_score) / 2
    """

    import pandas as pd

    # ---------- 1) 解析文件路径 ----------
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

    # ---------- 2) Top‑N 读取 ----------
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

    # ---------- 3) 交集 ----------
    merged = trend_df.merge(fund_df, on="ticker", how="inner")
    if merged.empty:
        log.warning("[Select] No overlap between top Trend and top Growth lists.")
        out_path = Path(cfg_sel["output_name"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_excel(out_path, index=False)
        return out_path

    # ---------- 4) 计算 final_score ----------
    merged["final_score"] = ((merged["trend_score"] + merged["fund_score"]) / 2).round(1)
    merged = merged.sort_values("final_score", ascending=False)

    # ---------- 5) 输出 ----------
    out_path = Path(cfg_sel["output_name"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["ticker", "trend_score", "fund_score", "final_score"]].to_excel(out_path, index=False)
    log.info("[Select] Exported composite_selection → %s (%d stocks)", out_path, len(merged))
    return out_path

# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------

def run_pipeline(*,
                 trend_run_stage: int = 0,
                 recalc_scores: bool = True,
                 do_selection: bool = True,
                 cfg_run: Path = CFG_RUN) -> None:
    start = dt.datetime.now()
    log.info("========== PIPELINE START ==========")

    # 1) 趋势分
    build_trend_scores(trend_run_stage)  # 0: run all stages

    # 2) 基本面分
    build_finance_scores(recalc_scores=recalc_scores)

    # 3) 组合筛选
    if do_selection:
        sel_cfg = _load_sel_cfg(cfg_run)
        composite_selection(sel_cfg)

    log.info("========== PIPELINE END (%.1fs) =========", (dt.datetime.now()-start).total_seconds())

# ------------------------------------------------------------
# Quick unit‑test helper
# ------------------------------------------------------------

def test_pipeline():
    """Convenience wrapper: 使用当前 config_run.ini 只做评分 + 组合筛选。"""
    run_pipeline(
        recalc_scores=True,      # 重新计算趋势+基本面分
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
