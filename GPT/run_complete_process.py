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
python orchestrator_select.py --update-trend-db --update-finance-db

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
from .Compute_Trend_score_SP500_GPT import (
    run_process_control as trend_run_ctrl,
    sync_from_common_db as trend_sync,
)
from .compute_high_growth_score_SP500_GPT import (
    sync_from_common_db as finance_sync,
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
    "output_name"   : "composite_selection.xlsx",
    "trend_thresh"  : 70,
    "fund_thresh"   : 70,
    "growth_thresh" : 80,
    "w_core"        : 0.8,
    "w_growth"      : 0.2,
    "trend_file"    : "",   # 可留空让代码自动寻找
    "fund_file"     : "",
}

def _load_sel_cfg(cfg_path: Path) -> Dict[str, Any]:
    """Read `[selection]` from *cfg_path* and merge with ``_SEL_DEFAULTS``.

    Empty values in the config file will not override the defaults.
    """

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

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

def build_trend_scores(update_db: bool, recalc_scores: bool) -> None:
    if update_db:
        log.info("[Trend] syncing from common DB …")
        trend_sync("../SP500_price_data.db")
        trend_run_ctrl(2)
        trend_run_ctrl(3)
    elif recalc_scores:
        log.info("[Trend] recomputing trend scores …")
        trend_run_ctrl(3)
    else:
        log.info("[Trend] skipped")


def build_finance_scores(update_db: bool, recalc_scores: bool) -> None:
    if update_db:
        log.info("[Finance] syncing from common DB …")
        finance_sync("../SP500_finance_data.db")
    if recalc_scores or update_db:
        log.info("[Finance] recomputing scores …")
        metrics = FIN_METRICS()
        scores = FIN_SCORES(metrics)
        FIN_EXPORT(scores)
    else:
        log.info("[Finance] skipped")


def composite_selection(cfg_sel: Dict[str, Any]) -> Path:
    """按照配置生成最终榜单。

    文件查找顺序（趋势分 / 基本面分分别执行）：
      1. cfg_sel 中显式 `trend_file` / `fund_file` 且文件存在。
      2. 固定文件名   trend_scores.xlsx / high_growth_scoring.xlsx
      3. 通配最新     trend_scores*.xlsx / high_growth_scoring_*.xlsx
    以上均不存在时抛 FileNotFoundError。
    """
    import pandas as pd

    # ---------- 1) 解析文件路径 ----------
    tried: list[str] = []

    def _resolve(name_key: str, fixed: str, pattern: str) -> Optional[Path]:
        """按优先级返回存在的文件, 并记录尝试过的路径"""
        # 1) 显式配置
        cfg_path = cfg_sel.get(name_key, "").strip()
        if cfg_path:
            p = Path(cfg_path)
            tried.append(str(p))
            if p.exists():
                return p
        # 2) 固定文件名
        p_fixed = Path(fixed)
        tried.append(str(p_fixed))
        if p_fixed.exists():
            return p_fixed
        # 3) 最新匹配
        latest = _latest_file(pattern)
        if latest is not None:
            tried.append(str(latest))
            return latest
        return None

    trend_file = _resolve("trend_file", "trend_scores.xlsx", "trend_scores*.xlsx")
    fund_file  = _resolve("fund_file",  "high_growth_scoring.xlsx", "high_growth_scoring*.xlsx")

    if trend_file is None or fund_file is None:
        log.error("[Select] Cannot locate required Excel files. Tried: %s", tried)
        raise FileNotFoundError("trend/fund excel not found – run score steps first")

    log.info("[Select] using trend file: %s", trend_file)
    log.info("[Select] using fund  file: %s", fund_file)

    # ---------- 2) 读取并合并 ----------
    trend_df = pd.read_excel(trend_file)[["Ticker", "TotalScore"]].rename(
        columns={"Ticker": "ticker", "TotalScore": "trend_score"})
    fund_df  = pd.read_excel(fund_file)[["ticker", "total_score", "growth_score"]].rename(
        columns={"total_score": "fund_score", "growth_score": "growth_sub"})
    merged = pd.merge(trend_df, fund_df, on="ticker", how="inner")

    # ---------- 3) 阈值过滤 ----------
    trend_th, fund_th, grow_th = (
        cfg_sel["trend_thresh"], cfg_sel["fund_thresh"], cfg_sel["growth_thresh"])
    filt = merged.query("trend_score >= @trend_th and fund_score >= @fund_th and growth_sub >= @grow_th")
    if filt.empty:
        log.warning("[Select] 0 stocks pass thresholds")
        return Path()

    # ---------- 4) 综合得分 ----------
    T, F, G = filt["trend_score"] / 100, filt["fund_score"] / 100, filt["growth_sub"] / 100
    tf_core = 2 * T * F / (T + F)
    filt["final_score"] = cfg_sel["w_core"] * tf_core + cfg_sel["w_growth"] * G
    out_df = filt.sort_values("final_score", ascending=False)

    out_path = Path(cfg_sel["output_name"])
    out_df.to_excel(out_path, index=False)
    log.info("[Select] exported → %s (%d stocks)", out_path, len(out_df))
    return out_path

# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------

def run_pipeline(*,
                 update_trend_db: bool = False,
                 update_finance_db: bool = False,
                 recalc_scores: bool = True,
                 do_selection: bool = True,
                 cfg_run: Path = CFG_RUN) -> None:
    start = dt.datetime.now()
    log.info("========== PIPELINE START ==========")

    # 1) 趋势分
    build_trend_scores(update_db=update_trend_db, recalc_scores=recalc_scores)

    # 2) 基本面分
    build_finance_scores(update_db=update_finance_db, recalc_scores=recalc_scores)

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
    run_pipeline(update_trend_db=False,  # 不刷新历史/价格 DB
                 update_finance_db=False,  # 不刷新财报 DB
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
    p.add_argument("--update-trend-db", action="store_true")
    p.add_argument("--update-finance-db", action="store_true")
    p.add_argument("--no-recalc", dest="recalc", action="store_false")
    p.add_argument("--no-select", dest="select", action="store_false")
    args = p.parse_args()

    run_pipeline(update_trend_db=args.update_trend_db,
                 update_finance_db=args.update_finance_db,
                 recalc_scores=args.recalc,
                 do_selection=args.select,
                 cfg_run=Path(args.cfg))
