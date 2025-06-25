# -*- coding: utf-8 -*-
"""
Main entry script ``Run_complete_program.py``.

Features
--------
1. Parse command‑line options to control the S&P 500 workflow (Gemini and/or GPT).
2. Invoke ``Gemini/run_sp500_processing.py``'s ``main_pipeline`` when needed.
3. Invoke ``GPT/run_complete_process.py``'s ``run_pipeline`` when needed.
4. Allow the user to skip data updates or filtering steps.

Command‑line arguments:
--skip-update-price-data   Skip ``yahoo_downloader.download_price_data()``
--skip-update-finance-data Skip ``yahoo_downloader.acquire_raw_financial_data_to_staging()``
--skip-Gemini-pipeline     Skip the Gemini pipeline
--skip-GPT-pipeline        Skip the GPT pipeline

Example usage:
  python Run_complete_program.py                       # run everything
  python Run_complete_program.py --skip-Gemini-pipeline
  python Run_complete_program.py --skip-GPT-pipeline
  python Run_complete_program.py --skip-update-price-data
  python Run_complete_program.py --skip-update-finance-data
"""

import argparse
import logging
import sys
import os
import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
import configparser
import datetime as dt
import yahoo_downloader
import sqlite3
import re
import html
from datetime import datetime, timedelta

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [MainRunner] %(message)s')
Gemini_dir = "Gemini"
GPT_dir = "GPT"
def change_working_directory(next_directory = ""):
    """
    Change the current working directory to the script location or a subdirectory.
    """
    # Get the absolute path of this script
    script_path = os.path.abspath(__file__)
    # Directory of the script
    script_dir = os.path.dirname(script_path)
    # Switch CWD to the script directory or its subdirectory
    if next_directory == "":
        os.chdir(script_dir)
        logging.info(f"Current working directory changed to: {script_dir}")
    else:
        next_absolute_dir = os.path.join(script_dir, next_directory)
        os.chdir(next_absolute_dir)
        logging.info(f"Current working directory changed to: {next_absolute_dir}")
# --- Import Gemini pipeline ---
GEMINI_AVAILABLE = False # not available by default
#change_working_directory(Gemini_dir) # switch into Gemini subdirectory
try:
    # 优先尝试从 Gemini 子目录导入
    from Gemini.run_sp500_processing import main_pipeline as gemini_main_pipeline
    logging.info("成功从 Gemini 子目录导入 gemini_main_pipeline 函数。")
    GEMINI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法从 'Gemini.run_sp500_processing' 导入 gemini_main_pipeline: {e}")
    # 备选：尝试直接导入
    try:
        from Gemini.run_sp500_processing import main_pipeline as gemini_main_pipeline
        logging.info("成功直接导入 gemini_main_pipeline 函数。")
        GEMINI_AVAILABLE = True
    except ImportError:
        logging.warning("直接导入 gemini_main_pipeline 也失败。Gemini 流程将不可用。")
except Exception as import_err:
    logging.error(f"导入 gemini_main_pipeline 时发生意外错误: {import_err}", exc_info=True)
    GEMINI_AVAILABLE = False

# --- 导入 GPT 处理流程 ---
GPT_AVAILABLE = False # 默认不可用
#change_working_directory(GPT_dir) # 切换到 GPT 子目录

try:
    # 优先尝试从 GPT 子目录导入
    # **重要**: 假设 run_complete_process.py 在名为 'GPT' 的子目录中
    from GPT.run_complete_process import run_pipeline as gpt_run_pipeline
    logging.info("成功从 GPT 子目录导入 gpt_run_pipeline 函数。")
    GPT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法从 'GPT.run_complete_process' 导入 gpt_run_pipeline: {e}")
    # 备选：尝试直接导入
    try:
        from GPT.run_complete_process import run_pipeline as gpt_run_pipeline
        logging.info("成功直接导入 gpt_run_pipeline 函数。")
        GPT_AVAILABLE = True
    except ImportError:
        logging.warning("直接导入 gpt_run_pipeline 也失败。GPT 流程将不可用。")
except Exception as import_err:
    logging.error(f"导入 gpt_run_pipeline 时发生意外错误: {import_err}", exc_info=True)
    GPT_AVAILABLE = False

#change_working_directory() # 切换回主目录
# --- 主流程执行函数 ---
def run_main_process():
    """
    解析参数并执行 S&P 500 处理流程 (Gemini 和/或 GPT)。
    """
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="运行 S&P 500 股票处理流程 (Gemini 和/或 GPT)。默认运行 Gemini 流程。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据更新控制参数（默认为执行，使用 --skip-* 跳过）
    parser.add_argument(
        '--skip-update-price-data',
        action='store_true',
        help='跳过从雅虎下载股价数据'
    )
    parser.add_argument(
        '--skip-update-finance-data',
        action='store_true',
        help='跳过从雅虎下载财务数据'
    )

    # Gemini/GPT 流程控制参数
    parser.add_argument(
        '--skip-Gemini-pipeline',
        action='store_true',
        help='跳过 Gemini 流程'
    )
    parser.add_argument(
        '--skip-GPT-pipeline',
        action='store_true',
        help='跳过 GPT 流程'
    )


    args = parser.parse_args()
    # 预计算 config.ini 绝对路径
    config_path = Path(__file__).resolve().parent / "config.ini"

    if not args.skip_update_price_data:
        logging.info("--- 下载/更新股价数据 ---")
        yahoo_downloader.download_price_data(config_file=str(config_path))

    if not args.skip_update_finance_data:
        logging.info("--- 下载/更新财务数据 ---")
        yahoo_downloader.acquire_raw_financial_data_to_staging(
            config_file=str(config_path)
        )


    # --- 执行 Gemini 流程 (如果可用且未被跳过) ---
    if GEMINI_AVAILABLE and not args.skip_Gemini_pipeline:
        logging.info("--- 开始执行主流程 (Gemini) ---")
        start_time_gemini = time.time()
        try:
            change_working_directory(Gemini_dir)
            gemini_main_pipeline(True)
            logging.info("--- 主流程 (Gemini) 执行完毕 ---")
        except TypeError as te:
            logging.error(f"调用 gemini_main_pipeline 时发生 TypeError: {te}", exc_info=True)
        except Exception as e:
            logging.error(f"执行 gemini_main_pipeline 时发生意外错误: {e}", exc_info=True)
        finally:
            end_time_gemini = time.time()
            duration_gemini = end_time_gemini - start_time_gemini
            logging.info(f"--- Gemini 流程执行时间: {duration_gemini:.2f} 秒 ---")
    elif args.skip_Gemini_pipeline:
         logging.info("--- 跳过 Gemini 流程 (根据 --skip-Gemini-pipeline 参数) ---")
    elif not GEMINI_AVAILABLE:
         logging.warning("--- Gemini 流程不可用 (导入失败) ---")


    # --- 执行 GPT 流程 (如果可用且未被跳过) ---
    if GPT_AVAILABLE and not args.skip_GPT_pipeline:
        recalc_scores_gpt = True  # 固定为 True
        do_selection_gpt = True  # 固定为 True

        logging.info("--- 开始执行主流程 (GPT) ---")
        logging.info(
            f"GPT 参数: recalc_scores={recalc_scores_gpt}, do_selection={do_selection_gpt}"
        )
        start_time_gpt = time.time()
        try:
            change_working_directory(GPT_dir)
            gpt_run_pipeline(
                trend_run_stage=0,
                recalc_scores=recalc_scores_gpt,
                do_selection=do_selection_gpt
            )
            logging.info("--- 主流程 (GPT) 执行完毕 ---")
        except TypeError as te:
             logging.error(f"调用 gpt_run_pipeline 时发生 TypeError: {te}", exc_info=True)
             logging.error("请检查 GPT/run_complete_process.py 中 run_pipeline 函数的参数定义。")
        except Exception as e:
            logging.error(f"执行 gpt_run_pipeline 时发生意外错误: {e}", exc_info=True)
        finally:
            end_time_gpt = time.time()
            duration_gpt = end_time_gpt - start_time_gpt
            logging.info(f"--- GPT 流程执行时间: {duration_gpt:.2f} 秒 ---")
            change_working_directory()
    elif args.skip_GPT_pipeline and not GPT_AVAILABLE:
         logging.warning("--- GPT 流程不可用 (导入失败)，但已跳过 GPT ---")
    elif not GPT_AVAILABLE:
         logging.warning("--- GPT 流程不可用 (导入失败) ---")


        

#######################################################################
# Helper: build holdings string from Excel + latest prices DB
#######################################################################
def _build_holdings_string(excel_path: Path, price_db: Path | None = None) -> str:
    """
    Parameters
    ----------
    excel_path : Path
        Excel file containing columns: Ticker, Cost Basis, Share Count.
    price_db : Path | None
        SQLite DB containing price history with schema:
            price_data(ticker TEXT, date TEXT, close REAL)
        If None, defaults to script_dir / 'data/SP500_price_data.db'.

    Returns
    -------
    str  e.g. "(HWM:$165.09, 153, $179.00), (RCL:$240.12, 556, $230.00)"
    """
    if not excel_path.exists():
        logging.error(f"[Holdings] Excel file not found: {excel_path}")
        return "N/A"

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logging.error(f"[Holdings] Failed to read holdings Excel: {e}")
        return "N/A"

    # Resolve columns ignoring case/space/underscore/non-alpha, allow prefix matching
    def _resolve(col_hint: str):
        """
        Return the actual column name that matches *col_hint* after
        stripping spaces, underscores, and non‑alphabetic chars.
        Accepts exact match **or** prefix/contain match to cover
        headers like 'Ticker/Cash'.
        """
        norm_hint = re.sub(r"[^a-z]", "", col_hint.lower())
        for c in df.columns:
            norm_c = re.sub(r"[^a-z]", "", str(c).lower())
            if norm_c == norm_hint or norm_c.startswith(norm_hint) or norm_hint.startswith(norm_c):
                return c
        raise KeyError(f"Column '{col_hint}' not found in {excel_path.name}")

    try:
        col_tic   = _resolve("ticker")
        col_cost  = _resolve("cost basis")
        col_share = _resolve("share count")
    except KeyError as ke:
        logging.error(f"[Holdings] {ke}")
        return "N/A"

    df_full = df.copy()  # keep for cash extraction
    df = df[[col_tic, col_cost, col_share]].dropna()
    if df.empty:
        logging.warning(f"[Holdings] No valid rows in {excel_path.name}")
        return "N/A"

    # --- price DB ---
    script_dir = Path(__file__).resolve().parent
    # Try to load price_db from config.ini if not explicitly provided
    if price_db is None:
        cfg_path = script_dir / "config.ini"
        if cfg_path.exists():
            cfg_tmp = configparser.ConfigParser()
            cfg_tmp.read(cfg_path, encoding="utf-8")
            if cfg_tmp.has_section("database"):
                price_db_token = cfg_tmp.get("database", "price_db",
                                             fallback="data/SP500_price_data.db")
                price_db = (script_dir / price_db_token) if not Path(price_db_token).is_absolute() \
                           else Path(price_db_token)
    if price_db is None:
        # primary fallback: root directory
        price_db = script_dir / "SP500_price_data.db"
        # secondary fallback: data sub‑dir
        if not price_db.exists():
            price_db = script_dir / "data" / "SP500_price_data.db"
    latest_price_cache: dict[str, float] = {}

    def _get_price(ticker: str) -> float | None:
        """
        Try to fetch the most recent close/adj_close for *ticker* from the
        SQLite price DB. Falls back to None if not found.
        """
        if ticker in latest_price_cache:
            return latest_price_cache[ticker]

        price_val = None
        if not price_db.exists():
            logging.warning(f"[Holdings-DEBUG] price DB not found at {price_db}.")
            latest_price_cache[ticker] = None
            return None

        try:
            with sqlite3.connect(price_db) as conn:
                # Gather candidate tables that have a `ticker` column
                tbls = [
                    r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                ]
                price_cols = ("close", "adj_close", "close_price", "price")
                date_cols  = ("date", "Date", "trade_date")
                for tbl in tbls:
                    # quick check
                    cols = [c[1].lower() for c in conn.execute(f"PRAGMA table_info('{tbl}')")]
                    if "ticker" not in cols:
                        continue
                    p_col = next((c for c in price_cols if c in cols), None)
                    d_col = next((c for c in date_cols  if c.lower() in cols), None)
                    if not p_col or not d_col:
                        continue

                    try:
                        cur = conn.execute(
                            f"""SELECT {p_col}
                                   FROM {tbl}
                                  WHERE UPPER(ticker)=UPPER(?)
                              ORDER BY {d_col} DESC
                                 LIMIT 1""",
                            (ticker,),
                        )
                        row = cur.fetchone()
                        if row and row[0] is not None:
                            price_val = float(row[0])
                            break
                    except sqlite3.Error:
                        continue
        except Exception as e:
            logging.warning(f"[Holdings] price db query failed: {e}")

        # Debug logging
        if price_val is None:
            logging.warning(f"[Holdings‑DEBUG] No price found for {ticker} in {price_db.name}")
        else:
            logging.debug(f"[Holdings‑DEBUG] Latest price for {ticker} = {price_val:.2f}")
        latest_price_cache[ticker] = price_val
        return price_val

    parts = []
    for _, row in df.iterrows():
        tic   = str(row[col_tic]).upper()
        cost  = float(row[col_cost])
        share = int(row[col_share])
        price = _get_price(tic)
        cost_str  = f"${cost:,.2f}"
        price_str = f"${price:,.2f}" if price is not None else "NA"
        parts.append(f"({tic}:{cost_str}, {share}, {price_str})")

    # ------------------------------------------------------------------
    # Append Cash row if present (strict 'TOTAL VALUE' column, case-insensitive, and Ticker exactly 'cash')
    # ------------------------------------------------------------------
    try:
        # Strict column name 'TOTAL VALUE' (case insensitive match)
        col_total_val = next(
            c for c in df_full.columns
            if c.strip().lower().replace(" ", "") == "totalvalue"
        )
        cash_rows = df_full[df_full[col_tic].astype(str).str.strip().str.lower() == "cash"]
        if not cash_rows.empty:
            cash_val = float(cash_rows.iloc[0][col_total_val])
            cash_str = f"${cash_val:,.0f}"
            parts.append(f"(Cash: {cash_str})")
    except StopIteration:
        logging.debug("[Holdings-DEBUG] 'TOTAL VALUE' column not found; cash not appended")
    except Exception as e_cash:
        logging.debug(f"[Holdings-DEBUG] Cash row not appended: {e_cash}")

    return ", ".join(parts)

# ------------------------------------------------------------
# Gemini prompt generator
# ------------------------------------------------------------
def generate_prompt_Gemini() -> str:
    """
    根据 Gemini 结果文件和 config.ini 生成最终 prompt。

    步骤
    ----
    1. 读取项目根目录下的 config.ini，定位:
       - [Gemini] screened_stocks_file_Gemini       (Excel 文件名)
       - [prompt_common] 里各占位符参数
    2. 读取 Excel:
       - Sheet Dual_Threshold_Filter  → ticker 列
       - Sheet Combined_Weighted_Score→ Trend_Score / Fundamental_Score / Combined_Score
       筛选 Dual_Threshold_Filter 里的 tickers 并拼成
       "TICK : (Trend, Fundamental, Combined)" 格式，逗号分隔
    3. 读取当前目录的 current_holdings.txt 作为 <Holdings_list>
    4. 将 <ticker_list>、<Holdings_list> 及 [prompt_common] 里所有
       "<key>" 占位符替换进 base_prompt
    5. 返回替换后的 prompt 字符串
    """
    # 0) 路径准备
    script_dir = Path(__file__).resolve().parent
    cfg_path   = script_dir / "config.ini"
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")

    # 1) 定位 Excel 路径
    try:
        screened_name = cfg["FILES"]["screened_stocks_file_Gemini"]
    except KeyError:
        raise KeyError("[FILES] section 缺少 screened_stocks_file_Gemini")

    # 1a) 构造 Excel 绝对路径
    screened_path = Path(screened_name)

    if screened_path.is_absolute():
        excel_path = screened_path
    elif screened_path.parts and screened_path.parts[0] == "Gemini":
        # 已包含 "Gemini/..." 相对路径
        excel_path = script_dir / screened_path
    else:
        # 仅文件名 → 默认放在 Gemini/results/
        excel_path = script_dir / "Gemini" / "results" / screened_path

    if not excel_path.exists():
        logging.warning(
            f"[Gemini Prompt] 筛选结果文件不存在，跳过生成: {excel_path}"
        )
        return ""

    # 2) 读取 Excel
    df_dual  = pd.read_excel(excel_path, sheet_name="Dual_Threshold_Filter")
    df_score = pd.read_excel(excel_path, sheet_name="Combined_Weighted_Score")

    def _resolve_col(df: pd.DataFrame, col_hint: str) -> str:
        """Return actual column name that matches col_hint ignoring case/underscore."""
        for c in df.columns:
            if c.lower().replace("_", "") == col_hint.lower().replace("_", ""):
                return c
        raise KeyError(f"在表 {df.attrs.get('sheet_name','')} 中找不到列 '{col_hint}'")

    ticker_col_dual  = _resolve_col(df_dual,  "ticker")
    ticker_col_score = _resolve_col(df_score, "ticker")
    tscore_col       = _resolve_col(df_score, "trend_score")
    fscore_col       = _resolve_col(df_score, "fundamental_score")
    cscore_col       = _resolve_col(df_score, "combined_score")

    tickers = df_dual[ticker_col_dual].astype(str).str.upper().unique().tolist()
    df_sel  = df_score[df_score[ticker_col_score].astype(str).str.upper().isin(tickers)]

    # 防止数值缺失
    df_sel = df_sel.fillna("NA")

    ticker_items = [
        f"{row[ticker_col_score]} : ({row[tscore_col]}, {row[fscore_col]}, {row[cscore_col]})"
        for _, row in df_sel.iterrows()
    ]
    ticker_list_str = ", ".join(ticker_items)

    # 3) 读取当前持仓
    holdings_token = cfg["FILES"].get("current_holdings_file_Gemini",
                                      "Portfolio/Gemini_current_holdings.xlsx")
    holdings_path  = Path(holdings_token)
    if not holdings_path.is_absolute():
        holdings_path = script_dir / holdings_path
    holdings_list = _build_holdings_string(holdings_path)

    # 4) 读取 base_prompt 以及其他占位符内容
    prompt_common = cfg["prompt_common"]

    # --- 4a. 处理 base_prompt -------------------------------------------------
    if "base_prompt" not in prompt_common:
        raise KeyError("[prompt_common] section 缺少 base_prompt")

    _bp_token = prompt_common["base_prompt"].strip()

    # 若值指向一个 txt 文件，则加载文件内容；否则直接把字符串当模板
    _bp_path = (script_dir / _bp_token) if not Path(_bp_token).is_absolute() else Path(_bp_token)
    if _bp_path.exists() and _bp_path.suffix.lower() == ".txt":
        base_prompt = _bp_path.read_text(encoding="utf-8")
    else:
        base_prompt = _bp_token

    final_prompt = base_prompt

    # --- 4b. 替换通用占位符 ---------------------------------------------------
    final_prompt = final_prompt.replace("<ticker_list>", ticker_list_str)
    final_prompt = final_prompt.replace("<Holdings_list>", holdings_list)

    # --- 保证 end_date 存在 (默认为今天 YYYY-MM-DD) --------------------------
    if ("end_date" not in prompt_common) or (prompt_common.get("end_date", "").strip() == ""):
        prompt_common["end_date"] = dt.date.today().strftime("%Y-%m-%d")

    # --- Ensure start_date (derive from end_date and previous_days) ----------
    if ("start_date" not in prompt_common) or (prompt_common.get("start_date", "").strip() == ""):
        # previous_days comes from [prompt_common]; default to 14 if missing/invalid
        try:
            prev_days = int(prompt_common.get("previous_days", "14").strip())
        except ValueError:
            prev_days = 14
        try:
            end_dt = dt.date.fromisoformat(prompt_common["end_date"])
        except ValueError:
            # Fallback: treat as today if end_date malformed
            end_dt = dt.date.today()
        start_dt = end_dt - dt.timedelta(days=prev_days)
        prompt_common["start_date"] = start_dt.strftime("%Y-%m-%d")

    # 对 [prompt_common] 其余键同样支持“指向 txt 文件”
    for key, value in prompt_common.items():
        if key == "base_prompt":
            continue  # 已处理
        token_value = value.strip()
        token_path  = (script_dir / token_value) if not Path(token_value).is_absolute() else Path(token_value)
        if token_path.exists() and token_path.suffix.lower() == ".txt":
            replacement = token_path.read_text(encoding="utf-8").strip()
        else:
            replacement = token_value
        # 先替换完整占位符 <key>
        final_prompt = final_prompt.replace(f"<{key}>", replacement)
        # 若键名以 _Gemini 结尾，再额外替换去掉后缀的占位符
        if key.lower().endswith("_gemini"):
            bare_key = key[:-7]  # 去掉 '_Gemini'
            final_prompt = final_prompt.replace(f"<{bare_key}>", replacement)

    # --- 4c. 处理 [prompt_description] 区段 (算法概述等) -----------------------
    if cfg.has_section("prompt_description"):
        prompt_desc = cfg["prompt_description"]
        for key, value in prompt_desc.items():
            token_value = value.strip()
            token_path  = (script_dir / token_value) if not Path(token_value).is_absolute() else Path(token_value)
            if token_path.exists() and token_path.suffix.lower() == ".txt":
                replacement = token_path.read_text(encoding="utf-8").strip()
            else:
                replacement = token_value
            final_prompt = final_prompt.replace(f"<{key}>", replacement)
    
    # 5) 将 prompt 写入指定输出目录
    try:
        files_sec     = cfg["FILES"]
        out_dir_name  = files_sec.get("output_dir", "Output")
        prompt_fname  = files_sec.get("AI_selection_Prompt_Gemini",
                                      "AI_selection_Prompt_Gemini.txt")
    except KeyError:
        # 若缺失 [FILES] 区段则回落到默认
        out_dir_name  = "Output"
        prompt_fname  = "AI_selection_Prompt_Gemini.txt"

    out_dir = script_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = out_dir / prompt_fname
    prompt_path.write_text(final_prompt, encoding="utf-8")
    logging.info(f"[Gemini Prompt] 已生成并写入 → {prompt_path}")

    return final_prompt

def generate_prompt_GPT() -> str:
    """
    根据 GPT 结果文件和 config.ini 生成最终 prompt。

    差异说明
    --------
    • Excel 文件由 [FILES] → screened_stocks_file_GPT 指定，
      若为纯文件名，默认位于 GPT/results/ 子目录。
    • 支持占位符 *_GPT 及去除后缀两种写法。
    • 输出文件名默认 AI_selection_Prompt_GPT.txt，可在
      [FILES] → AI_selection_Prompt_GPT 中自定义。
    """
    # 0) 路径准备
    script_dir = Path(__file__).resolve().parent
    cfg_path   = script_dir / "config.ini"
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")

    # 1) 定位 Excel 路径
    try:
        screened_name = cfg["FILES"]["screened_stocks_file_GPT"]
    except KeyError:
        raise KeyError("[FILES] section 缺少 screened_stocks_file_GPT")

    screened_path = Path(screened_name)
    if screened_path.is_absolute():
        excel_path = screened_path
    elif screened_path.parts and screened_path.parts[0] == "GPT":
        excel_path = script_dir / screened_path
    else:
        excel_path = script_dir / "GPT" / "results" / screened_path

    if not excel_path.exists():
        raise FileNotFoundError(f"未找到 GPT 筛选结果文件: {excel_path}")

    # 2) 读取 Excel（GPT 版本仅有一个 sheet）
    # ------------------------------------------------------------------
    # 获取列名配置（允许缺省，忽略大小写 / 下划线）
    files_sec = cfg["FILES"]
    cfg_ticker_col = files_sec.get("GPT_ticker_col", "ticker")
    cfg_trend_col  = files_sec.get("GPT_trend_score_col", "trend_score")
    cfg_fund_col   = files_sec.get("GPT_fundamental_col", "fund_score")
    cfg_final_col  = files_sec.get("GPT_final_score_col", "final_score")

    df = pd.read_excel(excel_path)  # 仅一个 sheet

    def _resolve_col(df: pd.DataFrame, col_hint: str) -> str:
        """Return actual column name that matches col_hint ignoring case/underscore."""
        for c in df.columns:
            if c.lower().replace("_", "") == col_hint.lower().replace("_", ""):
                return c
        raise KeyError(f"在文件 {excel_path.name} 中找不到列 '{col_hint}'")

    ticker_col = _resolve_col(df, cfg_ticker_col)
    trend_col  = _resolve_col(df, cfg_trend_col)
    fund_col   = _resolve_col(df, cfg_fund_col)
    final_col  = _resolve_col(df, cfg_final_col)

    df = df.fillna("NA")

    ticker_items = [
        f"{row[ticker_col]} : ({row[trend_col]}, {row[fund_col]}, {row[final_col]})"
        for _, row in df.iterrows()
    ]
    ticker_list_str = ", ".join(ticker_items)

    # 3) 读取当前持仓
    holdings_token = cfg["FILES"].get("current_holdings_file_GPT",
                                      "Portfolio/GPT_current_holdings.xlsx")
    holdings_path  = Path(holdings_token)
    if not holdings_path.is_absolute():
        holdings_path = script_dir / holdings_path
    holdings_list = _build_holdings_string(holdings_path)

    # 4) base_prompt 与占位符
    prompt_common = cfg["prompt_common"]
    if "base_prompt" not in prompt_common:
        raise KeyError("[prompt_common] section 缺少 base_prompt")

    _bp_token = prompt_common["base_prompt"].strip()
    _bp_path  = (script_dir / _bp_token) if not Path(_bp_token).is_absolute() else Path(_bp_token)
    base_prompt = _bp_path.read_text(encoding="utf-8") if _bp_path.exists() and _bp_path.suffix.lower()==".txt" else _bp_token
    final_prompt = base_prompt

    # 替换基础占位符
    final_prompt = final_prompt.replace("<ticker_list>", ticker_list_str)
    final_prompt = final_prompt.replace("<Holdings_list>", holdings_list)

    # end_date 默认化
    if ("end_date" not in prompt_common) or (prompt_common.get("end_date", "").strip() == ""):
        prompt_common["end_date"] = dt.date.today().strftime("%Y-%m-%d")

    # --- Ensure start_date (derive from end_date and previous_days) ----------
    if ("start_date" not in prompt_common) or (prompt_common.get("start_date", "").strip() == ""):
        try:
            prev_days = int(prompt_common.get("previous_days", "14").strip())
        except ValueError:
            prev_days = 14
        try:
            end_dt = dt.date.fromisoformat(prompt_common["end_date"])
        except ValueError:
            end_dt = dt.date.today()
        start_dt = end_dt - dt.timedelta(days=prev_days)
        prompt_common["start_date"] = start_dt.strftime("%Y-%m-%d")

    # 4b) 替换 [prompt_common] 其余键
    for key, value in prompt_common.items():
        if key == "base_prompt":
            continue
        token_value = value.strip()
        token_path  = (script_dir / token_value) if not Path(token_value).is_absolute() else Path(token_value)
        replacement = token_path.read_text(encoding="utf-8").strip() if token_path.exists() and token_path.suffix.lower()==".txt" else token_value
        final_prompt = final_prompt.replace(f"<{key}>", replacement)
        if key.lower().endswith("_gpt"):
            bare_key = key[:-4]  # 去掉 '_GPT'
            final_prompt = final_prompt.replace(f"<{bare_key}>", replacement)

    # 4c) prompt_description (同 Gemini)
    if cfg.has_section("prompt_description"):
        for key, value in cfg["prompt_description"].items():
            token_value = value.strip()
            token_path  = (script_dir / token_value) if not Path(token_value).is_absolute() else Path(token_value)
            replacement = token_path.read_text(encoding="utf-8").strip() if token_path.exists() and token_path.suffix.lower()==".txt" else token_value
            final_prompt = final_prompt.replace(f"<{key}>", replacement)

    # 5) 输出文件
    files_sec    = cfg["FILES"]
    out_dir_name = files_sec.get("output_dir", "Output")
    prompt_fname = files_sec.get("AI_selection_Prompt_GPT", "AI_selection_Prompt_GPT.txt")

    out_dir = script_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / prompt_fname
    prompt_path.write_text(final_prompt, encoding="utf-8")
    logging.info(f"[GPT Prompt] 已生成并写入 → {prompt_path}")

    return final_prompt

#
# ------------------------------------------------------------
# Helper: load & normalize fundamental overall scores to 0‑100
# ------------------------------------------------------------

def _get_normalized_fund_scores(cfg: configparser.ConfigParser,
                                script_dir: Path) -> dict[str, int]:
    """
    Returns { TICKER: normalized_overall_score } where scores are
    scaled linearly to 0‑100 based on min / max in the Excel file
    specified by [FILES] fundamental_scores_file_Gemini.
    """
    token = cfg["FILES"].get(
        "fundamental_scores_file_Gemini",
        "sp500_fundamental_scores.xlsx",
    )
    # Always resolve to script_dir / "Gemini" / "results" / <filename>
    path = script_dir / "Gemini" / "results" / Path(token).name

    if not path.exists():
        logging.warning(f"[to_html] Fundamental score file not found: {path}")
        return {}

    try:
        df_fund = pd.read_excel(path)
    except Exception as e:
        logging.warning(f"[to_html] Could not read fundamental Excel: {e}")
        return {}

    # Resolve columns case‑insensitively
    def _res_col(df, hint):
        for c in df.columns:
            if c.lower().replace("_", "") == hint.lower().replace("_", ""):
                return c
        raise KeyError

    try:
        col_tic  = _res_col(df_fund, "ticker")
        col_ovrl = _res_col(df_fund, "overall_score")
    except KeyError:
        logging.warning("[to_html] Columns 'Ticker' or 'Overall_Score' not found in fundamental Excel.")
        return {}

    df_fund = df_fund[[col_tic, col_ovrl]].dropna()
    if df_fund.empty:
        return {}

    mn, mx = df_fund[col_ovrl].min(), df_fund[col_ovrl].max()
    if mx == mn:
        df_fund["norm"] = 50
    else:
        df_fund["norm"] = ((df_fund[col_ovrl] - mn) / (mx - mn) * 100).round(0).astype(int)

    return dict(zip(df_fund[col_tic].astype(str).str.upper(), df_fund["norm"].astype(int)))

# ------------------------------------------------------------
# Helper: generic normalize a column to 0‑100
# ------------------------------------------------------------
def _build_norm_map(excel_path: Path,
                    value_col_hint: str,
                    ticker_col_hint: str = "ticker") -> dict[str, int]:
    """
    Read *excel_path* and return {ticker: norm_score} with the selected
    column scaled linearly to 0‑100.  Gracefully returns {} on error.
    """
    if not excel_path.exists():
        logging.warning(f"[to_html] Normalization source not found: {excel_path}")
        return {}

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logging.warning(f"[to_html] Failed to read {excel_path.name}: {e}")
        return {}

    def _res(df: pd.DataFrame, hint: str):
        for c in df.columns:
            if c.lower().replace("_", "") == hint.lower().replace("_", ""):
                return c
        raise KeyError

    try:
        col_tic = _res(df, ticker_col_hint)
        col_val = _res(df, value_col_hint)
    except KeyError:
        logging.warning(f"[to_html] Columns '{ticker_col_hint}' or '{value_col_hint}' "
                        f"not found in {excel_path.name}")
        return {}

    df = df[[col_tic, col_val]].dropna()
    if df.empty:
        return {}

    mn, mx = df[col_val].min(), df[col_val].max()
    df["norm"] = 50 if mn == mx else ((df[col_val]-mn)/(mx-mn)*100).round(0)
    return dict(zip(df[col_tic].astype(str).str.upper(), df["norm"].astype(int)))

# ------------------------------------------------------------
# Export Gemini screened_stocks Excel to a single‑page HTML
# ------------------------------------------------------------

def read_gemini_results():
    """
    读取 Gemini 相关结果文件（均在 script_dir / "Gemini" / "results" /）:
      - sp500_fundamental_scores.xlsx
      - sp500_trend_scores_Gemini.xlsx
      - screened_stocks.xlsx
    返回 dict: {'fundamental': df1, 'trend': df2, 'screened': df3}
    """
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "Gemini" / "results"

    fund_path = results_dir / "sp500_fundamental_scores.xlsx"
    trend_path = results_dir / "sp500_trend_scores_Gemini.xlsx"
    screened_path = results_dir / "screened_stocks.xlsx"

    dfs = {}
    try:
        dfs['fundamental'] = pd.read_excel(fund_path)
        # --- 归一化 Overall_Score 到 0-100 ---
        if dfs['fundamental'] is not None and not dfs['fundamental'].empty:
            # 查找 overall_score 列名（忽略大小写和下划线）
            def _res_col(df, hint):
                for c in df.columns:
                    if c.lower().replace("_", "") == hint.lower().replace("_", ""):
                        return c
                raise KeyError(f"Column '{hint}' not found.")
            try:
                col_ovrl = _res_col(dfs['fundamental'], "overall_score")
                # 先去除 NA
                valid_scores = dfs['fundamental'][col_ovrl].dropna()
                mn = 0
                mx = valid_scores.max()
                if mx == mn:
                    dfs['fundamental']["normalized_overall_score"] = 50
                else:
                    # 只对非 NA 行归一化，NA 保持为 NA
                    dfs['fundamental']["normalized_overall_score"] = (
                        (dfs['fundamental'][col_ovrl] - mn) / (mx - mn) * 100
                    ).round(0).astype("Int64")
            except Exception as e:
                logging.warning(f"[read_gemini_results] Fundamental normalization failed: {e}")
    except Exception as e:
        logging.warning(f"[read_gemini_results] Failed to read {fund_path.name}: {e}")
        dfs['fundamental'] = None

    try:
        dfs['trend'] = pd.read_excel(trend_path)
    except Exception as e:
        logging.warning(f"[read_gemini_results] Failed to read {trend_path.name}: {e}")
        dfs['trend'] = None

    try:
        dfs['screened'] = pd.read_excel(screened_path, sheet_name="Dual_Threshold_Filter")
    except Exception as e:
        logging.warning(f"[read_gemini_results] Failed to read {screened_path.name}: {e}")
        dfs['screened'] = None

    # 不再对 fundamental/trend 做归一化
    # 只做筛选表的合并与加权
    if dfs.get('screened') is not None and dfs.get('trend') is not None and dfs.get('fundamental') is not None:
        df_screened = dfs['screened']
        df_trend = dfs['trend']
        df_fund = dfs['fundamental']

        # Resolve column names
        def _res_col(df, hint):
            for c in df.columns:
                if c.lower().replace("_", "") == hint.lower().replace("_", ""):
                    return c
            raise KeyError(f"Column '{hint}' not found.")

        col_tic_screened = _res_col(df_screened, "ticker")
        col_tic_trend = _res_col(df_trend, "ticker")
        col_trend_score = _res_col(df_trend, "normalized_trend_score")
        col_tic_fund = _res_col(df_fund, "ticker")
        col_fund_score = _res_col(df_fund, "normalized_overall_score")

        # Merge scores into screened
        df_screened = df_screened.copy()
        df_screened["Ticker_upper"] = df_screened[col_tic_screened].astype(str).str.upper()
        trend_map = dict(zip(df_trend[col_tic_trend].astype(str).str.upper(), df_trend[col_trend_score]))
        fund_map = dict(zip(df_fund[col_tic_fund].astype(str).str.upper(), df_fund[col_fund_score]))

        df_screened["Trend Score"] = df_screened["Ticker_upper"].map(trend_map)
        df_screened["Fundamental Score"] = df_screened["Ticker_upper"].map(fund_map)

        # Calculate weighted sum (e.g., 0.5 * trend + 0.5 * fundamental)
        wt, wf = 0.6, 0.4
        s = wt + wf or 1
        wt, wf = wt/s, wf/s
        df_screened["Combined Score"] = (
            wt * pd.to_numeric(df_screened["Trend Score"], errors="coerce") +
            wf * pd.to_numeric(df_screened["Fundamental Score"], errors="coerce")
        ).round(2)

        # Drop helper column
        df_screened.drop(columns=["Ticker_upper"], inplace=True)
        # Sort by Combined Score descending
        df_screened = df_screened.sort_values(by="Combined Score", ascending=False, na_position="last")
        df_screened.reset_index(drop=True, inplace=True)
        df_screened.index = df_screened.index + 1  # 1-based index
        df_screened = df_screened.loc[:, ["Ticker", "Trend Score", "Fundamental Score", "Combined Score"]]
        # round scores to 0 decimal places and convert to Int64
        df_screened["Trend Score"] = df_screened["Trend Score"].round(0).astype("Int64")
        df_screened["Fundamental Score"] = df_screened["Fundamental Score"].round(0).astype("Int64")
        df_screened["Combined Score"] = df_screened["Combined Score"].round(0).astype("Int64")


    return df_screened

def read_gpt_results():
    """
    读取 GPT 相关结果文件（均在 script_dir / "GPT" / "results" /）:
      - composite_selection.xlsx
      - fundamental_scores.xlsx
      - trend_scores.xlsx
    返回 DataFrame，含 Ticker, Trend Score, Fundamental Score, Combined Score（均已归一化0-100）。
    """
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "GPT" / "results"

    comp_path = results_dir / "composite_selection.xlsx"
    fund_path = results_dir / "fundamental_scores.xlsx"
    trend_path = results_dir / "trend_scores.xlsx"

    try:
        df_comp = pd.read_excel(comp_path)
        df_fund = pd.read_excel(fund_path)
        df_trend = pd.read_excel(trend_path)
    except Exception as e:
        logging.warning(f"[read_gpt_results] Failed to read one or more GPT files: {e}")
        return None

    # Helper to resolve column names
    def _res_col(df, hint):
        for c in df.columns:
            if c.lower().replace("_", "") == hint.lower().replace("_", ""):
                return c
        raise KeyError(f"Column '{hint}' not found.")

    try:
        col_tic_comp = _res_col(df_comp, "ticker")
        col_tic_fund = _res_col(df_fund, "ticker")
        col_tic_trend = _res_col(df_trend, "ticker")
        col_score_fund = _res_col(df_fund, "total_score")
        col_score_trend = _res_col(df_trend, "total_score")
    except Exception as e:
        logging.warning(f"[read_gpt_results] Column resolution failed: {e}")
        return None

    # 归一化分数
    def _normalize(df, col):
        valid = df[col].dropna()
        mn, mx = valid.min(), valid.max()
        if mx == mn:
            return pd.Series([50]*len(df), index=df.index, dtype="Int64")
        normed = ((df[col] - mn) / (mx - mn) * 100).round(0)
        normed[df[col].isna()] = pd.NA
        return normed.astype("Int64")

    df_fund["normalized_fund_score"] = _normalize(df_fund, col_score_fund)
    df_trend["normalized_trend_score"] = _normalize(df_trend, col_score_trend)

    # 构建映射
    fund_map = dict(zip(df_fund[col_tic_fund].astype(str).str.upper(), df_fund["normalized_fund_score"]))
    trend_map = dict(zip(df_trend[col_tic_trend].astype(str).str.upper(), df_trend["normalized_trend_score"]))

    # 合并分数到 composite_selection
    df_comp = df_comp.copy()
    df_comp["Ticker_upper"] = df_comp[col_tic_comp].astype(str).str.upper()
    df_comp["Trend Score"] = df_comp["Ticker_upper"].map(trend_map)
    df_comp["Fundamental Score"] = df_comp["Ticker_upper"].map(fund_map)

    # 计算加权总分（默认0.6:0.4，可根据需要调整）
    wt, wf = 0.6, 0.4
    s = wt + wf or 1
    wt, wf = wt/s, wf/s
    df_comp["Combined Score"] = (
        wt * pd.to_numeric(df_comp["Trend Score"], errors="coerce") +
        wf * pd.to_numeric(df_comp["Fundamental Score"], errors="coerce")
    ).round(0).astype("Int64")

    # 整理输出
    df_comp = df_comp.loc[:, [col_tic_comp, "Trend Score", "Fundamental Score", "Combined Score"]]
    df_comp = df_comp.rename(columns={col_tic_comp: "Ticker"})
    df_comp = df_comp.sort_values(by="Combined Score", ascending=False, na_position="last")
    df_comp.reset_index(drop=True, inplace=True)
    df_comp.index = df_comp.index + 1  # 1-based index

    return df_comp

def export_screened_to_html(model = "gemini") -> Path | None:
    """
    将 Gemini 筛选结果 Excel 输出为美观的 HTML。
    """
    if model.lower() == "gpt":
        df = read_gpt_results()
    elif model.lower() == "gemini":
        df = read_gemini_results()

    if df is None or df.empty:
        logging.error("[to_html] No screened data to export.")
        return None

    # 行样式：综合评分>=80为蓝色，否则绿色
    def highlight_rows(s):
        if s['Combined Score'] < 80:
            return ['background-color: #E2EFD8' for _ in s]
        else:
            return ['background-color: #DEEBF7' for _ in s]

    # 列加粗
    def bold_columns(col):
        if col.name in ['Ticker', 'Combined Score']:
            return ['font-weight: bold' for _ in col]
        return ['font-weight: normal' for _ in col]

    # 格式化
    styled_df = df.style.apply(highlight_rows, axis=1)\
        .apply(bold_columns, axis=0)\
        .format({
            'Trend Score': "{:.0f}",
            'Fundamental Score': "{:.0f}",
            'Combined Score': "{:.0f}"
        })\
        .set_properties(**{'text-align': 'center'})\
        .set_table_styles([{
            'selector': 'th',
            'props': [('font-weight', 'bold'), ('text-align', 'center')]
        }])

    html_table = styled_df.to_html(index=False)

    now = datetime.now()
    yesterday = now - timedelta(days=1)
    if model.lower() == "gemini":
        table_title = f"<div style='text-align: center;'><h2>Gemini Screened Results - {yesterday.strftime('%Y.%m.%d')}  </h2></div>"
    elif model.lower() == "gpt":
        table_title = f"<div style='text-align: center;'><h2>GPT Screened Results - {yesterday.strftime('%Y.%m.%d')}  </h2></div>"
    full_html = f"{table_title}\n<div style='display: table; margin: 0 auto;'>{html_table}</div>"

    script_dir = Path(__file__).resolve().parent
    cfg = configparser.ConfigParser()
    cfg.read(script_dir / "config.ini", encoding="utf-8")
    out_dir = script_dir / cfg["FILES"].get("output_dir", "Output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if model.lower() == "gemini":
        html_path = out_dir / "screened_stocks_Gemini.html"
    elif model.lower() == "gpt":
        html_path = out_dir / "screened_stocks_GPT.html"
    try:
        html_path.write_text(full_html, encoding="utf-8")
        logging.info(f"[to_html] Exported → {html_path}")
        return html_path
    except Exception as e:
        logging.error(f"[to_html] Failed writing HTML: {e}")
        return None



def test_main_process():
    """
    测试主流程执行函数。
    仅在相应管道可用时才调用对应的 pipeline，避免 NameError。
    """
    try:
        # 先测试 Gemini
        cfg_path = Path(__file__).resolve().parent / "config.ini"
        yahoo_downloader.download_price_data(config_file=str(cfg_path))
        yahoo_downloader.acquire_raw_financial_data_to_staging(config_file=str(cfg_path))

        # 先测试 Gemini
        change_working_directory(Gemini_dir)
        gemini_main_pipeline(True)

        # 再测试 GPT（若成功导入）
        if GPT_AVAILABLE:
            change_working_directory(GPT_dir)
            gpt_run_pipeline(
                trend_run_stage=0,
                recalc_scores=True,
                do_selection=True
            )
        else:
            logging.warning("GPT pipeline 未导入成功，跳过 GPT 测试步骤。")

    except Exception as e:
        logging.error(f"主流程执行时发生错误: {e}", exc_info=True)
    finally:
        # 无论成功与否都切回脚本目录
        change_working_directory()


# --- 主执行逻辑 ---
if __name__ == "__main__":
    overall_start_time = time.time()
    run_main_process()  # 调用封装好的主流程函数
    #test_main_process()
    export_screened_to_html("gemini")  # 默认导出 Gemini 结果
    export_screened_to_html("gpt")
    generate_prompt_Gemini()
    generate_prompt_GPT()
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    logging.info(f"--- 脚本总执行时间: {overall_duration:.2f} 秒 ---")
    sys.exit(0) # 脚本正常结束


