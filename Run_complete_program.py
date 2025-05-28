# -*- coding: utf-8 -*-
"""
主执行脚本 (Run_complete_program.py)

功能:
1. 解析命令行参数以控制 S&P 500 处理流程 (Gemini 和/或 GPT 部分)。
2. 调用 Gemini/run_sp500_processing.py 中的 main_pipeline 函数 (如果需要)。
3. 调用 GPT/run_complete_process.py 中的 run_pipeline 函数 (如果需要)。
4. 允许用户选择性地跳过数据更新或筛选步骤。

命令行参数说明:
--skip-update-price-data   跳过 yahoo_downloader.download_price_data()
--skip-update-finance-data 跳过 yahoo_downloader.acquire_raw_financial_data_to_staging()
--skip-Gemini-pipeline     跳过 Gemini 处理流程
--skip-GPT-pipeline        跳过 GPT 处理流程

命令行用法示例:
  python Run_complete_program.py                       # 默认运行全部流程
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

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [MainRunner] %(message)s')
Gemini_dir = "Gemini"
GPT_dir = "GPT"
def change_working_directory(next_directory = ""):
    """
    更改当前工作目录为脚本所在目录，或其下的 next_directory。
    """
    # 获取当前脚本文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取脚本文件所在的目录
    script_dir = os.path.dirname(script_path)
    # 切换当前工作目录到脚本所在的目录或其下的 next_directory
    if next_directory == "":
        os.chdir(script_dir)
        logging.info(f"当前工作目录已更改为: {script_dir}")
    else:
        next_absolute_dir = os.path.join(script_dir, next_directory)
        os.chdir(next_absolute_dir)
        logging.info(f"当前工作目录已更改为: {next_absolute_dir}")
# --- 导入 Gemini 处理流程 ---
GEMINI_AVAILABLE = False # 默认不可用
#change_working_directory(Gemini_dir) # 切换到 Gemini 子目录
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
            cfg_tmp.read(cfg_path)
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
    cfg.read(cfg_path)

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
    cfg.read(cfg_path)

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
        "Gemini/results/sp500_fundamental_scores.xlsx",
    )
    path = Path(token)
    if not path.is_absolute():
        if path.parts and path.parts[0] == "Gemini":
            path = script_dir / path
        else:
            path = script_dir / "Gemini" / "results" / path

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
def export_gemini_screened_to_html() -> Path | None:
    """
    Convert the Excel specified by
    [FILES] screened_stocks_file_Gemini
    (default: Gemini/results/screened_stocks.xlsx)
    into a standalone HTML file under Output/.

    Returns
    -------
    Path | None
        Path to the generated HTML, or None if failed.
    """
    script_dir = Path(__file__).resolve().parent
    cfg_path   = script_dir / "config.ini"
    if not cfg_path.exists():
        logging.error("[to_html] config.ini not found.")
        return None

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    try:
        excel_token = cfg["FILES"]["screened_stocks_file_Gemini"]
    except KeyError:
        logging.error("[to_html] screened_stocks_file_Gemini not configured.")
        return None

    excel_path = Path(excel_token)
    if not excel_path.is_absolute():
        # replicate resolve logic from generate_prompt_Gemini
        if excel_path.parts and excel_path.parts[0] == "Gemini":
            excel_path = script_dir / excel_path
        else:
            excel_path = script_dir / "Gemini" / "results" / excel_path

    if not excel_path.exists():
        logging.error(f"[to_html] Excel not found: {excel_path}")
        return None

    # read all sheets
    try:
        sheets = pd.read_excel(excel_path, sheet_name=None)
    except Exception as e:
        logging.error(f"[to_html] Failed reading Excel: {e}")
        return None

    # ---- Dynamic row-split for Dual_Threshold_Filter colouring ----
    # (dual_half calculation is now unused, but retained for legacy; safe to remove if not used elsewhere)

    # --- Load normalized fundamental scores ---
    norm_map = _get_normalized_fund_scores(cfg, script_dir)

    # --- Weights for combined score ---
    if cfg.has_section("WEIGHTS"):
        w_trend = float(cfg["WEIGHTS"].get("trend_weight", "0.6"))
        w_fund  = float(cfg["WEIGHTS"].get("fund_weight",  "0.4"))
    else:
        logging.debug("[to_html] WEIGHTS section missing; defaulting to 0.5 / 0.5")
        w_trend, w_fund = 0.6, 0.4
    if (w_trend + w_fund) == 0:
        w_trend = w_fund = 0.5

    # build simple HTML page
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Gemini Screened Stocks</title>",
        f"<style>"
        "body{font-family:Arial,Helvetica,sans-serif;padding:20px;}"
        "h2{margin-top:40px;}"
        "table{border-collapse:collapse;table-layout:fixed;width:auto;margin:0 auto;border:none;}"
        "td,th{padding:2px 4px;text-align:center;"
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;width:90px;max-width:90px;}"
        "td:nth-child(1),th:nth-child(1){width:50px;max-width:50px;}"
        "th{text-align:center;font-weight:bold;border-bottom:2px solid #97d197;background:none;}"
        "tbody tr:nth-child(-n+10) td{background:#eaf4ff;}"   # light blue for top 10
        "tbody tr:nth-child(n+11)  td{background:#e8ffea;}"   # light green for others
        "td:nth-child(2),td:nth-child(5){font-weight:bold;}"
        "</style></head><body>",
        "<h1>Gemini Screened Stocks</h1>"
    ]
    for sheet_name, df in sheets.items():
        # Skip Combined_Weighted_Score sheet as per latest requirement
        if sheet_name.strip().lower() == "combined_weighted_score":
            continue
        # --- For Combined_Weighted_Score: keep top‑50 by 综合得分 ---
        if sheet_name.strip().lower() == "combined_weighted_score":
            # identify combined_score column case‑insensitively
            try:
                col_cb = next(c for c in df.columns
                              if c.lower().replace("_", "") == "combinedscore")
            except StopIteration:
                logging.warning("[to_html] Combined_Weighted_Score sheet missing Combined_Score col.")
            else:
                df = (df.sort_values(by=col_cb, ascending=False, na_position="last")
                        .head(50))

        # --- Update Dual_Threshold_Filter with normalized fundamental scores ---
        if sheet_name.strip().lower() == "dual_threshold_filter":
            def _res(df, hint):
                for c in df.columns:
                    if c.lower().replace("_", "") == hint.lower().replace("_", ""):
                        return c
                return None  # return None if not found

            col_tic = _res(df, "ticker")
            col_fu  = _res(df, "fundamental_score")
            col_tr  = _res(df, "trend_score")
            col_cb  = _res(df, "combined_score") or "Combined_Score"

            if col_tic is None or col_fu is None or col_tr is None:
                logging.warning("[to_html] Dual_Threshold_Filter sheet missing essential columns.")
            else:
                # ensure Fundamental_Score column exists
                if col_fu not in df.columns:
                    df[col_fu] = np.nan

                df.loc[:, col_fu] = (
                    df[col_tic].astype(str).str.upper().map(norm_map).fillna(df[col_fu])
                )

                # compute Combined_Score column even if it did not exist
                df.loc[:, col_cb] = (
                    w_trend * pd.to_numeric(df[col_tr], errors="coerce") +
                    w_fund  * pd.to_numeric(df[col_fu], errors="coerce")
                ).round(0)
            # --- sort Dual_Threshold_Filter by combined score descending ---
            if sheet_name.strip().lower() == "dual_threshold_filter":
                try:
                    _col_cb = next(c for c in df.columns
                                   if c.lower().replace("_", "") == "combinedscore")
                    df = df.sort_values(by=_col_cb, ascending=False, na_position="last")
                except StopIteration:
                    logging.warning("[to_html] Dual_Threshold_Filter sheet missing Combined_Score col for sorting.")

        # --- inject normalized fundamental score & recompute combined ---
        # (skipped for combined_weighted_score due to early continue)
        # --- Rename selected columns to Chinese ---
        rename_map = {
            "Ticker": "股票代码",
            "Trend_Score": "趋势得分",
            "Fundamental_Score": "基本面得分",
            "Combined_Score": "综合得分",
        }
        df = df.rename(columns=rename_map)

        # --- Add row number column ---
        df.insert(0, "序号", range(1, len(df) + 1))
        # --- Drop percentile columns if present ---
        drop_cols = [c for c in df.columns
                     if c.lower() in ("trend_percentile", "fundamental_percentile")]
        df = df.drop(columns=drop_cols, errors="ignore")

        # --- Round numeric columns to nearest integer for display ---
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].round(0).astype("Int64")

        html_parts.append(f"<h2>{html.escape(sheet_name)}</h2>")
        html_parts.append(df.to_html(index=False, border=0, classes="data-table"))
    html_parts.append("</body></html>")
    full_html = "\n".join(html_parts)

    out_dir = script_dir / cfg["FILES"].get("output_dir", "Output")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "screened_stocks_Gemini.html"
    try:
        html_path.write_text(full_html, encoding="utf-8")
        logging.info(f"[to_html] Exported → {html_path}")
        return html_path
    except Exception as e:
        logging.error(f"[to_html] Failed writing HTML: {e}")
        return None

# ------------------------------------------------------------
# Export GPT screened_stocks (composite_selection.xlsx) to HTML
# ------------------------------------------------------------
def export_gpt_screened_to_html() -> Path | None:
    """
    Convert GPT composite_selection.xlsx to HTML (style identical to Gemini).
    trend_score & fund_score are normalized via GPT/results/trend_scores.xlsx
    and GPT/results/fundamental_scores.xlsx. growth_sub column is omitted.
    """
    sd = Path(__file__).resolve().parent
    cfg = configparser.ConfigParser()
    cfg.read(sd / "config.ini")

    comp_token = cfg["FILES"].get("screened_stocks_file_GPT",
                                  "GPT/results/composite_selection.xlsx")
    comp_path = Path(comp_token) if Path(comp_token).is_absolute() else sd / comp_token
    if not comp_path.exists():
        logging.error(f"[to_html] GPT composite Excel missing: {comp_path}")
        return None

    try:
        df = pd.read_excel(comp_path)
    except Exception as e:
        logging.error(f"[to_html] Failed reading composite Excel: {e}")
        return None

    def _res(df: pd.DataFrame, hint: str):
        for c in df.columns:
            if c.lower().replace("_", "") == hint.lower().replace("_", ""):
                return c
        raise KeyError

    try:
        col_tic = _res(df, "ticker")
        col_tr  = _res(df, "trend_score")
        col_fu  = _res(df, "fund_score")
        col_cb  = _res(df, "final_score")
    except KeyError as ke:
        logging.error(f"[to_html] composite Excel missing column: {ke}")
        return None

    # Normalise using source files' original score columns
    trend_map = _build_norm_map(sd / "GPT/results/trend_scores.xlsx", "TotalScore")
    fund_map  = _build_norm_map(sd / "GPT/results/fundamental_scores.xlsx", "total_score")

    df.loc[:, col_tr] = df[col_tic].astype(str).str.upper().map(trend_map).fillna(df[col_tr])
    df.loc[:, col_fu] = df[col_tic].astype(str).str.upper().map(fund_map).fillna(df[col_fu])

    # Recompute combined score
    if cfg.has_section("WEIGHTS"):
        wt = float(cfg["WEIGHTS"].get("trend_weight", "0.6"))
        wf = float(cfg["WEIGHTS"].get("fund_weight",  "0.4"))
    else:
        wt, wf = 0.6, 0.4
    s = wt + wf or 1
    wt, wf = wt/s, wf/s
    df.loc[:, col_cb] = (
        wt * pd.to_numeric(df[col_tr], errors="coerce") +
        wf * pd.to_numeric(df[col_fu], errors="coerce")
    ).round(0)

    # Drop growth_sub if present
    drop_cols = [c for c in df.columns if c.lower().replace("_","") == "growthsub"]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Sort & top‑50
    df = df.sort_values(by=col_cb, ascending=False, na_position="last").head(50)

    # Rename & add row number
    rename = {"ticker":"股票代码", "trend_score":"趋势得分",
              "fund_score":"基本面得分", "final_score":"综合得分"}
    df = df.rename(columns=rename)
    df.insert(0, "序号", range(1, len(df)+1))
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].round(0).astype("Int64")

    # Build HTML
    style = (
        "body{font-family:Arial,Helvetica,sans-serif;padding:20px;}"
        "table{border-collapse:collapse;table-layout:fixed;width:auto;margin:0 auto;border:none;}"
        "td,th{padding:2px 4px;text-align:center;"
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;width:90px;max-width:90px;}"
        "td:nth-child(1),th:nth-child(1){width:50px;max-width:50px;}"
        "th{text-align:center;font-weight:bold;border-bottom:2px solid #97d197;background:none;}"
        "tbody tr:nth-child(-n+10) td{background:#eaf4ff;}"
        "tbody tr:nth-child(n+11)  td{background:#e8ffea;}"
        "td:nth-child(2),td:nth-child(5){font-weight:bold;}"
    )
    html_str = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>GPT Screened Stocks</title>"
        f"<style>{style}</style></head><body>"
        "<h1>GPT Screened Stocks</h1>"
        f"{df.to_html(index=False, border=0)}"
        "</body></html>"
    )
    out_dir = sd / cfg["FILES"].get("output_dir", "Output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "screened_stocks_GPT.html"
    try:
        out_path.write_text(html_str, encoding="utf-8")
        logging.info(f"[to_html] GPT Exported → {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"[to_html] Failed writing GPT HTML: {e}")
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
    export_gemini_screened_to_html()
    export_gpt_screened_to_html()
    generate_prompt_Gemini()
    generate_prompt_GPT()
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    logging.info(f"--- 脚本总执行时间: {overall_duration:.2f} 秒 ---")
    sys.exit(0) # 脚本正常结束
