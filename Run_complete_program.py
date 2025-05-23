# -*- coding: utf-8 -*-
"""
主执行脚本 (Run_complete_program.py)

功能:
1. 解析命令行参数以控制 S&P 500 处理流程 (Gemini 和/或 GPT 部分)。
2. 调用 Gemini/run_sp500_processing.py 中的 main_pipeline 函数 (如果需要)。
3. 调用 GPT/run_complete_process.py 中的 run_pipeline 函数 (如果需要)。
4. 允许用户选择性地跳过数据更新或筛选步骤。

命令行用法示例:
# --- Gemini 流程 ---
- 运行完整 Gemini 流程:
  python Run_complete_program.py

- 跳过 Gemini 最终筛选:
  python Run_complete_program.py --skip-Gemini-screening
- 完全跳过 Gemini 流程:
  python Run_complete_program.py --skip-Gemini

# --- GPT 流程 ---
- 运行完整 GPT 流程 (默认执行):
  python Run_complete_program.py
- 跳过 GPT 流程:
  python Run_complete_program.py --skip-GPT

# --- 数据下载 ---
- 更新股价和财务数据后再运行:
  python Run_complete_program.py --update-price-data --update-finance-data
- 仅更新股价数据:
  python Run_complete_program.py --update-price-data
# --- 同时运行 (示例: 跳过 Gemini) ---
  python Run_complete_program.py --skip-Gemini
"""

import argparse
import logging
import sys
import os
import time
import traceback
import pandas as pd
from pathlib import Path
import configparser
import datetime as dt
import yahoo_downloader

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [MainRunner] %(message)s')
Gemini_dir = "Gemini"
GPT_dir = "GPT"
def change_working_directory(next_directory = ""):
    """
    更改当前工作目录为脚本所在目录。
    """
        # 获取当前脚本文件的绝对路径
    # __file__ 是一个内置变量，包含了当前脚本的文件名（可能是相对路径或绝对路径）
    # os.path.abspath() 将其转换为绝对路径
    script_path = os.path.abspath(__file__)

    # 获取脚本文件所在的目录
    # os.path.dirname() 返回路径中的目录部分
    script_dir = os.path.dirname(script_path)

    # 切换当前工作目录到脚本所在的目录
    # os.chdir() 相当于终端的 cd 命令
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

    # Gemini 控制参数
    parser.add_argument(
        '--update-price-data',
        action='store_true',
        help='在运行管道前下载/更新股价数据'
    )
    parser.add_argument(
        '--update-finance-data',
        action='store_true',
        help='在运行管道前下载/更新财务数据'
    )
    # Gemini 控制参数
    gemini_group = parser.add_argument_group('Gemini Pipeline Control (默认执行, 除非 --skip-Gemini)')
    gemini_group.add_argument(
        '--skip-Gemini-screening',
        action='store_true',
        help="跳过 Gemini 流程的最终筛选步骤。"
    )
    gemini_group.add_argument(
        '--skip-Gemini',
        action='store_true',
        help="完全跳过 Gemini 流程的执行。"
    )

    # GPT 控制参数
    gpt_group = parser.add_argument_group('GPT Pipeline Control')
    gpt_group.add_argument(
        '--skip-GPT',
        action='store_true',
        help="完全跳过 GPT 流程（默认执行）。"
    )


    args = parser.parse_args()
    if args.update_price_data:
        logging.info("--- 下载/更新股价数据 ---")
        yahoo_downloader.download_price_data()

    if args.update_finance_data:
        logging.info("--- 下载/更新财务数据 ---")
        yahoo_downloader.download_financial_data()


    # --- 执行 Gemini 流程 (如果可用且未被跳过) ---
    if GEMINI_AVAILABLE and not args.skip_Gemini:
        
        run_gemini_screening = not args.skip_Gemini_screening

        logging.info("--- 开始执行主流程 (Gemini) ---")
        logging.info(
            f"Gemini 参数: run_final_screening={run_gemini_screening}"
        )

        start_time_gemini = time.time()
        try:
            change_working_directory(Gemini_dir)
            gemini_main_pipeline(
                run_final_screening=run_gemini_screening
            )
            logging.info("--- 主流程 (Gemini) 执行完毕 ---")
        except TypeError as te:
            logging.error(f"调用 gemini_main_pipeline 时发生 TypeError: {te}", exc_info=True)
        except Exception as e:
            logging.error(f"执行 gemini_main_pipeline 时发生意外错误: {e}", exc_info=True)
        finally:
            end_time_gemini = time.time()
            duration_gemini = end_time_gemini - start_time_gemini
            logging.info(f"--- Gemini 流程执行时间: {duration_gemini:.2f} 秒 ---")
    elif args.skip_Gemini:
         logging.info("--- 跳过 Gemini 流程 (根据 --skip-Gemini 参数) ---")
    elif not GEMINI_AVAILABLE:
         logging.warning("--- Gemini 流程不可用 (导入失败) ---")


    # --- 执行 GPT 流程 (如果可用且未被跳过) ---
    if GPT_AVAILABLE and not args.skip_GPT:
        recalc_scores_gpt = True  # 固定为 True
        do_selection_gpt = True  # 固定为 True

        logging.info("--- 开始执行主流程 (GPT) ---")
        logging.info(
            f"GPT 参数: update_finance_db=False, recalc_scores={recalc_scores_gpt}, do_selection={do_selection_gpt}"
        )
        start_time_gpt = time.time()
        try:
            change_working_directory(GPT_dir)
            gpt_run_pipeline(
                update_finance_db=False,
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
    elif args.skip_GPT and not GPT_AVAILABLE:
         logging.warning("--- GPT 流程不可用 (导入失败)，但已跳过 GPT ---")
    elif not GPT_AVAILABLE:
         logging.warning("--- GPT 流程不可用 (导入失败) ---")


        

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
    try:
        holdings_token = cfg["FILES"].get("current_holdings_file_Gemini", "current_holdings.txt")
    except KeyError:
        holdings_token = "current_holdings.txt"

    holdings_path = Path(holdings_token)
    if not holdings_path.is_absolute():
        holdings_path = script_dir / holdings_path
    if holdings_path.exists():
        holdings_list = holdings_path.read_text(encoding="utf-8").strip()
    else:
        logging.warning(f"持仓文件不存在，使用占位 'N/A': {holdings_path}")
        holdings_list = "N/A"

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
    holdings_token = cfg["FILES"].get("current_holdings_file_GPT", "current_holdings.txt")
    holdings_path  = Path(holdings_token)
    if not holdings_path.is_absolute():
        holdings_path = script_dir / holdings_path
    holdings_list = holdings_path.read_text(encoding="utf-8").strip() if holdings_path.exists() else "N/A"

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

def test_main_process():
    """
    测试主流程执行函数。
    仅在相应管道可用时才调用对应的 pipeline，避免 NameError。
    """
    try:
        # 先测试 Gemini
        #yahoo_downloader.download_price_data()
        #yahoo_downloader.acquire_raw_financial_data_to_staging()
        #yahoo_downloader.process_staged_data_to_final_db()

        # 先测试 Gemini
        #change_working_directory(Gemini_dir)
        #gemini_main_pipeline(True)

        # 再测试 GPT（若成功导入）
        if GPT_AVAILABLE:
            change_working_directory(GPT_dir)
            gpt_run_pipeline(
                trend_run_stage=0,
                update_finance_db=False,
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
    #run_main_process() # 调用封装好的主流程函数
    test_main_process() # 调用测试函数
    generate_prompt_Gemini()
    generate_prompt_GPT()
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    logging.info(f"--- 脚本总执行时间: {overall_duration:.2f} 秒 ---")
    sys.exit(0) # 脚本正常结束
