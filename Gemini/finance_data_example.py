import sqlite3
import pandas as pd
import logging
import os
import configparser

# 尝试导入 openpyxl，用于写入 Excel
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logging.warning("未找到 'openpyxl' 库。如果需要保存到 Excel，请运行: pip install openpyxl")

# 配置基本的日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [GetData] %(message)s')


def _get_finance_db(cfg_path: str = "config.ini") -> str:
    cfg = configparser.ConfigParser()
    if os.path.exists(cfg_path):
        cfg.read(cfg_path)
    return cfg.get("database", "finance_db", fallback="SP500_finance_data.db")

def get_stock_financials(ticker: str,
                         db_path: str = _get_finance_db(),
                         save_to_excel: bool = False):
    """
    从指定的 SQLite 数据库获取单个股票的年度和季度财务数据，
    并可选择将数据保存到与脚本同目录的 Excel 文件中。

    Args:
        ticker (str): 要查询的股票代码 (例如 'AAPL', 'MSFT').
        db_path (str): SQLite 数据库文件的路径。
                       默认为 'S&P500_finance_data.db'。
        save_to_excel (bool): 如果为 True，则将获取的数据保存到 Excel 文件。
                              默认为 False。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
        包含两个 pandas DataFrame 的元组 (annual_data, quarterly_data)。
        如果发生错误或找不到数据，则返回 (None, None)。
        即使保存 Excel 失败，如果数据获取成功，仍会返回数据。
    """
    conn = None
    annual_data = None
    quarterly_data = None

    # 检查数据库文件是否存在
    if not os.path.exists(db_path):
        logging.error(f"数据库文件未找到: {db_path}")
        return None, None

    try:
        # 创建数据库连接
        conn = sqlite3.connect(db_path)
        logging.info(f"成功连接到数据库: {db_path}")

        # --- 查询年度数据 ---
        try:
            query_annual = f'SELECT * FROM "annual_financials" WHERE ticker = ? ORDER BY period'
            annual_data = pd.read_sql_query(query_annual, conn, params=(ticker,))
            if annual_data.empty:
                logging.warning(f"在 'annual_financials' 表中未找到 Ticker '{ticker}' 的年度数据。")
            else:
                logging.info(f"成功获取 Ticker '{ticker}' 的 {len(annual_data)} 条年度财务记录。")
                annual_data['period'] = pd.to_numeric(annual_data['period'], errors='coerce')
        except pd.io.sql.DatabaseError as e:
             if "no such table: annual_financials" in str(e).lower():
                 logging.error(f"错误：数据库 '{db_path}' 中缺少 'annual_financials' 表。")
             else:
                 logging.error(f"查询年度数据时出错: {e}")
             annual_data = None
        except Exception as e:
            logging.error(f"查询年度数据时发生意外错误: {e}", exc_info=True)
            annual_data = None

        # --- 查询季度数据 ---
        try:
            query_quarterly = f'SELECT * FROM "quarterly_financials" WHERE ticker = ? ORDER BY period'
            quarterly_data = pd.read_sql_query(query_quarterly, conn, params=(ticker,), parse_dates=['period'])
            if quarterly_data.empty:
                logging.warning(f"在 'quarterly_financials' 表中未找到 Ticker '{ticker}' 的季度数据。")
            else:
                logging.info(f"成功获取 Ticker '{ticker}' 的 {len(quarterly_data)} 条季度财务记录。")
        except pd.io.sql.DatabaseError as e:
             if "no such table: quarterly_financials" in str(e).lower():
                 logging.error(f"错误：数据库 '{db_path}' 中缺少 'quarterly_financials' 表。")
             else:
                 logging.error(f"查询季度数据时出错: {e}")
             quarterly_data = None
        except Exception as e:
            logging.error(f"查询季度数据时发生意外错误: {e}", exc_info=True)
            quarterly_data = None

        # --- 保存到 Excel (如果请求并且数据有效) ---
        if save_to_excel and annual_data is not None and quarterly_data is not None:
            if not OPENPYXL_AVAILABLE:
                logging.error("无法保存到 Excel：缺少 'openpyxl' 库。请先安装。")
            else:
                # 确定输出路径 (与脚本同目录)
                try:
                    # __file__ 在直接运行时可用，但在某些环境（如 REPL）中可能不可用
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                except NameError:
                    # 回退到当前工作目录
                    script_dir = os.getcwd()
                    logging.warning("__file__ 变量不可用，Excel 文件将保存在当前工作目录中。")

                excel_filename = f"{ticker}_financials.xlsx"
                output_path = os.path.join(script_dir, excel_filename)

                logging.info(f"尝试将数据保存到 Excel 文件: {output_path}")
                try:
                    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                        # 写入年度数据 (如果非空)
                        if not annual_data.empty:
                            annual_data.to_excel(writer, sheet_name='Annual_Data', index=False)
                            logging.debug(f"已将年度数据写入工作表 'Annual_Data'")
                        else:
                            logging.info("年度数据为空，不写入 Excel。")

                        # 写入季度数据 (如果非空)
                        if not quarterly_data.empty:
                            quarterly_data.to_excel(writer, sheet_name='Quarterly_Data', index=False)
                            logging.debug(f"已将季度数据写入工作表 'Quarterly_Data'")
                        else:
                            logging.info("季度数据为空，不写入 Excel。")
                    logging.info(f"数据成功保存到: {output_path}")
                except PermissionError:
                    logging.error(f"保存到 '{output_path}' 时权限被拒绝。请检查文件是否已打开或文件夹权限。")
                except Exception as e_save:
                    logging.error(f"保存 Excel 文件时发生错误: {e_save}", exc_info=True)
        elif save_to_excel:
             # 如果请求保存但数据获取失败
             logging.warning("请求保存到 Excel，但年度或季度数据获取失败，无法保存。")


        # --- 返回获取的数据 ---
        # 即使保存失败，如果数据获取成功，仍然返回数据
        if annual_data is None or quarterly_data is None:
             return None, None # 返回 None 如果任一查询失败
        else:
             # 返回获取到的 DataFrame (可能是空的)
             return annual_data, quarterly_data

    except sqlite3.Error as e:
        logging.error(f"数据库连接或操作错误: {e}", exc_info=True)
        return None, None
    except Exception as e:
        logging.error(f"获取财务数据时发生未知错误: {e}", exc_info=True)
        return None, None
    finally:
        if conn:
            conn.close()
            logging.info("数据库连接已关闭。")


# --- 示例用法 ---
if __name__ == "__main__":
    db_file = _get_finance_db()
    example_ticker = 'NVDA' # 换一个 Ticker 示例

    # --- 示例 1: 只获取数据，不保存 ---
    print(f"\n[示例 1] 正在尝试获取 Ticker '{example_ticker}' 的财务数据 (不保存)...")
    annual_df_1, quarterly_df_1 = get_stock_financials(example_ticker, db_file, save_to_excel=False)
    if annual_df_1 is not None:
        print(f"获取到 {len(annual_df_1)} 条年度记录。")
    if quarterly_df_1 is not None:
        print(f"获取到 {len(quarterly_df_1)} 条季度记录。")

    # --- 示例 2: 获取数据并保存到 Excel ---
    print(f"\n[示例 2] 正在尝试获取 Ticker '{example_ticker}' 的财务数据并保存到 Excel...")
    annual_df_2, quarterly_df_2 = get_stock_financials(example_ticker, db_file, save_to_excel=True)

    if annual_df_2 is not None and quarterly_df_2 is not None:
        print(f"\n--- Ticker '{example_ticker}' 年度数据 (前5行) ---")
        if not annual_df_2.empty:
            print(annual_df_2.head())
        else:
            print("无年度数据。")

        print(f"\n--- Ticker '{example_ticker}' 季度数据 (前5行) ---")
        if not quarterly_df_2.empty:
            print(quarterly_df_2.head())
        else:
            print("无季度数据。")
        print(f"\n请检查运行目录下是否生成了 '{example_ticker}_financials.xlsx' 文件。")
    else:
        print(f"\n无法获取 Ticker '{example_ticker}' 的财务数据。")

