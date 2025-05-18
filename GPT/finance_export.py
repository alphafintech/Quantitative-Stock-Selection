import configparser
import sqlite3
from pathlib import Path
import pandas as pd
import traceback
from typing import Dict

def export_ticker_financials_to_excel(ticker: str,
                                      cfg_path: str = "config_finance.ini",
                                      raw_table: str = "raw_financials",
                                      metrics_table: str = "derived_metrics") -> Path:
    """
    导出指定 Ticker 在数据库中的所有财报数据到 Excel，并在终端打印三张表。
    参数
    ----
    ticker : str
        目标股票代码，大小写不敏感（函数内部统一大写比较）
    cfg_path : str
        config_finance.ini 的路径；默认与脚本同目录
    raw_table / metrics_table : str
        表名，如有自定义可修改

    返回值
    -----
    Path
        生成的 Excel 文件路径
    """
    # 读取 db 路径
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.read(cfg_path, encoding="utf-8")
    db_name = cfg["database"].get("db_name", "sp500_finance.db")
    db_path = Path(cfg_path).parent / db_name

    if not db_path.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    # 连接数据库 —— 动态抓取含 ticker 列的所有表
    sheet_dfs: Dict[str, pd.DataFrame] = {}

    with sqlite3.connect(db_path) as conn:
        # 找到所有用户表
        table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()

        for tbl in table_names:
            # 只处理包含 ticker 字段的表
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
            if not any(c.lower() == "ticker" for c in cols):
                continue

            try:
                df = pd.read_sql(f'SELECT * FROM "{tbl}" WHERE UPPER(ticker)=UPPER(?)', conn, params=(ticker,))
                if not df.empty:
                    sheet_dfs[tbl] = df.sort_values(df.columns[0], ignore_index=True)
            except Exception:
                traceback.print_exc()
                continue

    # 打印到控制台，便于快速查看
    if not sheet_dfs:
        raise ValueError(f"在数据库中未找到 {ticker} 相关数据。")

    for tbl, df in sheet_dfs.items():
        print(f"\n========== {tbl.upper()} =========")
        print(df.head())

    # 将常见日期列转换为字符串
    for df in sheet_dfs.values():
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    # 输出到 Excel，每张表一个 sheet
    out_name = f"{ticker.upper()}_financials.xlsx"
    out_path = Path(__file__).with_name(out_name)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for tbl, df in sheet_dfs.items():
            writer.book.create_sheet(tbl[:31])   # sheet 名最长 31 字符
            df.to_excel(writer, sheet_name=tbl[:31], index=False)

    print(f"[INFO] 已导出到 {out_path.resolve()}")
    return out_path


