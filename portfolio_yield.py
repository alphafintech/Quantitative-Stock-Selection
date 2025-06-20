#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# 路径设置
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent           # 项目根目录
DEFAULT_EXCEL_PATH = ROOT_DIR / "Portfolio" / "portfolio_performance.xlsx"

RESULT_DIR = ROOT_DIR / "result_output"                      # 结果输出目录
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def compute_and_plot_yield(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_name: str = "Sheet1",
    output_excel: str | Path | None = None,
    output_png: str | Path | None = None,
) -> pd.DataFrame:
    """
    读取投资组合表现数据，计算三条累计收益率曲线，保存更新后的 Excel 和对比图，
    并返回 DataFrame。

    Parameters
    ----------
    excel_path : str | Path
        源 Excel 文件路径，默认读取项目根目录下 Portfolio 文件夹中的
        portfolio_performance.xlsx。
    sheet_name : str
        需要读取的数据表名称。
    output_excel : str | Path | None
        更新后的文件保存路径，默认写入 result_output 目录。
    output_png : str | Path | None
        绘图 PNG 保存路径，默认写入 result_output/performance_comparison.png。

    Returns
    -------
    pd.DataFrame
        含新增累计收益率列的 DataFrame。
    """
    # -------------------------- 读取 -----------------------------
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # --------------------- 计算累计收益率 ------------------------
    base = df.iloc[0][["S&P500", "ChatGPT", "Gemini"]]
    df["S&P500累计收益率"] = (df["S&P500"] / base["S&P500"]) * 100
    df["ChatGPT累计收益率"] = (df["ChatGPT"] / base["ChatGPT"]) * 100
    df["Gemini累计收益率"] = (df["Gemini"] / base["Gemini"]) * 100
    df[["S&P500累计收益率", "ChatGPT累计收益率", "Gemini累计收益率"]] = (
        df[["S&P500累计收益率", "ChatGPT累计收益率", "Gemini累计收益率"]].astype(int)
    )
    df["Date"] = df["Date"].dt.date  # 去掉时间部分

    # ------------------------ 保存 Excel -------------------------
    if output_excel is None:
        output_excel = RESULT_DIR / Path(excel_path).name
    df.to_excel(str(output_excel), sheet_name=sheet_name, index=False)

    # -------------------------- 绘图 -----------------------------
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10, 6))

    plt.plot(df["Date"], df["S&P500累计收益率"], label="S&P500",
             marker="o", color="#4B9CD3", linewidth=3, linestyle="--")
    plt.plot(df["Date"], df["ChatGPT累计收益率"], label="ChatGPT",
             marker="o", color="#90EE90", linewidth=3)
    plt.plot(df["Date"], df["Gemini累计收益率"], label="Gemini",
             marker="o", color="#FF9999", linewidth=3)

    plt.ylabel("Cumulative Return (%)", fontsize=18)
    plt.xticks(df["Date"], rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()

    if output_png is None:
        output_png = RESULT_DIR / "performance_comparison.png"
    plt.savefig(str(output_png))
    plt.close()

    return df


# ----------------------- 独立运行测试 ----------------------------
if __name__ == "__main__":
    compute_and_plot_yield()