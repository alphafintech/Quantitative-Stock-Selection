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
    df["S&P500累计收益率"] = (df["S&P500"] / base["S&P500"]) * 100 - 100
    df["ChatGPT累计收益率"] = (df["ChatGPT"] / base["ChatGPT"]) * 100 - 100
    df["Gemini累计收益率"] = (df["Gemini"] / base["Gemini"]) * 100 - 100
    df[["S&P500累计收益率", "ChatGPT累计收益率", "Gemini累计收益率"]] = (
        df[["S&P500累计收益率", "ChatGPT累计收益率", "Gemini累计收益率"]].astype(int)
    )
    df["Date"] = df["Date"].dt.date  # 去掉时间部分

    # ------------------------ 保存 Excel -------------------------
    if output_excel is None:
        output_excel = RESULT_DIR / Path(excel_path).name
    df.to_excel(str(output_excel), sheet_name=sheet_name, index=False)

    # -------------------------- 绘图 -----------------------------
    plt.style.use("seaborn-v0_8-whitegrid")  # 更专业的背景
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制三条曲线，增加平滑线条和阴影
    ax.plot(df["Date"], df["Gemini累计收益率"], label="Gemini",
            marker="o", color="#90EE90", linewidth=3, alpha=0.95)
    ax.plot(df["Date"], df["ChatGPT累计收益率"], label="ChatGPT",
            marker="s", color="#FF9999", linewidth=3, alpha=0.95, linestyle=":")
    ax.plot(df["Date"], df["S&P500累计收益率"], label="S&P500",
            marker="*", color="#4B9CD3", linewidth=3, linestyle="--", alpha=0.95)


    # 填充曲线下方区域，增加视觉层次
    ax.fill_between(df["Date"], df["S&P500累计收益率"], alpha=0.08, color="#4B9CD3")
    ax.fill_between(df["Date"], df["ChatGPT累计收益率"], alpha=0.08, color="#FF9999")
    ax.fill_between(df["Date"], df["Gemini累计收益率"], alpha=0.08, color="#90EE90")

    # 设置标题和标签
    ax.set_title("Cumulative Return Comparison", fontsize=22, fontweight="bold", pad=18)
    ax.set_ylabel("Cumulative Return (%)", fontsize=18)
    ax.set_xlabel("Date", fontsize=18)
    ax.set_xticks(df["Date"])
    ax.set_xticklabels(df["Date"], rotation=45, fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

    # 优化图例（去除阴影）
    ax.legend(fontsize=17, loc="upper left", frameon=True, fancybox=True, borderpad=1)

    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=2.0)

    if output_png is None:
        output_png = RESULT_DIR / "performance_comparison.png"
    plt.savefig(str(output_png), dpi=180, bbox_inches="tight")
    plt.close()

    return df


# ----------------------- 独立运行测试 ----------------------------
if __name__ == "__main__":
    compute_and_plot_yield()