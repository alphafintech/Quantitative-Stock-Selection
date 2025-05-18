from pathlib import Path
from GPT.Compute_Trend_score_SP500_GPT import Update_DB
from Gemini import Compute_growth_score_sp500 as gem
ROOT = Path(__file__).resolve().parent
PRICE_DB = ROOT / "SP500_price_data.db"
FIN_DB = ROOT / "SP500_finance_data.db"

def download_price_data(db_path: Path = PRICE_DB):
    """Download price history for all S&P 500 stocks into *db_path*."""
    Update_DB(str(db_path))


def download_finance_data(db_path: Path = FIN_DB):
    """Download financial statements for all S&P 500 stocks into *db_path* using
    the Gemini finance module."""

    config = gem.load_config()
    data_cfg = config.setdefault("Data", {})
    data_cfg["db_name"] = str(db_path)

    conn = gem.create_db_connection(str(db_path))
    if not conn:
        raise RuntimeError("Failed to connect to finance database")
    gem.create_tables(conn)

    url = config.get("General", {}).get("sp500_list_url")
    tickers = list(gem.get_sp500_tickers_and_industries(url).keys())
    for ticker in tickers:
        gem.download_data_for_ticker(ticker, config, conn)

    conn.close()


def main():
    download_price_data()
    download_finance_data()

if __name__ == "__main__":
    main()
