from pathlib import Path
from sqlalchemy import create_engine
import GPT.compute_high_growth_score_SP500_GPT as fin
from GPT.Compute_Trend_score_SP500_GPT import Update_DB

PRICE_DB = Path("SP500_price_data.db")
FIN_DB = Path("SP500_finance_data.db")

def download_price_data(db_path: Path = PRICE_DB):
    """Download price history for all S&P 500 stocks into *db_path*."""
    Update_DB(str(db_path))


def download_finance_data(db_path: Path = FIN_DB):
    """Download financial statements for all S&P 500 stocks into *db_path*."""
    # override target DB before calling download_all
    fin.DB_PATH = Path(db_path)
    fin.CFG["database"]["db_name"] = str(db_path)
    fin.engine = create_engine(f"sqlite:///{db_path}")
    fin.download_all()


def main():
    download_price_data()
    download_finance_data()

if __name__ == "__main__":
    main()
