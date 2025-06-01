import sqlite3
from pathlib import Path
import configparser
from importlib import util
import sys

# Ensure repo root is on module search path *before* importing project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yahoo_downloader import export_finance_data_to_excel


if __name__ == "__main__":
    export_ticker = "NVDA"  # Example ticker for export

    # Export to the same folder as this script
    script_dir = Path(__file__).parent
    output_path = script_dir / f"{export_ticker}_raw_finance.xlsx"

    export_finance_data_to_excel(export_ticker, output_path=output_path)
    print(f"Exported to: {output_path}")