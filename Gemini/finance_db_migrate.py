import sqlite3
import os
import configparser
from pathlib import Path
import logging
import pandas as pd

CREATE_ANNUAL = """
CREATE TABLE IF NOT EXISTS annual_financials (
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,
    revenue REAL, net_income REAL, eps REAL, op_income REAL,
    equity REAL, total_debt REAL, ocf REAL, capex REAL,
    ebit REAL, interest_exp REAL,
    PRIMARY KEY (ticker, period)
);
"""

CREATE_QUARTERLY = """
CREATE TABLE IF NOT EXISTS quarterly_financials (
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,
    revenue REAL, net_income REAL, eps REAL, op_income REAL,
    equity REAL, total_debt REAL, ocf REAL, capex REAL,
    ebit REAL, interest_exp REAL,
    PRIMARY KEY (ticker, period)
);
"""

INSERT_ANNUAL = """
INSERT OR REPLACE INTO annual_financials
SELECT
    ticker,
    strftime('%Y', report_date) AS period,
    CAST(total_revenue AS REAL)            AS revenue,
    CAST(net_income AS REAL)               AS net_income,
    CAST(eps AS REAL)                      AS eps,
    CAST(operating_income AS REAL)         AS op_income,
    CAST(total_assets - total_liabilities AS REAL) AS equity,
    CAST(total_debt AS REAL)               AS total_debt,
    CAST(operating_cash_flow AS REAL)      AS ocf,
    CAST(capital_expenditures AS REAL)     AS capex,
    CAST(operating_income AS REAL)         AS ebit,
    CAST(interest_expense AS REAL)         AS interest_exp
FROM raw_financials
WHERE period='A';
"""

INSERT_QUARTERLY = """
INSERT OR REPLACE INTO quarterly_financials
SELECT
    ticker,
    report_date AS period,
    CAST(total_revenue AS REAL)            AS revenue,
    CAST(net_income AS REAL)               AS net_income,
    CAST(eps AS REAL)                      AS eps,
    CAST(operating_income AS REAL)         AS op_income,
    CAST(total_assets - total_liabilities AS REAL) AS equity,
    CAST(total_debt AS REAL)               AS total_debt,
    CAST(operating_cash_flow AS REAL)      AS ocf,
    CAST(capital_expenditures AS REAL)     AS capex,
    CAST(operating_income AS REAL)         AS ebit,
    CAST(interest_expense AS REAL)         AS interest_exp
FROM raw_financials
WHERE period='Q';
"""

def _get_finance_db(cfg_path: str = "config.ini") -> str:
    parser = configparser.ConfigParser()
    if os.path.exists(cfg_path):
        parser.read(cfg_path)
    return parser.get("database", "finance_db", fallback="SP500_finance_data.db")

def migrate_connection(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_financials'")
    if cur.fetchone() is None:
        return False

    cur.execute(CREATE_ANNUAL)
    cur.execute(CREATE_QUARTERLY)

    df = pd.read_sql_query(
        "SELECT * FROM raw_financials", conn, parse_dates=["report_date"]
    )
    if df.empty:
        return False

    rename_map = {
        "Total Revenue": "revenue",
        "TotalRevenue": "revenue",
        "Net Income Common Stockholders": "net_income",
        "Net Income": "net_income",
        "NetIncome": "net_income",
        "Diluted EPS": "eps",
        "DilutedEPS": "eps",
        "Operating Income": "op_income",
        "OperatingIncome": "op_income",
        "Total Operating Income As Reported": "op_income",
        "EBIT": "ebit",
        "Interest Expense": "interest_exp",
        "InterestExpense": "interest_exp",
        "Stockholders Equity": "equity",
        "Total Stockholder Equity": "equity",
        "ShareholderEquity": "equity",
        "TotalEquityGrossMinorityInterest": "equity",
        "Total Debt": "total_debt",
        "Operating Cash Flow": "ocf",
        "OperatingCashFlow": "ocf",
        "CashFlowFromContinuingOperatingActivities": "ocf",
        "Capital Expenditure": "capex",
        "CapitalExpenditures": "capex",
    }

    df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].unique()
        logging.warning(
            "Duplicate columns after rename: %s. Keeping first.", dups.tolist()
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "revenue" not in df.columns and "total_revenue" in df.columns:
        df["revenue"] = df["total_revenue"]
    if "op_income" not in df.columns and "operating_income" in df.columns:
        df["op_income"] = df["operating_income"]
    if "ocf" not in df.columns and "operating_cash_flow" in df.columns:
        df["ocf"] = df["operating_cash_flow"]
    if "capex" not in df.columns and "capital_expenditures" in df.columns:
        df["capex"] = df["capital_expenditures"]
    if "interest_exp" not in df.columns and "interest_expense" in df.columns:
        df["interest_exp"] = df["interest_expense"]
    if "equity" not in df.columns and {
        "total_assets",
        "total_liabilities",
    }.issubset(df.columns):
        df["equity"] = pd.to_numeric(df["total_assets"], errors="coerce") - pd.to_numeric(
            df["total_liabilities"], errors="coerce"
        )
    if "ebit" not in df.columns:
        if "op_income" in df.columns:
            df["ebit"] = df["op_income"]
        elif "operating_income" in df.columns:
            df["ebit"] = df["operating_income"]

    required_cols = [
        "revenue",
        "net_income",
        "eps",
        "op_income",
        "equity",
        "total_debt",
        "ocf",
        "capex",
        "ebit",
        "interest_exp",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    annual = df[df["period"] == "A"].copy()
    quarterly = df[df["period"] == "Q"].copy()

    annual["period"] = pd.to_datetime(annual["report_date"], errors="coerce").dt.year.astype(
        "Int64"
    ).astype(str)
    annual.dropna(subset=["period"], inplace=True)
    quarterly["period"] = pd.to_datetime(quarterly["report_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    quarterly.dropna(subset=["period"], inplace=True)

    final_cols = ["ticker", "period"] + required_cols

    conn.execute("DELETE FROM annual_financials")
    conn.execute("DELETE FROM quarterly_financials")
    annual[final_cols].to_sql("annual_financials", conn, if_exists="append", index=False)
    quarterly[final_cols].to_sql(
        "quarterly_financials", conn, if_exists="append", index=False
    )
    conn.commit()
    return True

def migrate_db(db_path: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        return migrate_connection(conn)

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Migrate raw_financials to summary tables")
    parser.add_argument("--config", default="config.ini", help="Configuration file with finance_db setting")
    parser.add_argument("--db", help="Finance database path (overrides config)")
    args = parser.parse_args(argv)

    db_path = args.db or _get_finance_db(args.config)
    db_path = Path(db_path)
    if not db_path.is_absolute():
        db_path = Path(args.config).resolve().parent / db_path

    if not db_path.exists():
        parser.error(f"Database file not found: {db_path}")

    migrated = migrate_db(str(db_path))
    if migrated:
        logging.info("Migration completed")
    else:
        logging.info("Migration not required")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
