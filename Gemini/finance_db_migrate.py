import sqlite3
import os
import configparser
from pathlib import Path
import logging

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
SELECT ticker,
       strftime('%Y', report_date) AS period,
       total_revenue AS revenue,
       net_income,
       eps,
       operating_income AS op_income,
       total_assets - total_liabilities AS equity,
       total_debt,
       operating_cash_flow AS ocf,
       capital_expenditures AS capex,
       operating_income AS ebit,
       interest_expense AS interest_exp
FROM raw_financials
WHERE period='A';
"""

INSERT_QUARTERLY = """
INSERT OR REPLACE INTO quarterly_financials
SELECT ticker,
       report_date AS period,
       total_revenue AS revenue,
       net_income,
       eps,
       operating_income AS op_income,
       total_assets - total_liabilities AS equity,
       total_debt,
       operating_cash_flow AS ocf,
       capital_expenditures AS capex,
       operating_income AS ebit,
       interest_expense AS interest_exp
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

    try:
        cur.execute("SELECT COUNT(*) FROM annual_financials")
        ann_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM quarterly_financials")
        qtr_count = cur.fetchone()[0]
    except sqlite3.Error:
        ann_count = qtr_count = 0

    if ann_count == 0 and qtr_count == 0:
        cur.execute(INSERT_ANNUAL)
        cur.execute(INSERT_QUARTERLY)
        conn.commit()
        return True
    return False

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
