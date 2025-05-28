# ──────────────────────────────────────────────────────────────
#  S&P 500 High‑Growth Scoring  ——  FY + Q‑Seq Combo 版
#  ------------------------------------------------------------
#  • FY 視角：最近 2 年年报 YoY 增长
#  • Q‑Seq 視角：最近 4 季，连续 3 个季度环比增速取均值
#  • 两套分数按权重融合 → growth_score
#  • 质量 / 效率 / 安全 / 估值维持原先逻辑
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import configparser, datetime as dt, logging, sqlite3, sys, re, os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import time
RETRIES = 3
SLEEP_SEC = 0.2      # pause between retries to avoid Yahoo rate‑limit
BATCH_PAUSE = 0.1     # seconds to sleep between ticker downloads
from sqlalchemy import create_engine, text
import sqlalchemy
from tqdm import tqdm


def _get_finance_db(cfg_path: str = "config.ini") -> str:
    """Return finance DB path from config or default.

    When executed from a subdirectory, ``cfg_path`` may not be found in
    the current working directory. This helper also checks for the file
    relative to this script so that database paths resolve correctly.
    """
    cfg = configparser.ConfigParser()
    path = Path(cfg_path)
    if not path.exists():
        alt = Path(__file__).resolve().parent.parent / cfg_path
        if alt.exists():
            path = alt
    if path.exists():
        cfg.read(path, encoding="utf-8")
    db_val = None
    if cfg.has_option("database", "GPT_finance_db"):
        db_val = cfg.get("database", "GPT_finance_db")
    else:
        db_val = cfg.get("database", "finance_db", fallback="SP500_finance_data.db")

    db_path = Path(db_val)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parent.parent / db_val
    return str(db_path)


import json

# ----------------------------------------------------------------------
# Helper : load JSON → DataFrame from staging tables (works with new format)
# ----------------------------------------------------------------------
def _load_json_df(cur: sqlite3.Cursor, ticker: str, table: str) -> pd.DataFrame:
    cur.execute(f"SELECT data_json FROM {table} WHERE ticker=?", (ticker,))
    row = cur.fetchone()
    if not row or row[0] in (None, ""):
        return pd.DataFrame()
    try:
        df = pd.read_json(row[0], orient="split")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        return df.sort_index()
    except (ValueError, TypeError):
        return pd.DataFrame()

# ----------------------------------------------------------------------
# Internal util: write one ticker's data into raw_financials
# ----------------------------------------------------------------------
def _save_financial_data_for_ticker(conn: sqlite3.Connection,
                                    ticker: str,
                                    batches: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                                    price: float,
                                    forward_eps: float) -> None:
    """
    Insert quarterly/annual financial statements for *ticker* into the
    raw_financials table inside *conn*.

    *batches* = [("Q"|"A", income_df, balance_df, cashflow_df), ...]
    All DataFrames have DatetimeIndex (report period end).
    """
    for period, inc, bal, cf in batches:
        # Normalise indices
        inc.index = pd.to_datetime(inc.index)
        bal.index = pd.to_datetime(bal.index)
        cf.index  = pd.to_datetime(cf.index)

        # union index to ensure alignment
        idx = inc.index.union(bal.index).union(cf.index)
        inc = inc.reindex(idx)
        bal = bal.reindex(idx)
        cf  = cf.reindex(idx)

        if inc.empty:
            continue

        # Helper for field extraction (reuse existing first_available & _norm defined below)
        def grab(df: pd.DataFrame, keys: list[str], default=np.nan):
            return first_available(df, keys, default=default, idx=idx)

        out = pd.DataFrame({
            "ticker": ticker,
            "report_date": idx,
            "total_revenue": grab(inc, ["Total Revenue", "Revenue"]),
            "eps": grab(inc, ["Diluted EPS", "EPS"]),
            "gross_profit": grab(inc, ["Gross Profit"]),
            "operating_income": grab(inc, ["Operating Income"]),
            "net_income": grab(inc, ["Net Income"]),
            "research_development": grab(inc, ["Research and development", "R&D"]),
            "interest_expense": grab(inc, ["Interest Expense"]),
            "ebitda": grab(inc, ["EBITDA"]),
            # balance sheet
            "total_assets": grab(bal, ["Total Assets"]),
            "total_current_assets": grab(bal, ["Total Current Assets"]),
            "total_current_liabilities": grab(bal, ["Total Current Liabilities"]),
            "cash_and_eq": grab(bal, ["Cash And Cash Equivalents"]),
            "minority_interest": grab(bal, ["Minority Interest"], default=0.0),
            "total_debt": (grab(bal, ["Long Term Debt"], default=0.0) +
                           grab(bal, ["Short Long Term Debt"], default=0.0)),
            "shares_outstanding": grab(bal, ["Ordinary Shares Number", "Share Issued"]),
            "total_liabilities": grab(bal, ["Total Liab", "Total Liabilities"]),
            # cash‑flow
            "operating_cash_flow": grab(cf, ["Operating Cash Flow"]),
            "capital_expenditures": grab(cf, ["Capital Expenditures", "CapEx"]),
            # valuation snapshots (same for every row in this batch)
            "price": price,
            "forward_eps": forward_eps,
        })

        out["period"] = period
        out["free_cash_flow"] = (
            pd.to_numeric(out["operating_cash_flow"], errors="coerce").fillna(0.0) +
            pd.to_numeric(out["capital_expenditures"], errors="coerce").fillna(0.0)
        )

        # Ensure column order & types
        out = out.reindex(columns=RAW_COLS)
        numeric_cols = [c for c in out.columns if c not in ("ticker", "report_date", "period")]
        out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
        out["report_date"] = pd.to_datetime(out["report_date"]).dt.strftime("%Y-%m-%d")

        out.to_sql(RAW_TABLE, conn, if_exists="append", index=False)


def _process_staged_to_raw_db(stage_db: Path, out_db: Path) -> None:
    """
    Convert the JSON blobs produced by `acquire_raw_financial_data_to_staging`
    into the **raw_financials** table used by compute_metrics().

    raw_financials schema is defined by RAW_COLS list at top of this script.
    """
    # Ensure destination directory
    out_db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(stage_db) as stg_conn, sqlite3.connect(out_db) as fin_conn:
        cur = stg_conn.cursor()

        # Drop & recreate raw table
        fin_conn.execute(f"DROP TABLE IF EXISTS {RAW_TABLE}")
        col_defs = ", ".join(
            [f"{c} REAL" if c not in ("ticker", "report_date", "period") else f"{c} TEXT"
             for c in RAW_COLS]
        )
        fin_conn.execute(
            f"CREATE TABLE {RAW_TABLE} ({col_defs})"
        )

        # Fetch tickers from any one staging table
        tickers = [
            r[0] for r in cur.execute("SELECT DISTINCT ticker FROM stg_quarterly_income")
        ]

        for tk in tqdm(tickers, desc="Build‑GPT‑DB", leave=False):
            # Load quarterly / annual statements
            q_inc = _load_json_df(cur, tk, "stg_quarterly_income")
            q_bal = _load_json_df(cur, tk, "stg_quarterly_balance")
            q_cf  = _load_json_df(cur, tk, "stg_quarterly_cashflow")
            a_inc = _load_json_df(cur, tk, "stg_annual_income")
            a_bal = _load_json_df(cur, tk, "stg_annual_balance")
            a_cf  = _load_json_df(cur, tk, "stg_annual_cashflow")

            # price & forward_eps saved as scalar JSON (or float string)
            def _load_scalar(table: str):
                cur.execute(f"SELECT data_json FROM {table} WHERE ticker=?", (tk,))
                row = cur.fetchone()
                if row and row[0]:
                    try:
                        val = json.loads(row[0])
                        return float(val)
                    except Exception:
                        pass
                return np.nan

            price_val = _load_scalar("stg_price_summary")
            fwd_eps_val = _load_scalar("stg_key_stats")

            batches = []
            if not q_inc.empty:
                batches.append(("Q", q_inc, q_bal, q_cf))
            if not a_inc.empty:
                batches.append(("A", a_inc, a_bal, a_cf))

            if not batches:
                continue  # nothing to save

            _save_financial_data_for_ticker(fin_conn, tk, batches, price_val, fwd_eps_val)

        fin_conn.commit()


# ----------------------------------------------------------------------
# Public function called by main workflow
# ----------------------------------------------------------------------
def _prepare_gpt_finance_db(cfg_path: str = "config.ini") -> str:
    """
    Read the staging DB produced by `acquire_raw_financial_data_to_staging`,
    convert it to GPT's finance DB format (raw_financials), and return the path.
    """
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = (Path(__file__).resolve().parent.parent / cfg_file).resolve()

    cfg = configparser.ConfigParser()
    cfg.read(cfg_file, encoding="utf-8")

    stage_val = cfg.get("database", "raw_stage_db", fallback="data/SP500_raw_stage.db")
    gpt_val   = cfg.get("database", "GPT_finance_db", fallback="GPT/SP500_finance_data_GPT.db")

    stage_db = Path(stage_val)
    if not stage_db.is_absolute():
        stage_db = (cfg_file.parent / stage_db).resolve()

    gpt_db = Path(gpt_val)
    if not gpt_db.is_absolute():
        gpt_db = (cfg_file.parent / gpt_db).resolve()

    # Rebuild if GPT DB missing or older than staging DB
    rebuild = (not gpt_db.exists()) or stage_db.stat().st_mtime > gpt_db.stat().st_mtime
    if rebuild:
        logging.info("[GPT‑DB] Rebuilding GPT finance DB from staging …")
        _process_staged_to_raw_db(stage_db, gpt_db)
        logging.info("[GPT‑DB] Build complete → %s", gpt_db)
    else:
        logging.info("[GPT‑DB] Using existing GPT finance DB → %s", gpt_db)

    return str(gpt_db)


# ══════════════════ CONFIG ═══════════════════════════════════
CONFIG_FILE = Path(__file__).with_name("config_finance.ini")
DEFAULT_CONFIG = """[data]
start_date = 2010-01-01
end_date   =
update_mode = incremental          ; incremental / full

[database]
db_name = SP500_finance_data.db

[weights]           ; Σ=1
growth     = 0.45
quality    = 0.20
efficiency = 0.10
safety     = 0.15
valuation  = 0.10

[metric_parameters]
winsor_min = 0.05
winsor_max = 0.95
percentile_scope  = industry
fy_years = 2                  ; how many annual reports to look back
fy_calc  = average            ; average  |  cagr
min_industry_size = 5        ; fallback to market pct when industry sample < N

[rating_thresholds]
five_star  = 85
four_star  = 70
three_star = 55
two_star   = 40

[combo_weights]      ; FY 与 Q‑Seq 融合
fy    = 0.40
qseq  = 0.60
"""
# ------------ 保证配置文件存在且完整 -------------
def ensure_config() -> configparser.ConfigParser:
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(DEFAULT_CONFIG)
        print("[INFO] Generated default config_finance.ini")

    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.read(CONFIG_FILE, encoding="utf-8")
    base = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    base.read_string(DEFAULT_CONFIG)

    changed = False
    for sec in base.sections():
        if not cfg.has_section(sec):
            cfg.add_section(sec); changed = True
        for k, v in base.items(sec):
            if not cfg.has_option(sec, k):
                cfg.set(sec, k, v); changed = True
    if changed:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            cfg.write(f)
        print("[INFO] Missing keys added to config")
    return cfg

CFG = ensure_config()

# ═══════════════ LOGGING ═════════════════════════════════════
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("sp500-growth")

# ═════════════ GLOBAL PARAMS ═════════════════════════════════
DB_PATH  = Path(_get_finance_db())
engine   = create_engine(f"sqlite:///{DB_PATH}")
START_DATE = pd.to_datetime(CFG["data"]["start_date"])
END_DATE   = pd.to_datetime(CFG["data"].get("end_date")) if CFG["data"].get("end_date") else pd.Timestamp.today()
FY_YEARS = CFG.getint("metric_parameters", "fy_years", fallback=2)
FY_CALC  = CFG["metric_parameters"].get("fy_calc", "average").lower()
UPDATE_MODE = CFG["data"]["update_mode"].lower()

FY_W   = CFG.getfloat("combo_weights", "fy",   fallback=0.4)
QSEQ_W = CFG.getfloat("combo_weights", "qseq", fallback=0.6)
_COMBO_DENOM = FY_W + QSEQ_W if FY_W + QSEQ_W else 1.0

RAW_TABLE, METRICS_TABLE, SCORES_TABLE = "raw_financials", "derived_metrics", "scores"
 
# ───────────── Raw‑table canonical column order ──────────────
RAW_COLS = [
    "ticker", "report_date", "period",
    "total_revenue", "eps", "gross_profit", "operating_income", "net_income",
    "research_development", "interest_expense", "ebitda",
    "total_assets", "total_current_assets", "total_current_liabilities",
    "cash_and_eq", "minority_interest", "total_debt", "shares_outstanding", "total_liabilities",
    "operating_cash_flow", "capital_expenditures", "free_cash_flow",
    "price", "forward_eps",
]

# ═════════════ S&P 500 元数据 ════════════════════════════════
def get_sp500_tickers() -> pd.DataFrame:
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df.rename(columns={"Symbol":"ticker","Security":"company",
                              "GICS Sector":"sector","GICS Sub-Industry":"industry"})[
                              ["ticker","company","sector","industry"]]

SP500_META = get_sp500_tickers()
logger.info("Loaded %d S&P 500 tickers", len(SP500_META))

def latest_report_date_in_db(ticker:str):
    try:
        with engine.connect() as conn:
            res = conn.execute(text(f"SELECT MAX(report_date) FROM {RAW_TABLE} WHERE ticker=:t"),
                               {"t":ticker}).scalar()
        return pd.to_datetime(res) if res else None
    except Exception as exc:
        if isinstance(exc,(sqlite3.OperationalError, sqlalchemy.exc.OperationalError)):
            return None
        raise

# ═════════════ RAW DOWNLOAD & SAVE ═══════════════════════════
def download_single_ticker(ticker: str) -> List[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """返回 [("Q", inc, bal, cf), ("A", inc, bal, cf)] 可能缺其中之一"""
    yf_tic = yf.Ticker(ticker)
    results: list[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    def _safe(getter) -> pd.DataFrame:
        """
        Call yfinance getter with up to `RETRIES` attempts.
        Returns an empty DataFrame on persistent failure.
        """
        for attempt in range(1, RETRIES + 1):
            try:
                df = getter().T
                # yfinance returns {} when no data; convert to empty DF
                if isinstance(df, dict) or df.empty:
                    raise ValueError("empty")
                return df
            except Exception as exc:
                if attempt == RETRIES:
                    getter_name = getattr(getter, "__name__", getter.__class__.__name__)
                    logger.debug("getter %s failed after %d retries : %s",
                                 getter_name, RETRIES, exc)
                    return pd.DataFrame()
                time.sleep(SLEEP_SEC)

    q_inc = _safe(lambda: yf_tic.quarterly_income_stmt)
    q_bal = _safe(lambda: yf_tic.quarterly_balance_sheet)
    q_cf  = _safe(lambda: yf_tic.quarterly_cashflow)
    # 若收入表就绪，先保留季度批；其余两张缺失时后续指标再按列空值处理
    if not q_inc.empty:
        results.append(("Q", q_inc, q_bal, q_cf))

    a_inc = _safe(lambda: yf_tic.income_stmt)
    a_bal = _safe(lambda: yf_tic.balance_sheet)
    a_cf  = _safe(lambda: yf_tic.cashflow)
    # ── Annual data processing ───────────────────────────────
    if not a_inc.empty:
        union_idx_a = a_inc.index.union(a_bal.index).union(a_cf.index)
        a_inc = a_inc.reindex(union_idx_a)
        a_bal = a_bal.reindex(union_idx_a)
        a_cf  = a_cf.reindex(union_idx_a)

        # 要求至少 FY_YEARS 期收入表
        if a_inc.dropna(how="all").shape[0] >= FY_YEARS:
            results.append(("A", a_inc, a_bal, a_cf))
        else:
            logger.debug("%s annual rows < FY_YEARS after reindex (%d)",
                         ticker,
                         a_inc.dropna(how='all').shape[0])

    if not results:
        logger.warning("%s download failed – quarterly:%s  annual:%s",
                       ticker, q_inc.empty, a_inc.empty)
    return results

def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())

def first_available(df: pd.DataFrame, keys: list[str], default=np.nan,
                    idx: pd.Index | None = None) -> pd.Series:
    if df.empty:
        return pd.Series(default, index=idx if idx is not None else [])
    cols = list(df.columns)

    # 1. exact
    for k in keys:
        if k in cols:
            s = df[k]
            return s.reindex(idx) if idx is not None else s

    # 2. norm exact
    norm_map = {_norm(c): c for c in cols}
    for k in keys:
        nk = _norm(k)
        if nk in norm_map:
            s = df[norm_map[nk]]
            return s.reindex(idx) if idx is not None else s

    # 3. norm contains
    norm_keys = [_norm(k) for k in keys]
    for col in cols:
        nc = _norm(col)
        if any(nk in nc or nc in nk for nk in norm_keys):
            s = df[col]
            return s.reindex(idx) if idx is not None else s
    return pd.Series(default, index=idx if idx is not None else df.index)

def save_raw_to_db(ticker: str, batches):
    if not batches: return
    for period, inc, bal, cf in batches:
        for df in (inc, bal, cf):
            df.index = pd.to_datetime(df.index).tz_localize(None)
        idx = inc.index

        # incremental skip
        if UPDATE_MODE == "incremental":
            last = latest_report_date_in_db(ticker)
            if last is not None:
                mask = idx > last
                inc, bal, cf = inc[mask], bal[mask], cf[mask]
                idx = inc.index
                if inc.empty: continue

        # valuation snapshot
        try:
            y = yf.Ticker(ticker)
            price = y.history(period="1d")["Close"].iloc[-1]
            fwd   = y.info.get("forwardEps", np.nan)
        except Exception:
            price = fwd = np.nan

        out = pd.DataFrame({
            "ticker": ticker,
            "report_date": idx,
            "total_revenue": first_available(inc, ["Total Revenue","Revenue"], idx=idx),
            "eps": first_available(inc, ["Diluted EPS","EPS"], idx=idx),
            "gross_profit": first_available(inc,["Gross Profit"], idx=idx),
            "operating_income": first_available(inc,["Operating Income"], idx=idx),
            "net_income": first_available(inc,["Net Income"], idx=idx),
            "research_development": first_available(
                inc,["Research and development","R&D"], default=np.nan, idx=idx),
            "interest_expense": first_available(
                inc, ["Interest Expense"], default=np.nan, idx=idx),
            "ebitda": first_available(inc, ["EBITDA"], default=np.nan, idx=idx),

            # balance
            "total_assets": first_available(bal,["Total Assets"], idx=idx),
            "total_current_assets": first_available(bal,["Total Current Assets"], default=np.nan, idx=idx),
            "total_current_liabilities": first_available(bal,["Total Current Liabilities"], default=np.nan, idx=idx),
            "cash_and_eq": first_available(bal,["Cash And Cash Equivalents"], default=np.nan, idx=idx),
            "minority_interest": first_available(bal,["Minority Interest"], default=0, idx=idx),
            "total_debt": first_available(bal,["Long Term Debt"], default=0, idx=idx) +
                          first_available(bal,["Short Long Term Debt"], default=0, idx=idx),
            "shares_outstanding": first_available(bal,["Ordinary Shares Number","Share Issued"], default=np.nan, idx=idx),
            "total_liabilities": first_available(bal,["Total Liab","Total Liabilities"], idx=idx),

            # cash‑flow
            "operating_cash_flow": first_available(cf,["Operating Cash Flow"], idx=idx),
            "capital_expenditures": first_available(cf,["Capital Expenditures","CapEx"], default=np.nan, idx=idx),

            # valuation
            "price": price,
            "forward_eps": fwd,
        })
        out["period"] = period
        out["free_cash_flow"] = (
            pd.to_numeric(out["operating_cash_flow"], errors="coerce").fillna(0.0) +
            pd.to_numeric(out["capital_expenditures"], errors="coerce").fillna(0.0)
        )
        # re‑order & guarantee full column set so every append matches schema
        out = out.reindex(columns=RAW_COLS)
        out["report_date"] = pd.to_datetime(out["report_date"]).dt.strftime("%Y-%m-%d")

        # --- serialised DB write (fresh connection for each batch) ---
        numeric_cols = [c for c in out.columns if c not in ("ticker", "report_date", "period")]
        out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
 
        with sqlite3.connect(DB_PATH) as conn:
            out.to_sql(RAW_TABLE, conn, if_exists="append", index=False)

def download_all():
    """Disabled financial statement download."""
    logger.info("[download_all] Download step disabled – using existing database.")
# ═════════════ METRIC COMPUTATION ════════════════════════════
def seq_growth(series: pd.Series) -> float:
    """Rolling Q‑to‑Q growth (last 3 pairs) with turn‑positive patch."""
    s = series.dropna().astype(float)
    if len(s) < 4:
        return np.nan
    s = s.iloc[-4:]
    prev, curr = s.iloc[:-1].values, s.iloc[1:].values
    growth_vec = np.where((prev <= 0) & (curr > 0), 1.0, (curr / prev) - 1)
    growth_vec = np.where(np.isfinite(growth_vec), growth_vec, np.nan)
    valid = growth_vec[~np.isnan(growth_vec)]
    return valid.mean() if valid.size else np.nan 

def compute_metrics(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    raw = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", sqlite3.connect(db_path))
    # ---- force numeric dtypes -------------------------------------------------
    # SQLite may return all columns as TEXT when NaNs are present; convert every
    # non‑identifier column to numeric so downstream math (e.g. net_income /
    # shares_outstanding) doesn’t raise “unsupported operand type” errors.
    numeric_cols = [c for c in raw.columns if c not in ("ticker", "period", "report_date")]
    raw[numeric_cols] = raw[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if raw.empty: raise RuntimeError("raw_financials table is empty")
    raw["report_date"] = pd.to_datetime(raw["report_date"])
    raw_q = raw[raw["period"] == "Q"].sort_values(["ticker","report_date"])
    raw_a = raw[raw["period"] == "A"].sort_values(["ticker","report_date"])

    records=[]
    for tic in raw["ticker"].unique():
        g_q = raw_q[raw_q.ticker==tic].copy()
        g_a = raw_a[raw_a.ticker==tic].copy()
        if g_q.empty and len(g_a) < 2: continue      # 数据严重不足

        # EPS 填补
        for g in (g_q, g_a):
            if g.empty: continue
            eps=g["eps"].copy()
            mask=eps.isna()
            if mask.any():
                shares=g["shares_outstanding"].replace(0,np.nan)
                eps[mask]=g["net_income"]/shares
            g["eps_filled"]=eps

        # ───── FY YoY ─────
        if len(g_a) >= FY_YEARS:
            g_a_recent = g_a.iloc[-FY_YEARS:]                   # keep last N year‑ends
            n = FY_YEARS - 1

            def _series(name):
                return g_a_recent[name].astype(float)

            rev_s   = _series("total_revenue")
            eps_s   = _series("eps_filled")
            fcf_s   = _series("free_cash_flow")
            margin_s = (g_a_recent["gross_profit"] / g_a_recent["total_revenue"]).astype(float)

            if FY_CALC == "average":
                def _avg_growth(series: pd.Series) -> float:
                    """
                    Compute period‑over‑period growth for a 1‑D numeric Series and
                    return the average of valid growth observations.

                    Rules
                    -----
                    1. If fewer than 2 non‑NaN points → return np.nan.
                    2. growth = (curr / prev) − 1
                    3. Special case: if prev ≤ 0 and curr > 0 (turning positive),
                       treat growth as +100 % (1.0).
                    4. Exclude inf/‑inf/NaN before averaging.

                    This implementation avoids the previous 'base_growth' variable
                    misuse that triggered UnboundLocalError.
                    """
                    s = series.astype(float)
                    if s.dropna().shape[0] < 2:
                        return np.nan

                    prev = s.shift(1)
                    growth = (s / prev) - 1

                    # handle negative→positive turnaround
                    turn_pos_mask = (prev <= 0) & (s > 0)
                    growth.loc[turn_pos_mask] = 1.0

                    growth = growth.replace([np.inf, -np.inf], np.nan).dropna()
                    return growth.mean() if not growth.empty else np.nan

                fy_rev_y = _avg_growth(rev_s)
                fy_eps_y = _avg_growth(eps_s)
                fy_fcf_y = _avg_growth(fcf_s)
                diff_margin = margin_s.iloc[1:].values - margin_s.iloc[:-1].values
                valid = diff_margin[~np.isnan(diff_margin)]
                fy_margin_delta = valid.mean() if valid.size else np.nan
            else:          # 'cagr'
                def _cagr(last_val: float, first_val: float) -> float:
                    """CAGR with turn‑positive patch."""
                    if pd.isna(first_val) or pd.isna(last_val):
                        return np.nan
                    # 负→正 直接给 1.0 (100 %/年)；其它非法值返回 NaN
                    if first_val <= 0 and last_val > 0:
                        return 1.0
                    if first_val <= 0 or last_val <= 0:
                        return np.nan
                    try:
                        return (last_val / first_val) ** (1 / n) - 1
                    except Exception:
                        return np.nan

                fy_rev_y = _cagr(rev_s.iloc[-1], rev_s.iloc[0])
                fy_eps_y = _cagr(eps_s.iloc[-1], eps_s.iloc[0])
                fy_fcf_y = _cagr(fcf_s.iloc[-1], fcf_s.iloc[0])
                fy_margin_delta = (margin_s.iloc[-1] - margin_s.iloc[0]) / n
        else:
            fy_rev_y = fy_eps_y = fy_fcf_y = fy_margin_delta = np.nan

        # ───── Q‑Seq ─────
        qseq_rev   = seq_growth(g_q["total_revenue"])
        qseq_eps   = seq_growth(g_q["eps_filled"])
        qseq_fcf   = seq_growth(g_q["free_cash_flow"])
        # 毛利率环比
        if len(g_q.dropna(subset=["total_revenue","gross_profit"])) >= 4:
            gross_margin = (g_q["gross_profit"]/g_q["total_revenue"]).astype(float)
            qseq_margin_delta = seq_growth(gross_margin)
        else:
            qseq_margin_delta = np.nan

        # ---------- 其它维度基于最新季度 ----------
        latest = (g_q if not g_q.empty else g_a).iloc[-1]
        td   = latest["total_debt"] or 0
        mi   = latest["minority_interest"] or 0
        ta   = latest["total_assets"] or 0
        tl   = latest["total_liabilities"] or 0
        cash = latest["cash_and_eq"] or 0

        nopat = latest["operating_income"]*0.79 if pd.notna(latest["operating_income"]) else np.nan
        invested_cap = td + mi + (ta - tl) - cash
        roic = np.nan if (pd.isna(invested_cap) or invested_cap <= 0) else nopat / invested_cap

        equity = ta - tl
        roe = latest["net_income"] / equity if equity else np.nan

        ocf_ratio      = latest["operating_cash_flow"] / latest["total_revenue"] if latest["total_revenue"] else np.nan
        asset_turnover = latest["total_revenue"] / ta if ta else np.nan

        net_debt       = td - cash
        net_debt_ebitda = net_debt / latest["ebitda"] if latest["ebitda"] else np.nan
        interest_cov    = latest["operating_income"] / abs(latest["interest_expense"]) if latest["interest_expense"] else np.nan
        current_ratio   = latest["total_current_assets"] / latest["total_current_liabilities"] if latest["total_current_liabilities"] else np.nan

        price, eps_fwd = latest["price"], latest["forward_eps"]
        peg = (price/eps_fwd)/(qseq_eps) if price and eps_fwd and qseq_eps and qseq_eps>0 else np.nan
        fcf_yield = latest["free_cash_flow"] / (price*latest["shares_outstanding"]) \
                    if price and latest["shares_outstanding"] else np.nan

        records.append({
            "ticker": tic,
            "report_date": latest["report_date"],
            # FY
            "fy_rev_y": fy_rev_y, "fy_eps_y": fy_eps_y,
            "fy_fcf_y": fy_fcf_y, "fy_margin_delta": fy_margin_delta,
            # Q‑Seq
            "qseq_rev": qseq_rev, "qseq_eps": qseq_eps,
            "qseq_fcf": qseq_fcf,"qseq_margin_delta": qseq_margin_delta,
            # quality & others
            "roic": roic, "roe": roe,
            "ocf_ratio": ocf_ratio, "asset_turnover": asset_turnover,
            "net_debt_ebitda": net_debt_ebitda, "interest_coverage": interest_cov,
            "current_ratio": current_ratio,
            "peg": peg, "fcf_yield": fcf_yield,
        })

    dfm = pd.DataFrame(records).merge(SP500_META, on="ticker", how="left")
    dfm.to_sql(METRICS_TABLE, sqlite3.connect(DB_PATH), if_exists="replace", index=False)
    logger.info("Computed metrics for %d companies", len(dfm))
    return dfm

# ──────────────────────────────────────────────────────────────
#   Part 2 —— Scoring / Export / main
# ──────────────────────────────────────────────────────────────
# ---------- 评分参数 ----------
WINSOR_MIN = CFG.getfloat("metric_parameters","winsor_min",fallback=0.05)
WINSOR_MAX = CFG.getfloat("metric_parameters","winsor_max",fallback=0.95)
PCT_SCOPE  = CFG["metric_parameters"].get("percentile_scope","industry")
MIN_INDUSTRY_SIZE = CFG.getint("metric_parameters", "min_industry_size", fallback=5)

WEIGHTS = {k:CFG.getfloat("weights",k) for k in
           ["growth","quality","efficiency","safety","valuation"]}

DIM_MAP = {
    "growth_fy"  : ["fy_rev_y","fy_eps_y","fy_fcf_y","fy_margin_delta"],
    "growth_qseq": ["qseq_rev","qseq_eps","qseq_fcf","qseq_margin_delta"],
    "quality"    : ["roic","roe"],
    "efficiency" : ["ocf_ratio","asset_turnover"],
    "safety"     : ["net_debt_ebitda","interest_coverage","current_ratio"],
    "valuation"  : ["peg","fcf_yield"],
}

HIGH_IS_GOOD = {
    # Growth
    "fy_rev_y": True,  "fy_eps_y": True,  "fy_fcf_y": True,  "fy_margin_delta": True,
    "qseq_rev": True,  "qseq_eps": True,  "qseq_fcf": True, "qseq_margin_delta": True,
    # Quality / Efficiency
    "roic": True, "roe": True,
    "ocf_ratio": True, "asset_turnover": True,
    # Safety (低为好)
    "net_debt_ebitda": False,
    "interest_coverage": True,          # 越高越好
    "current_ratio": True,              # 过高亦不佳，可视需求设软阈值
    # Valuation
    "peg": False,                       # 越低越好
    "fcf_yield": True                   # 越高越好
}

HIGH_GOOD = HIGH_IS_GOOD


# ---------- 工具函数 ----------
def winsor(s: pd.Series) -> pd.Series:
    return s.clip(s.quantile(WINSOR_MIN), s.quantile(WINSOR_MAX))

def calc_scores(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    # 1) 缺失列补空
    for col in HIGH_GOOD:
        if col not in df2.columns:
            df2[col] = np.nan

    # 2) 缺失值 → 行业中位数，再 winsorize
    for col in HIGH_GOOD:
        if df2[col].isna().all():
            df2[col] = 0         # 行业全空时补 0，得分中性
        else:
            df2[col] = df2[col].fillna(
                df2.groupby("industry")[col].transform("median")
            )
        df2[col] = winsor(df2[col])

    # 3) 分位化得分（0‑100），小行业样本(<MIN_INDUSTRY_SIZE)回退到全市场分位
    scope = PCT_SCOPE
    industry_sizes = df2["industry"].map(df2["industry"].value_counts())
    for col in HIGH_GOOD:
        global_pct   = df2[col].rank(pct=True)
        if scope == "all":
            pct = global_pct
        else:
            industry_pct = df2.groupby("industry")[col].rank(pct=True)
            pct = np.where(industry_sizes >= MIN_INDUSTRY_SIZE,
                           industry_pct,
                           global_pct)
        df2[f"{col}_score"] = pct * 100 if HIGH_GOOD[col] else (100 - pct * 100)

    # 4) FY‑score & Q‑Seq‑score → 融合 (修正使用 *_score 列)
    base_metrics = {"rev":"rev_y","eps":"eps_y","fcf":"fcf_y","margin":"margin_delta"}
    for key, suffix in base_metrics.items():
        fy_col = f"fy_{suffix}_score"
        qs_col = f"qseq_{suffix if key!='margin' else 'margin_delta'}_score"
        fy  = df2.get(fy_col, 0)
        qs  = df2.get(qs_col, 0)
        df2[f"c_{key}_score"] = (FY_W*fy + QSEQ_W*qs) / _COMBO_DENOM

    # 5) Growth 维度 —— 权重自适应剔除 NaN
    G_W = np.array([0.5, 0.3, 0.1, 0.1])            # rev, eps, fcf, margin
    G_COLS = ["c_rev_score", "c_eps_score", "c_fcf_score", "c_margin_score"]

    score_mat = df2[G_COLS].to_numpy(dtype=float)
    weight_mat = np.tile(G_W, (len(df2), 1))

    nan_mask = np.isnan(score_mat)
    weight_mat[nan_mask] = 0.0                      # 缺失项权重→0
    score_mat = np.nan_to_num(score_mat, copy=False)

    weight_sum = weight_mat.sum(axis=1)
    # 全部缺失时 weight_sum==0 → growth_score 设为 NaN
    growth_scores = np.where(
        weight_sum > 0,
        (weight_mat * score_mat).sum(axis=1) / weight_sum,
        np.nan
    )
    df2["growth_score"] = growth_scores

    # 6) 其余维度：质量 / 效率 / 安全 / 估值
    for dim in ["quality", "efficiency", "safety", "valuation"]:
        score_cols = [f"{c}_score" for c in DIM_MAP[dim]]
        df2[f"{dim}_score"] = df2[score_cols].mean(axis=1)

    # 7) Total Score —— 权重自适应，剔除 NaN 维度
    DIM_COLS   = ["growth_score", "quality_score", "efficiency_score",
                  "safety_score", "valuation_score"]
    DIM_W      = np.array([
        WEIGHTS["growth"],
        WEIGHTS["quality"],
        WEIGHTS["efficiency"],
        WEIGHTS["safety"],
        WEIGHTS["valuation"],
    ], dtype=float)

    dim_mat    = df2[DIM_COLS].to_numpy(dtype=float)
    weight_mat = np.tile(DIM_W, (len(df2), 1))

    nan_mask   = np.isnan(dim_mat)
    weight_mat[nan_mask] = 0.0
    dim_mat    = np.nan_to_num(dim_mat, copy=False)

    w_sum      = weight_mat.sum(axis=1)
    total_scores = np.where(
        w_sum > 0,
        (weight_mat * dim_mat).sum(axis=1) / w_sum,
        np.nan
    )
    df2["total_score"] = total_scores

    # 写回数据库
    df2.to_sql(SCORES_TABLE, sqlite3.connect(DB_PATH),
               if_exists="replace", index=False)
    return df2

# ---------- 导出 ----------
# ---------- 导出 ----------
def export_excel(df: pd.DataFrame, out_path: str | None = None):
    """
    导出基本面评分榜单到 Excel。
    1. 若显式传入 out_path，则优先使用该值；
    2. 否则从 config_finance.ini → [export] → excel_file_name 读取；
    3. 若配置缺失，则退回默认 'high_growth_scoring_YYYYMMDD.xlsx'。
    """
    # ―― 1) 读取配置 ──────────────────────────────────────────
    cfg      = CFG                      # CFG 已在文件顶部通过 ensure_config() 读取
    out_cfg  = None
    if cfg.has_section("export") and cfg.has_option("export", "excel_file_name"):
        out_cfg = cfg["export"]["excel_file_name"].strip()

    # ―― 2) 选择最终文件名 ────────────────────────────────────
    if out_path:
        out_file = Path(out_path)
    elif out_cfg:
        out_file = Path(out_cfg)
    else:
        out_file = Path(f"high_growth_scoring_{dt.date.today():%Y%m%d}.xlsx")

    # ―― 3) 导出 ─────────────────────────────────────────────
    cols = list(df.columns)
    cols.insert(2, cols.pop(cols.index("total_score")))   # 保持列次序
    df.sort_values("total_score", ascending=False)[cols].to_excel(out_file, index=False)
    logger.info("Exported fundamental scores → %s", out_file)


# ---------- main ----------
def Testmain():
    download_all()              # 如不想更新，可注释
    db_path = _prepare_gpt_finance_db()
    metrics = compute_metrics(db_path=db_path)
    scores  = calc_scores(metrics)
    export_excel(scores)

if __name__ == "__main__":
    Testmain()



