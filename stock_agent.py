"""
stock_agent.py

Daily stock-screening agent for:
  - Canada: S&P/TSX 60
  - U.S.: S&P 500 + Nasdaq-100

The model ranks stocks using:
  1) 1-, 3-, 6-, and 12-month momentum
  2) relative strength vs. a market benchmark
  3) trend vs. 50- and 200-day moving average
  4) volatility and drawdown
  5) volume acceleration
  6) RSI "fitness"
  7) trend quality (R² of price vs linear trend) [NEW]
  8) 52-week high proximity [NEW]
  9) sector-neutralised cross-sectional ranking [NEW]
  10) a lightweight fundamental-quality overlay for finalists
      (PEG ratio replaces raw P/E in valuation component) [NEW]
  11) earnings-event flag — finalists within 5 days of reporting
      are surfaced with a warning [NEW]

IMPORTANT:
This is a screening/ranking model, not a prediction or guarantee of future returns.
Backtest and paper-trade it before using real money.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIGURATION
# -----------------------------

REPORT_DIR = Path("stock_agent_reports")

MIN_PRICE_USD_CAD = 5.00
MIN_AVG_DOLLAR_VOLUME = 20_000_000   # roughly $20M/day
FINALISTS_PER_MARKET = 20
TOP_TO_DISPLAY = 5
STRONG_SIGNAL_THRESHOLD = 72.0

# Technical score weights — must sum to 1.00
# New: trend_r2 (0.05) and high52w (0.05) added;
#      ret_126 trimmed from 0.20→0.17, rs_126 from 0.15→0.13,
#      above_sma200 from 0.10→0.08 to keep total at 1.00.
TECH_WEIGHTS = {
    "ret_126":       0.17,   # ~6 months
    "ret_63":        0.15,   # ~3 months
    "ret_21":        0.10,   # ~1 month
    "rs_126":        0.13,   # relative strength vs benchmark
    "above_sma200":  0.08,   # price vs 200-day MA
    "low_volatility":0.10,
    "low_drawdown":  0.10,
    "volume_ratio":  0.05,
    "rsi_fitness":   0.05,
    "trend_r2":      0.05,   # NEW: quality / linearity of the trend
    "high52w":       0.02,   # NEW: proximity to 52-week high
}

# Sector-neutral blend: 70% cross-sectional rank + 30% within-sector rank
SECTOR_NEUTRAL_WEIGHT = 0.30

# Final score: technical + fundamental overlay
TECHNICAL_FINAL_WEIGHT = 0.78
FUNDAMENTAL_FINAL_WEIGHT = 0.22

# Earnings proximity warning window (trading days)
EARNINGS_WARN_DAYS = 5

USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/153 Safari/537.36"
    )
}


# -----------------------------
# UNIVERSE
# -----------------------------

def _read_html_tables(url: str) -> List[pd.DataFrame]:
    """Read HTML tables using a normal browser User-Agent."""
    r = requests.get(url, headers=USER_AGENT, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def _find_table_with_column(
    tables: List[pd.DataFrame], candidates: Iterable[str]
) -> pd.DataFrame:
    candidates_lower = {c.lower() for c in candidates}
    matches = []
    for table in tables:
        cols = {str(c).strip().lower() for c in table.columns}
        if cols & candidates_lower:
            matches.append(table)
    if not matches:
        raise RuntimeError(
            f"Could not find a table containing any of: {candidates}"
        )
    return max(matches, key=len)


def yahoo_us_symbol(symbol: str) -> str:
    return str(symbol).strip().replace(".", "-")


def yahoo_tsx_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().replace(".", "-")
    return f"{symbol}.TO"


def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html_tables(url)
    table = _find_table_with_column(tables, ["Symbol"])
    col = next(c for c in table.columns if str(c).strip().lower() == "symbol")
    return [yahoo_us_symbol(x) for x in table[col].dropna().tolist()]


def get_nasdaq100_tickers() -> List[str]:
    """
    Load current Nasdaq-100 constituents from Nasdaq's JSON endpoint.

    The JSON endpoint is tried first; the dedicated Wikipedia constituent page
    is retained as a fallback so a temporary Nasdaq API change does not bring
    down the whole advisory.
    """
    nasdaq_url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
    headers = {
        **USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nasdaq.com/",
    }

    try:
        r = requests.get(nasdaq_url, headers=headers, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("data", {}).get("data", {}).get("rows", [])
        values = [yahoo_us_symbol(row.get("symbol", "")) for row in rows]
        values = [x for x in values if 1 <= len(x) <= 10 and " " not in x]
        if len(values) >= 90:
            return sorted(set(values))
        raise RuntimeError(
            f"Nasdaq returned only {len(values)} usable symbols"
        )
    except Exception as exc:
        print(
            f"Nasdaq-100 direct load failed ({exc}); using fallback source..."
        )

    # Fallback: dedicated Wikipedia constituent page.
    fallback_url = "https://en.wikipedia.org/wiki/List_of_Nasdaq-100_companies"
    tables = _read_html_tables(fallback_url)
    table = _find_table_with_column(tables, ["Ticker", "Symbol"])
    col = next(
        c for c in table.columns
        if str(c).strip().lower() in {"ticker", "symbol"}
    )
    values = [yahoo_us_symbol(x) for x in table[col].dropna().tolist()]
    return sorted(
        set(x for x in values if 1 <= len(x) <= 10 and " " not in x)
    )


def get_tsx60_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/S%26P/TSX_60"
    tables = _read_html_tables(url)
    table = _find_table_with_column(tables, ["Symbol", "Ticker"])
    col = next(
        c for c in table.columns
        if str(c).strip().lower() in {"symbol", "ticker"}
    )
    values = [str(x).strip() for x in table[col].dropna().tolist()]
    return [
        yahoo_tsx_symbol(x)
        for x in values
        if 1 <= len(x) <= 10 and " " not in x
    ]


def get_universe() -> Dict[str, List[str]]:
    us = sorted(set(get_sp500_tickers()) | set(get_nasdaq100_tickers()))
    canada = sorted(set(get_tsx60_tickers()))
    return {"US": us, "CANADA": canada}


# -----------------------------
# MARKET DATA
# -----------------------------

def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def download_history(
    tickers: List[str],
    period: str = "1y",
    batch_size: int = 75,
    pause_seconds: float = 0.8,
) -> Dict[str, pd.DataFrame]:
    """Download adjusted daily data in batches. Returns {ticker: dataframe}."""
    result: Dict[str, pd.DataFrame] = {}

    for batch in chunks(tickers, batch_size):
        try:
            data = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print(f"Batch download failed: {exc}")
            time.sleep(2)
            continue

        if len(batch) == 1:
            ticker = batch[0]
            df = data.copy()
            if not df.empty and not _yf_series(df, "Close").empty:
                result[ticker] = df.dropna(how="all")
        else:
            for ticker in batch:
                try:
                    df = data[ticker].copy()
                    if not df.empty and not _yf_series(df, "Close").empty:
                        result[ticker] = df.dropna(how="all")
                except Exception:
                    pass

        time.sleep(pause_seconds)

    return result


def download_benchmark(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    return df.dropna(how="all")


# -----------------------------
# YFINANCE COLUMN NORMALISATION
# -----------------------------

def _yf_series(df: pd.DataFrame, field: str) -> pd.Series:
    """Return a 1-D numeric Series from flat or MultiIndex yfinance output."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    obj = None

    if field in df.columns:
        obj = df[field]
    elif isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            vals = df.columns.get_level_values(level)
            if field in vals:
                try:
                    obj = df.xs(field, axis=1, level=level)
                    break
                except Exception:
                    pass

    if obj is None:
        return pd.Series(dtype=float)

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype=float)
        obj = obj.iloc[:, 0]

    return pd.to_numeric(obj, errors="coerce").dropna()


def ohlcv_series(df: pd.DataFrame, field: str) -> pd.Series:
    """Public OHLCV normaliser used by the Flask advisory backend."""
    return _yf_series(df, field)


# -----------------------------
# FEATURE HELPERS
# -----------------------------

def safe_return(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return np.nan
    old = float(close.iloc[-periods - 1])
    new = float(close.iloc[-1])
    if old <= 0:
        return np.nan
    return new / old - 1.0


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()

    if len(close) < period + 2:
        return np.nan

    avg_gain = gain.iloc[-1]
    avg_loss = loss.iloc[-1]

    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return np.nan
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def max_drawdown(close: pd.Series, lookback: int = 126) -> float:
    x = close.tail(lookback)
    if len(x) < 20:
        return np.nan
    peak = x.cummax()
    dd = x / peak - 1.0
    return float(dd.min())


def trend_r2(close: pd.Series, lookback: int = 63) -> float:
    """
    NEW: R² of closing price vs a linear time trend over `lookback` days.
    A value near 1.0 means the stock has been moving in a clean, consistent
    direction.  A low value means choppy, range-bound price action.
    Returns 0.0 on insufficient data.
    """
    y = close.tail(lookback).values
    if len(y) < 20:
        return 0.0
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def days_to_earnings(ticker: str) -> Optional[int]:
    """
    NEW: Return the number of calendar days until the next earnings event,
    or None if the calendar is unavailable.
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        # yfinance >= 0.2 returns a dict; older versions return a DataFrame.
        if isinstance(cal, dict):
            date_val = cal.get("Earnings Date")
            if date_val is None:
                return None
            # May be a list of dates; take the soonest.
            if isinstance(date_val, (list, tuple)):
                date_val = date_val[0]
            ed = pd.Timestamp(date_val)
        else:
            if cal.empty:
                return None
            ed = pd.Timestamp(cal.iloc[0, 0])
        delta = (ed - pd.Timestamp.now(tz=ed.tz)).days
        return int(delta)
    except Exception:
        return None


def benchmark_returns(benchmark_df: pd.DataFrame) -> Dict[str, float]:
    close = _yf_series(benchmark_df, "Close")
    return {
        "ret_63": safe_return(close, 63),
        "ret_126": safe_return(close, 126),
    }


def make_feature_row(
    ticker: str,
    market: str,
    df: pd.DataFrame,
    bench: Dict[str, float],
) -> Optional[dict]:
    if df is None or df.empty:
        return None

    close = _yf_series(df, "Close")
    volume = _yf_series(df, "Volume").reindex(close.index).fillna(0)
    if close.empty or volume.empty:
        return None

    if len(close) < 210:
        return None

    price = float(close.iloc[-1])
    if not math.isfinite(price) or price < MIN_PRICE_USD_CAD:
        return None

    dollar_volume_20 = float((close * volume).tail(20).mean())
    if (
        not math.isfinite(dollar_volume_20)
        or dollar_volume_20 < MIN_AVG_DOLLAR_VOLUME
    ):
        return None

    daily_ret = close.pct_change().dropna()
    vol_63 = float(daily_ret.tail(63).std() * np.sqrt(252))

    sma200 = float(close.tail(200).mean())
    ret21  = safe_return(close, 21)
    ret63  = safe_return(close, 63)
    ret126 = safe_return(close, 126)

    vol20 = float(volume.tail(20).mean())
    vol60 = float(volume.tail(60).mean())
    volume_ratio = vol20 / vol60 if vol60 > 0 else np.nan

    rsi = compute_rsi(close)
    rsi_fitness = (
        np.clip(1.0 - abs(rsi - 60.0) / 35.0, 0.0, 1.0)
        if np.isfinite(rsi)
        else 0.5
    )

    # NEW: trend quality and 52-week high proximity
    tr2 = trend_r2(close, lookback=63)
    high_252 = float(close.tail(252).max())
    dist_52w_high = (price / high_252 - 1.0) if high_252 > 0 else np.nan

    return {
        "ticker": ticker,
        "market": market,
        "price": price,
        "avg_dollar_volume_20": dollar_volume_20,
        "ret_21": ret21,
        "ret_63": ret63,
        "ret_126": ret126,
        "rs_126": ret126 - bench["ret_126"],
        "above_sma200": price / sma200 - 1.0 if sma200 > 0 else np.nan,
        "volatility_63": vol_63,
        "max_drawdown_126": max_drawdown(close, 126),
        "volume_ratio": volume_ratio,
        "rsi_14": rsi,
        "rsi_fitness_raw": rsi_fitness,
        "trend_r2": tr2,                    # NEW
        "dist_52w_high": dist_52w_high,     # NEW
    }


# -----------------------------
# CROSS-SECTIONAL SCORING
# -----------------------------

def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Convert a feature to 0..100 cross-sectional percentile rank."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(50.0, index=series.index)

    if higher_is_better:
        ranked = s.rank(pct=True, ascending=True) * 100.0
    else:
        ranked = (1.0 - s.rank(pct=True, ascending=True)) * 100.0

    return ranked.fillna(50.0).clip(0, 100)


def _sector_relative_rank(
    df: pd.DataFrame, score_col: str
) -> pd.Series:
    """
    NEW: Within-sector percentile rank of `score_col`.
    Stocks whose sector is unknown are grouped together.
    Returns a 0..100 series aligned to df.index.
    """
    sector_col = "sector" if "sector" in df.columns else None
    if sector_col is None:
        return pd.Series(50.0, index=df.index)

    result = pd.Series(50.0, index=df.index)
    for _, grp in df.groupby(sector_col, dropna=False):
        s = pd.to_numeric(grp[score_col], errors="coerce")
        if s.notna().sum() < 2:
            result.loc[grp.index] = 50.0
        else:
            result.loc[grp.index] = (
                s.rank(pct=True, ascending=True) * 100.0
            ).fillna(50.0).clip(0, 100)
    return result


def score_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["score_ret_126"]       = percentile(out["ret_126"],         True)
    out["score_ret_63"]        = percentile(out["ret_63"],          True)
    out["score_ret_21"]        = percentile(out["ret_21"],          True)
    out["score_rs_126"]        = percentile(out["rs_126"],          True)
    out["score_above_sma200"]  = percentile(out["above_sma200"],    True)
    out["score_low_volatility"]= percentile(out["volatility_63"],   False)
    out["score_low_drawdown"]  = percentile(out["max_drawdown_126"],True)
    out["score_volume_ratio"]  = percentile(out["volume_ratio"],    True)
    out["score_rsi_fitness"]   = out["rsi_fitness_raw"] * 100.0
    out["score_trend_r2"]      = out["trend_r2"] * 100.0          # NEW
    # NEW: 52-week high — stocks near (or at) new highs score highest.
    # dist_52w_high is ≤ 0 (at high = 0, below high = negative).
    # Invert so a smaller drawdown from the high = higher score.
    out["score_high52w"]       = percentile(out["dist_52w_high"],   True)

    # Raw cross-sectional technical score
    raw_score = (
        TECH_WEIGHTS["ret_126"]        * out["score_ret_126"]
        + TECH_WEIGHTS["ret_63"]       * out["score_ret_63"]
        + TECH_WEIGHTS["ret_21"]       * out["score_ret_21"]
        + TECH_WEIGHTS["rs_126"]       * out["score_rs_126"]
        + TECH_WEIGHTS["above_sma200"] * out["score_above_sma200"]
        + TECH_WEIGHTS["low_volatility"]* out["score_low_volatility"]
        + TECH_WEIGHTS["low_drawdown"] * out["score_low_drawdown"]
        + TECH_WEIGHTS["volume_ratio"] * out["score_volume_ratio"]
        + TECH_WEIGHTS["rsi_fitness"]  * out["score_rsi_fitness"]
        + TECH_WEIGHTS["trend_r2"]     * out["score_trend_r2"]
        + TECH_WEIGHTS["high52w"]      * out["score_high52w"]
    )

    # NEW: sector-neutral blend — prevents one hot sector flooding the top 5
    sector_rank = _sector_relative_rank(out, "score_ret_126")
    out["technical_score"] = (
        (1.0 - SECTOR_NEUTRAL_WEIGHT) * raw_score
        + SECTOR_NEUTRAL_WEIGHT * sector_rank
    ).clip(0, 100)

    return out


# -----------------------------
# FUNDAMENTAL OVERLAY
# -----------------------------

def bounded_score(value: Optional[float], low: float, high: float) -> float:
    if value is None:
        return 50.0
    try:
        x = float(value)
    except Exception:
        return 50.0
    if not np.isfinite(x):
        return 50.0
    if high == low:
        return 50.0
    return float(np.clip((x - low) / (high - low), 0.0, 1.0) * 100.0)


def peg_score(forward_pe: Optional[float], earnings_growth: Optional[float]) -> float:
    """
    NEW: PEG-based valuation score.
    PEG = forward P/E / (earnings growth rate expressed as a percentage).
    PEG < 1 is conventionally considered cheap relative to growth;
    PEG > 2.5 is expensive.  Missing data → neutral 50.
    """
    try:
        pe = float(forward_pe)
        eg = float(earnings_growth)
    except (TypeError, ValueError):
        return 50.0
    if not (np.isfinite(pe) and np.isfinite(eg)):
        return 50.0
    if pe <= 0 or eg <= 0:
        # Negative earnings growth or negative P/E — can't compute meaningful PEG
        return 35.0
    eg_pct = eg * 100.0   # earnings_growth arrives as a decimal (e.g. 0.25 = 25%)
    peg = pe / eg_pct
    if peg < 0.75:
        return 95.0
    if peg < 1.0:
        return 85.0
    if peg < 1.5:
        return 72.0
    if peg < 2.5:
        return 55.0
    return 30.0


def fetch_fundamental_snapshot(ticker: str) -> dict:
    """
    yfinance fundamentals can be missing for some securities.
    Missing values receive neutral scores rather than causing failure.
    Valuation component now uses PEG ratio instead of raw P/E.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    revenue_growth  = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")
    profit_margin   = info.get("profitMargins")
    roe             = info.get("returnOnEquity")
    forward_pe      = info.get("forwardPE")
    market_cap      = info.get("marketCap")
    sector          = info.get("sector")
    company         = info.get("shortName") or info.get("longName") or ticker

    score = (
        0.25 * bounded_score(revenue_growth,  -0.05, 0.25)
        + 0.25 * bounded_score(earnings_growth, -0.10, 0.30)
        + 0.20 * bounded_score(profit_margin,    0.00, 0.30)
        + 0.20 * bounded_score(roe,              0.00, 0.30)
        + 0.10 * peg_score(forward_pe, earnings_growth)   # NEW: PEG replaces raw P/E
    )

    return {
        "company":         company,
        "sector":          sector,
        "market_cap":      market_cap,
        "revenue_growth":  revenue_growth,
        "earnings_growth": earnings_growth,
        "profit_margin":   profit_margin,
        "roe":             roe,
        "forward_pe":      forward_pe,
        "fundamental_score": score,
    }


def enrich_finalists(scored: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for market in ["US", "CANADA"]:
        subset = (
            scored[scored["market"] == market]
            .nlargest(FINALISTS_PER_MARKET, "technical_score")
            .copy()
        )

        fundamentals = []
        for ticker in subset["ticker"]:
            snap = fetch_fundamental_snapshot(ticker)
            # NEW: attach earnings proximity warning
            dte = days_to_earnings(ticker)
            snap["days_to_earnings"] = dte
            snap["earnings_near"] = (
                dte is not None and 0 <= dte <= EARNINGS_WARN_DAYS
            )
            fundamentals.append(snap)
            time.sleep(0.15)

        fund_df = pd.DataFrame(fundamentals, index=subset.index)
        subset = pd.concat([subset, fund_df], axis=1)
        frames.append(subset)

    finalists = pd.concat(frames, ignore_index=True)

    finalists["final_score"] = (
        TECHNICAL_FINAL_WEIGHT  * finalists["technical_score"]
        + FUNDAMENTAL_FINAL_WEIGHT * finalists["fundamental_score"]
    )

    return finalists.sort_values("final_score", ascending=False).reset_index(
        drop=True
    )


# -----------------------------
# OPTIONAL HEADLINES FOR TOP PICKS
# -----------------------------

def recent_headlines(ticker: str, n: int = 3) -> List[str]:
    """Headlines are displayed for context only; NOT used in the score."""
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    headlines = []
    for item in news:
        title = None
        if isinstance(item, dict):
            title = item.get("title")
            if not title and isinstance(item.get("content"), dict):
                title = item["content"].get("title")
        if title:
            headlines.append(str(title).strip())
        if len(headlines) >= n:
            break

    return headlines


# -----------------------------
# REPORTING
# -----------------------------

def pct(x) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "n/a"


def money(x) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x):,.2f}"
    except Exception:
        return "n/a"


def build_report(finalists: pd.DataFrame) -> str:
    timestamp = datetime.now().astimezone()
    lines = []
    lines.append("=" * 78)
    lines.append("DAILY STOCK OPPORTUNITY SCREEN")
    lines.append(f"Generated: {timestamp:%Y-%m-%d %H:%M %Z}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        "Model = 78% technical score (sector-neutralised) + 22% fundamental overlay."
    )
    lines.append(
        "Technical score includes trend R², 52-week high proximity and PEG-based valuation. [v2]"
    )
    lines.append(
        "This is a screening model, not a forecast or guarantee. "
        "A high score can still lose money."
    )
    lines.append("")

    if finalists.empty:
        lines.append("No qualifying stocks were found.")
        return "\n".join(lines)

    overall = finalists.iloc[0]
    lines.append("HIGHEST MODEL SCORE TODAY")
    lines.append(
        f"{overall['ticker']} | {overall.get('company', overall['ticker'])} | "
        f"{overall['market']} | Score {overall['final_score']:.1f}/100 | "
        f"Price {money(overall['price'])}"
    )

    if float(overall["final_score"]) < STRONG_SIGNAL_THRESHOLD:
        lines.append(
            f"NO STRONG SIGNAL: top score is below the configured "
            f"{STRONG_SIGNAL_THRESHOLD:.0f}/100 threshold."
        )

    lines.append("")
    lines.append(
        f"6M return {pct(overall['ret_126'])} | "
        f"3M return {pct(overall['ret_63'])} | "
        f"1M return {pct(overall['ret_21'])} | "
        f"RS vs benchmark {pct(overall['rs_126'])}"
    )
    lines.append(
        f"Above 200DMA {pct(overall['above_sma200'])} | "
        f"63D vol {pct(overall['volatility_63'])} | "
        f"126D max drawdown {pct(overall['max_drawdown_126'])} | "
        f"RSI {overall['rsi_14']:.1f} | "
        f"Trend R² {overall.get('trend_r2', float('nan')):.2f}"
    )
    lines.append("")

    for market in ["CANADA", "US"]:
        lines.append("-" * 78)
        lines.append(f"TOP {TOP_TO_DISPLAY} — {market}")
        lines.append("-" * 78)

        sub = finalists[finalists["market"] == market].nlargest(
            TOP_TO_DISPLAY, "final_score"
        )

        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            # NEW: earnings proximity warning
            warn = ""
            dte = row.get("days_to_earnings")
            if row.get("earnings_near"):
                warn = f" ⚠ EARNINGS IN ~{dte}d"

            lines.append(
                f"{rank}. {row['ticker']} | {row.get('company', row['ticker'])} | "
                f"{row['final_score']:.1f}/100 | Price {money(row['price'])}{warn}"
            )
            lines.append(
                f"   Technical {row['technical_score']:.1f} | "
                f"Fundamental {row['fundamental_score']:.1f} | "
                f"Trend R² {row.get('trend_r2', float('nan')):.2f} | "
                f"6M {pct(row['ret_126'])} | 3M {pct(row['ret_63'])} | "
                f"1M {pct(row['ret_21'])}"
            )
            lines.append(
                f"   Revenue growth {pct(row.get('revenue_growth'))} | "
                f"Earnings growth {pct(row.get('earnings_growth'))} | "
                f"Profit margin {pct(row.get('profit_margin'))} | "
                f"Fwd P/E {row.get('forward_pe') if pd.notna(row.get('forward_pe')) else 'n/a'}"
            )

            heads = recent_headlines(row["ticker"], n=2)
            for h in heads:
                lines.append(f"   • {h}")

        lines.append("")

    lines.append("=" * 78)
    lines.append(
        "Interpretation: use the ranking as a shortlist for further research. "
        "Do not treat the #1 line as an automatic buy instruction."
    )
    return "\n".join(lines)


def save_outputs(finalists: pd.DataFrame, report: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now().strftime("%Y-%m-%d")

    csv_path = REPORT_DIR / f"stock_screen_{date_stamp}.csv"
    txt_path = REPORT_DIR / f"stock_screen_{date_stamp}.txt"

    finalists.to_csv(csv_path, index=False)
    txt_path.write_text(report, encoding="utf-8")

    print(f"\nSaved CSV: {csv_path.resolve()}")
    print(f"Saved report: {txt_path.resolve()}")


# -----------------------------
# MAIN AGENT RUN
# -----------------------------

def run_screen() -> pd.DataFrame:
    print("\nLoading stock universes...")
    universe = get_universe()
    print(f"U.S. universe: {len(universe['US'])} symbols")
    print(f"Canada universe: {len(universe['CANADA'])} symbols")

    print("\nDownloading benchmark data...")
    us_bench = benchmark_returns(download_benchmark("SPY"))
    ca_bench = benchmark_returns(download_benchmark("XIU.TO"))

    print("\nDownloading U.S. price history...")
    us_hist = download_history(universe["US"])

    print("Downloading Canadian price history...")
    ca_hist = download_history(universe["CANADA"])

    rows = []
    print("\nComputing features...")
    for ticker, df in us_hist.items():
        row = make_feature_row(ticker, "US", df, us_bench)
        if row:
            rows.append(row)

    for ticker, df in ca_hist.items():
        row = make_feature_row(ticker, "CANADA", df, ca_bench)
        if row:
            rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No stocks passed the data/liquidity filters.")

    # Score each market separately so Canada is not distorted by U.S. cross-section.
    # NOTE: sector column is populated during enrich_finalists, so the first pass
    # of score_technicals will use a plain cross-sectional rank (sector_col absent).
    # The sector-neutral blend kicks in on the second pass for finalists if desired,
    # but for the primary ranking pass the cross-section is within each market
    # which already provides natural separation.
    scored_parts = []
    for market in ["US", "CANADA"]:
        market_df = features[features["market"] == market].copy()
        if not market_df.empty:
            scored_parts.append(score_technicals(market_df))

    scored = pd.concat(scored_parts, ignore_index=True)

    print("Adding fundamentals to top technical candidates...")
    finalists = enrich_finalists(scored)

    # Re-score with sector now populated so sector-neutral blend is fully active.
    rescored_parts = []
    for market in ["US", "CANADA"]:
        market_df = finalists[finalists["market"] == market].copy()
        if not market_df.empty:
            rescored_parts.append(score_technicals(market_df))
    if rescored_parts:
        finalists = pd.concat(rescored_parts, ignore_index=True)
        finalists["final_score"] = (
            TECHNICAL_FINAL_WEIGHT  * finalists["technical_score"]
            + FUNDAMENTAL_FINAL_WEIGHT * finalists["fundamental_score"]
        )
        finalists = finalists.sort_values(
            "final_score", ascending=False
        ).reset_index(drop=True)

    report = build_report(finalists)
    print("\n" + report)
    save_outputs(finalists, report)

    return finalists


# -----------------------------
# SCHEDULER MODE
# -----------------------------

def run_daemon(hour: int = 8, minute: int = 0) -> None:
    """Keep the Python process alive and run every weekday."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from zoneinfo import ZoneInfo
    except ImportError as exc:
        raise RuntimeError(
            "Daemon mode requires APScheduler. Run: pip install apscheduler"
        ) from exc

    tz = ZoneInfo("America/Chicago")
    scheduler = BlockingScheduler(timezone=tz)

    scheduler.add_job(
        run_screen,
        trigger="cron",
        day_of_week="mon-fri",
        hour=hour,
        minute=minute,
        id="daily_stock_screen",
        max_instances=1,
        coalesce=True,
    )

    print(
        f"Stock agent running. Scheduled Monday-Friday at "
        f"{hour:02d}:{minute:02d} America/Chicago."
    )
    print("Press Ctrl+C to stop.")
    scheduler.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Keep running and execute automatically Monday-Friday.",
    )
    parser.add_argument("--hour",   type=int, default=8)
    parser.add_argument("--minute", type=int, default=0)
    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.hour, args.minute)
    else:
        run_screen()


if __name__ == "__main__":
    main()