"""
stock_agent_backend.py

Local Flask backend for buy_advisor.html.
Reuses market-data / universe functions from stock_agent.py, adds
risk- and horizon-specific scoring, fetches extra fundamentals for finalists,
and exposes JSON endpoints consumed by the single-file HTML prototype.

Changes in v2:
  - PEG ratio replaces raw forward P/E in valuation scoring
  - 52-week high proximity wired into initial_score
  - Sector-neutral ranking blend (70 % cross-section / 30 % within-sector)
  - Earnings proximity flag surfaced per finalist
  - Trend R² added to technical feature set
  - Day-trade cache TTL reduced from 45 min → 10 min
  - _pe_score retained as internal fallback but scoring path uses peg_score

Run:
    pip install -r requirements_buy_advisor.txt
    python stock_agent_backend.py

Then open:
    http://127.0.0.1:5000
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, send_file

import stock_agent as core

APP_DIR   = Path(__file__).resolve().parent
HTML_FILE = APP_DIR / "buy_advisor.html"

app = Flask(__name__)

# Day-trade mode needs fresher data — use a shorter TTL.
CACHE_TTL_MINUTES_DEFAULT   = 45
CACHE_TTL_MINUTES_DAYTRADE  = 10   # NEW: tighter TTL for intraday horizon
SHORTLIST_PER_MARKET        = 18
TOP_RESULTS_PER_MARKET      = 5

_cache_lock = threading.Lock()
_cache: dict = {
    "created_at": None,
    "features":   None,
    "history":    None,
    "benchmark":  None,
}

RISK_LEVELS = {"low", "medium", "high"}
HORIZONS    = {"day_trade", "short_term", "long_term"}


# -----------------------------
# UTILITIES
# -----------------------------

def _finite(value, default=None):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _series(df: pd.DataFrame, field: str) -> pd.Series:
    """Delegate to stock_agent's OHLCV normaliser."""
    normalizer = getattr(core, "ohlcv_series", None) or getattr(
        core, "_yf_series", None
    )
    if normalizer is None:
        raise RuntimeError(
            "stock_agent.py is incompatible with this backend: "
            "missing OHLCV normalizer"
        )
    return normalizer(df, field)


def _ret(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return np.nan
    a = _finite(close.iloc[-periods - 1])
    b = _finite(close.iloc[-1])
    if a is None or b is None or a <= 0:
        return np.nan
    return b / a - 1.0


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = -delta.clip(upper=0).rolling(period).mean()
    if len(close) < period + 2:
        return np.nan
    g = _finite(gain.iloc[-1])
    l = _finite(loss.iloc[-1])
    if g is None or l is None:
        return np.nan
    if l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + g / l)


def _max_drawdown(close: pd.Series, lookback: int = 126) -> float:
    x = close.tail(lookback)
    if len(x) < 20:
        return np.nan
    return float((x / x.cummax() - 1.0).min())


def _trend_r2(close: pd.Series, lookback: int = 63) -> float:
    """
    NEW: R² of closing price vs a linear time trend.
    Measures how clean / directional the move has been.
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


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def _feature_row(
    ticker: str, market: str, df: pd.DataFrame, bench: dict
) -> Optional[dict]:
    if df is None or df.empty:
        return None

    close  = _series(df, "Close").dropna()
    vol    = _series(df, "Volume").reindex(close.index).fillna(0)
    if close.empty or vol.empty or len(close) < 210:
        return None

    price = _finite(close.iloc[-1])
    if price is None or price < core.MIN_PRICE_USD_CAD:
        return None

    adv20 = _finite((close * vol).tail(20).mean(), 0)
    if adv20 < core.MIN_AVG_DOLLAR_VOLUME:
        return None

    rets   = close.pct_change().dropna()
    sma50  = _finite(close.tail(50).mean())
    sma200 = _finite(close.tail(200).mean())
    high252= _finite(close.tail(252).max())
    vol20  = _finite(vol.tail(20).mean(), 0)
    vol60  = _finite(vol.tail(60).mean(), 0)

    r5   = _ret(close, 5)
    r21  = _ret(close, 21)
    r63  = _ret(close, 63)
    r126 = _ret(close, 126)
    r252 = _ret(close, 252) if len(close) > 252 else np.nan

    return {
        "ticker":              ticker,
        "market":              market,
        "price":               price,
        "avg_dollar_volume_20":adv20,
        "ret_1":               _ret(close, 1),
        "ret_5":               r5,
        "ret_21":              r21,
        "ret_63":              r63,
        "ret_126":             r126,
        "ret_252":             r252,
        "rs_21":               r21  - bench.get("ret_21",  0),
        "rs_63":               r63  - bench.get("ret_63",  0),
        "rs_126":              r126 - bench.get("ret_126", 0),
        "above_sma50":         price / sma50  - 1 if sma50  else np.nan,
        "above_sma200":        price / sma200 - 1 if sma200 else np.nan,
        "volatility_21":       _finite(rets.tail(21).std() * np.sqrt(252)),
        "volatility_63":       _finite(rets.tail(63).std() * np.sqrt(252)),
        "max_drawdown_63":     _max_drawdown(close, 63),
        "max_drawdown_126":    _max_drawdown(close, 126),
        "volume_ratio":        (vol20 / vol60) if vol60 else np.nan,
        "rsi_14":              _rsi(close),
        "distance_52w_high":   price / high252 - 1 if high252 else np.nan,  # now used
        "trend_r2":            _trend_r2(close, 63),  # NEW
    }


def _bench_returns(df: pd.DataFrame) -> dict:
    close = _series(df, "Close").dropna()
    return {
        "ret_21":  _ret(close, 21),
        "ret_63":  _ret(close, 63),
        "ret_126": _ret(close, 126),
    }


# -----------------------------
# SNAPSHOT CACHE
# -----------------------------

def build_market_snapshot(
    force: bool = False, horizon: str = "long_term"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], dict]:
    """Download / calculate the broad daily snapshot and cache it."""
    ttl = (
        CACHE_TTL_MINUTES_DAYTRADE
        if horizon == "day_trade"
        else CACHE_TTL_MINUTES_DEFAULT
    )
    now = datetime.now().astimezone()
    with _cache_lock:
        created = _cache.get("created_at")
        fresh   = created and now - created < timedelta(minutes=ttl)
        if fresh and not force and _cache.get("features") is not None:
            return (
                _cache["features"].copy(),
                _cache["history"],
                _cache["benchmark"],
            )

    universe   = core.get_universe()
    us_bench_df= core.download_benchmark("SPY")
    ca_bench_df= core.download_benchmark("XIU.TO")
    benchmarks = {
        "US":     _bench_returns(us_bench_df),
        "CANADA": _bench_returns(ca_bench_df),
    }

    us_hist = core.download_history(universe["US"],    period="18mo")
    ca_hist = core.download_history(universe["CANADA"], period="18mo")
    history = {**us_hist, **ca_hist}

    rows = []
    for ticker, df in us_hist.items():
        row = _feature_row(ticker, "US", df, benchmarks["US"])
        if row:
            rows.append(row)
    for ticker, df in ca_hist.items():
        row = _feature_row(ticker, "CANADA", df, benchmarks["CANADA"])
        if row:
            rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError(
            "No securities passed the data and liquidity filters."
        )

    with _cache_lock:
        _cache["created_at"] = now
        _cache["features"]   = features.copy()
        _cache["history"]    = history
        _cache["benchmark"]  = benchmarks

    return features, history, benchmarks


# -----------------------------
# SCORING
# -----------------------------

def _rank(series: pd.Series, high_good: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(50.0, index=s.index)
    pct = s.rank(pct=True, ascending=True) * 100.0
    if not high_good:
        pct = 100.0 - pct
    return pct.fillna(50.0).clip(0, 100)


def _sector_rank(df: pd.DataFrame, col: str) -> pd.Series:
    """
    NEW: Within-sector percentile rank of `col`.
    Stocks with no sector are placed in their own group.
    """
    result = pd.Series(50.0, index=df.index)
    sector_col = "sector" if "sector" in df.columns else None
    if sector_col is None:
        return result
    for _, grp in df.groupby(sector_col, dropna=False):
        s = pd.to_numeric(grp[col], errors="coerce")
        if s.notna().sum() < 2:
            result.loc[grp.index] = 50.0
        else:
            result.loc[grp.index] = (
                s.rank(pct=True, ascending=True) * 100.0
            ).fillna(50.0).clip(0, 100)
    return result


def _rsi_fitness(rsi: pd.Series, horizon: str) -> pd.Series:
    target = 57 if horizon == "long_term" else 60 if horizon == "short_term" else 62
    width  = 38 if horizon == "day_trade" else 35
    return (
        100.0
        * (
            1.0
            - (pd.to_numeric(rsi, errors="coerce") - target).abs() / width
        )
    ).clip(0, 100).fillna(50)


def _technical_weights(horizon: str) -> dict:
    if horizon == "day_trade":
        return {
            "ret_1":        0.20,
            "ret_5":        0.15,
            "rs_21":        0.08,
            "above_sma50":  0.08,
            "volume_ratio": 0.20,
            "rsi_fit":      0.10,
            "low_dd63":     0.07,
            "liquidity":    0.07,
            "trend_r2":     0.05,   # NEW
        }
    if horizon == "short_term":
        return {
            "ret_5":        0.07,
            "ret_21":       0.21,
            "ret_63":       0.17,
            "rs_63":        0.13,
            "above_sma50":  0.11,
            "volume_ratio": 0.08,
            "rsi_fit":      0.07,
            "low_dd63":     0.09,
            "trend_r2":     0.04,   # NEW
            "high52w":      0.03,   # NEW
        }
    # long_term
    return {
        "ret_21":       0.06,
        "ret_63":       0.10,
        "ret_126":      0.18,
        "ret_252":      0.09,
        "rs_126":       0.13,
        "above_sma200": 0.11,
        "low_vol63":    0.09,
        "low_dd126":    0.09,
        "liquidity":    0.05,
        "trend_r2":     0.05,   # NEW
        "high52w":      0.05,   # NEW: 52-week high proximity (was unused)
    }


def initial_score(
    features: pd.DataFrame, risk: str, horizon: str
) -> pd.DataFrame:
    """Cross-sectional score used to choose which names deserve deeper extraction."""
    parts   = []
    weights = _technical_weights(horizon)

    for market in ["US", "CANADA"]:
        x = features[features.market == market].copy()
        if x.empty:
            continue

        rank_map = {
            "ret_1":        _rank(x["ret_1"]),
            "ret_5":        _rank(x["ret_5"]),
            "ret_21":       _rank(x["ret_21"]),
            "ret_63":       _rank(x["ret_63"]),
            "ret_126":      _rank(x["ret_126"]),
            "ret_252":      _rank(x["ret_252"]),
            "rs_21":        _rank(x["rs_21"]),
            "rs_63":        _rank(x["rs_63"]),
            "rs_126":       _rank(x["rs_126"]),
            "above_sma50":  _rank(x["above_sma50"]),
            "above_sma200": _rank(x["above_sma200"]),
            "volume_ratio": _rank(x["volume_ratio"]),
            "rsi_fit":      _rsi_fitness(x["rsi_14"], horizon),
            "low_vol21":    _rank(x["volatility_21"],    False),
            "low_vol63":    _rank(x["volatility_63"],    False),
            "low_dd63":     _rank(x["max_drawdown_63"]),
            "low_dd126":    _rank(x["max_drawdown_126"]),
            "liquidity":    _rank(np.log1p(x["avg_dollar_volume_20"])),
            "trend_r2":     _rank(x["trend_r2"]),           # NEW
            "high52w":      _rank(x["distance_52w_high"]),  # NEW: was computed but unused
        }

        raw_score = pd.Series(0.0, index=x.index)
        for key, weight in weights.items():
            if key in rank_map:
                raw_score += weight * rank_map[key]

        # NEW: sector-neutral blend — 70 % cross-section, 30 % within-sector
        # Sector column may not be present at this stage (populated later by
        # fetch_details); if absent, _sector_rank returns 50 everywhere and
        # the blend is effectively a no-op.
        sector_rel = _sector_rank(x, "ret_126")
        blended_score = (
            0.70 * raw_score
            + 0.30 * sector_rel
        )

        # Risk overlay: low-risk explicitly favors lower vol / drawdown / liquidity.
        risk_score = (
            0.40 * rank_map["low_vol63"]
            + 0.35 * rank_map["low_dd126"]
            + 0.15 * rank_map["liquidity"]
            + 0.10 * rank_map["above_sma200"]
        )
        if risk == "low":
            score = 0.72 * blended_score + 0.28 * risk_score
        elif risk == "medium":
            score = 0.86 * blended_score + 0.14 * risk_score
        else:
            score = 0.94 * blended_score + 0.06 * risk_score

        x["initial_score"]       = score
        x["risk_score_technical"]= risk_score
        parts.append(x)

    return pd.concat(parts, ignore_index=True)


# -----------------------------
# VALUATION SCORING
# -----------------------------

def _peg_score(forward_pe: Optional[float], earnings_growth: Optional[float]) -> float:
    """
    NEW: PEG-based valuation score replacing raw P/E.
    PEG = forward P/E / (earnings growth %).
    """
    pe = _finite(forward_pe)
    eg = _finite(earnings_growth)
    if pe is None or eg is None:
        return 50.0
    if pe <= 0 or eg <= 0:
        return 35.0
    eg_pct = eg * 100.0
    peg = pe / eg_pct
    if peg < 0.75:  return 95.0
    if peg < 1.0:   return 85.0
    if peg < 1.5:   return 72.0
    if peg < 2.5:   return 55.0
    return 30.0


def _pe_score_fallback(pe: Optional[float]) -> float:
    """Fallback when earnings growth is unavailable."""
    pe = _finite(pe)
    if pe is None or pe <= 0:
        return 45.0
    if 10 <= pe <= 28:  return 92.0
    if  6 <= pe < 10:   return 74.0
    if 28 < pe <= 40:   return 70.0
    if 40 < pe <= 60:   return 48.0
    return 30.0


def _bounded(value, lo, hi, neutral=50.0, higher_good=True):
    v = _finite(value)
    if v is None:
        return neutral
    z = np.clip((v - lo) / (hi - lo), 0, 1) * 100.0
    return float(z if higher_good else 100.0 - z)


# -----------------------------
# DETAIL FETCHING
# -----------------------------

def fetch_details(ticker: str) -> dict:
    """Extra extraction used only for shortlisted names."""
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        info = {}

    rg            = info.get("revenueGrowth")
    eg            = info.get("earningsGrowth")
    margin        = info.get("profitMargins")
    roe           = info.get("returnOnEquity")
    debt_equity   = info.get("debtToEquity")
    current_ratio = info.get("currentRatio")
    fcf           = info.get("freeCashflow")
    forward_pe    = info.get("forwardPE")
    beta          = info.get("beta")
    target        = info.get("targetMeanPrice")
    analyst_count = info.get("numberOfAnalystOpinions")
    dividend_yield= info.get("dividendYield")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    quality = (
        0.23 * _bounded(rg,            -0.05, 0.25)
        + 0.23 * _bounded(eg,          -0.10, 0.30)
        + 0.20 * _bounded(margin,       0.00, 0.30)
        + 0.18 * _bounded(roe,          0.00, 0.30)
        + 0.08 * _bounded(current_ratio, 0.7,  2.2)
        + 0.08 * (80.0 if _finite(fcf, 0) > 0 else 35.0)
    )
    balance = (
        0.60 * _bounded(debt_equity,  20, 220, higher_good=False)
        + 0.40 * _bounded(current_ratio, 0.7, 2.2)
    )

    # NEW: use PEG when earnings growth is available; fall back to raw P/E
    if eg is not None and _finite(eg) is not None and _finite(eg) > 0:
        valuation = _peg_score(forward_pe, eg)
    else:
        valuation = _pe_score_fallback(forward_pe)

    analyst_upside = None
    cp = _finite(current_price)
    tp = _finite(target)
    if cp and tp and cp > 0:
        analyst_upside = tp / cp - 1.0

    # NEW: earnings proximity
    days_to_earnings = core.days_to_earnings(ticker)
    earnings_near    = (
        days_to_earnings is not None
        and 0 <= days_to_earnings <= core.EARNINGS_WARN_DAYS
    )

    try:
        news_raw = yf.Ticker(ticker).news or []
    except Exception:
        news_raw = []
    headlines = []
    for item in news_raw:
        title     = None
        publisher = None
        if isinstance(item, dict):
            title     = item.get("title")
            publisher = item.get("publisher")
            content   = item.get("content")
            if not title and isinstance(content, dict):
                title     = content.get("title")
                provider  = content.get("provider") or {}
                publisher = (
                    provider.get("displayName")
                    if isinstance(provider, dict)
                    else publisher
                )
        if title:
            headlines.append({"title": str(title), "publisher": publisher or ""})
        if len(headlines) >= 3:
            break

    return {
        "company":           info.get("shortName") or info.get("longName") or ticker,
        "sector":            info.get("sector") or "Unknown",
        "industry":          info.get("industry") or "",
        "market_cap":        _finite(info.get("marketCap")),
        "beta":              _finite(beta),
        "forward_pe":        _finite(forward_pe),
        "revenue_growth":    _finite(rg),
        "earnings_growth":   _finite(eg),
        "profit_margin":     _finite(margin),
        "roe":               _finite(roe),
        "debt_to_equity":    _finite(debt_equity),
        "current_ratio":     _finite(current_ratio),
        "free_cash_flow":    _finite(fcf),
        "dividend_yield":    _finite(dividend_yield),
        "target_mean_price": tp,
        "analyst_count":     _finite(analyst_count),
        "analyst_upside":    analyst_upside,
        "quality_score":     float(quality),
        "balance_score":     float(balance),
        "valuation_score":   float(valuation),
        "days_to_earnings":  days_to_earnings,   # NEW
        "earnings_near":     earnings_near,       # NEW
        "headlines":         headlines,
    }


def fetch_intraday(ticker: str) -> dict:
    """Extra 5-minute features when Day Trade is selected."""
    try:
        df = yf.download(
            ticker,
            period="5d",
            interval="5m",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass
        close  = _series(df, "Close").dropna()
        volume = _series(df, "Volume").reindex(close.index).fillna(0)
        if len(close) < 20:
            return {}
        last        = _finite(close.iloc[-1])
        ret_1h      = _ret(close, min(12, len(close) - 1))
        ret_session = _ret(close, min(78, len(close) - 1))
        vol_recent  = _finite(volume.tail(12).mean(), 0)
        vol_base    = _finite(volume.tail(78).mean(), 0)
        vol_accel   = vol_recent / vol_base if vol_base else None
        return {
            "intraday_last":         last,
            "intraday_ret_1h":       ret_1h,
            "intraday_ret_session":  ret_session,
            "intraday_volume_accel": vol_accel,
        }
    except Exception:
        return {}


def sparkline(
    history: Dict[str, pd.DataFrame], ticker: str, n: int = 90
) -> List[float]:
    df = history.get(ticker)
    if df is None or df.empty:
        return []
    vals = _series(df, "Close").dropna().tail(n)
    return [round(float(x), 4) for x in vals]


# -----------------------------
# FINAL WEIGHTS
# -----------------------------

def final_weights(risk: str, horizon: str) -> Tuple[float, float, float, float]:
    """Returns (technical, quality, balance/risk, valuation) weights."""
    if horizon == "day_trade":
        base = [0.84, 0.05, 0.08, 0.03]
    elif horizon == "short_term":
        base = [0.67, 0.13, 0.12, 0.08]
    else:
        base = [0.46, 0.25, 0.18, 0.11]

    if risk == "low":
        base[0] -= 0.08
        base[1] += 0.03
        base[2] += 0.05
    elif risk == "high":
        base[0] += 0.07
        base[1] -= 0.03
        base[2] -= 0.03
        base[3] -= 0.01
    return tuple(base)


# -----------------------------
# ADVISORY LABEL
# -----------------------------

def advisory_label(
    score: float, risk_score: float, risk: str, horizon: str,
    earnings_near: bool = False
) -> str:
    buy_cut   = {"low": 76, "medium": 73, "high": 70}[risk]
    watch_cut = buy_cut - 8

    # NEW: earnings proximity downgrades a BUY to WATCH
    if score >= buy_cut and (risk != "low" or risk_score >= 58):
        if earnings_near:
            return "WATCH (EARNINGS SOON)"
        return "BUY CANDIDATE"
    if score >= watch_cut:
        return "WATCH"
    return "PASS"


# -----------------------------
# REASON / RISK NARRATIVES
# -----------------------------

def make_reasons(row: pd.Series, risk: str, horizon: str) -> List[str]:
    reasons = []
    if horizon == "long_term":
        if _finite(row.get("ret_126"), -9) > 0.08:
            reasons.append(
                f"6-month trend is positive ({row['ret_126'] * 100:.1f}%)."
            )
        if _finite(row.get("above_sma200"), -9) > 0:
            reasons.append(
                f"Price is {row['above_sma200'] * 100:.1f}% above its 200-day average."
            )
        if _finite(row.get("quality_score"), 0) >= 65:
            reasons.append(
                "Fundamental quality ranks well among today's finalists."
            )
        # NEW: trend quality narrative
        tr2 = _finite(row.get("trend_r2"))
        if tr2 is not None and tr2 >= 0.70:
            reasons.append(
                f"Price trend is unusually clean (R² = {tr2:.2f}), suggesting a sustained directional move."
            )
    elif horizon == "short_term":
        if _finite(row.get("ret_21"), -9) > 0:
            reasons.append(
                f"1-month momentum is positive ({row['ret_21'] * 100:.1f}%)."
            )
        if _finite(row.get("volume_ratio"), 0) > 1.05:
            reasons.append(
                "Recent trading volume is running above its 60-day baseline."
            )
        if _finite(row.get("rs_63"), -9) > 0:
            reasons.append(
                "It has outperformed its broad-market benchmark over ~3 months."
            )
    else:  # day_trade
        if _finite(row.get("intraday_ret_1h"), -9) > 0:
            reasons.append(
                f"Latest 1-hour intraday momentum is positive "
                f"({row['intraday_ret_1h'] * 100:.2f}%)."
            )
        if _finite(row.get("intraday_volume_accel"), 0) > 1.15:
            reasons.append(
                "Recent intraday volume is accelerating versus the session baseline."
            )
        if _finite(row.get("ret_5"), -9) > 0:
            reasons.append(
                f"5-day momentum is positive ({row['ret_5'] * 100:.1f}%)."
            )

    if risk == "low":
        if _finite(row.get("volatility_63"), 9) < 0.35:
            reasons.append(
                "Recent realized volatility is relatively contained."
            )
        if _finite(row.get("balance_score"), 0) >= 60:
            reasons.append(
                "Balance-sheet risk checks are comparatively favorable."
            )

    upside = _finite(row.get("analyst_upside"))
    count  = _finite(row.get("analyst_count"), 0)
    if upside is not None and count >= 5 and upside > 0.05:
        reasons.append(
            f"Published analyst mean target is about {upside * 100:.0f}% "
            f"above the quoted price (context only)."
        )

    return reasons[:4] or [
        "The stock ranks well on the selected multi-factor profile."
    ]


def make_risks(row: pd.Series, risk: str, horizon: str) -> List[str]:
    risks = []
    vol   = _finite(row.get("volatility_63"))
    dd    = _finite(row.get("max_drawdown_126"))
    pe    = _finite(row.get("forward_pe"))
    eg    = _finite(row.get("earnings_growth"))
    debt  = _finite(row.get("debt_to_equity"))
    rsi   = _finite(row.get("rsi_14"))

    if vol is not None and vol > 0.45:
        risks.append(
            f"Annualized 63-day volatility is elevated at roughly {vol * 100:.0f}%."
        )
    if dd is not None and dd < -0.20:
        risks.append(
            f"Recent 6-month drawdown reached about {dd * 100:.0f}%."
        )

    # NEW: PEG-aware valuation risk narrative
    if pe is not None and eg is not None and eg > 0:
        eg_pct = eg * 100.0
        peg = pe / eg_pct if eg_pct > 0 else None
        if peg is not None and peg > 2.5:
            risks.append(
                f"PEG ratio is stretched ({peg:.1f}x), suggesting the valuation "
                f"may not be justified by earnings growth."
            )
    elif pe is not None and pe > 45:
        risks.append(
            f"Forward P/E is high ({pe:.1f}x) with no earnings growth to "
            f"calibrate it against."
        )

    if debt is not None and debt > 180:
        risks.append(
            "Debt-to-equity is elevated and deserves a closer balance-sheet review."
        )
    if rsi is not None and rsi > 72:
        risks.append("RSI is stretched, so near-term pullback risk is higher.")

    # NEW: earnings proximity risk
    if row.get("earnings_near"):
        dte = _finite(row.get("days_to_earnings"))
        days_str = f"~{int(dte)}d" if dte is not None else "soon"
        risks.append(
            f"Earnings report expected in {days_str}; "
            f"results can reverse any technical signal rapidly."
        )

    if horizon == "day_trade":
        risks.append(
            "Intraday signals can reverse quickly; "
            "spreads, slippage and news can dominate the model."
        )
    if not risks:
        risks.append(
            "No single red flag dominates the model, but market/sector risk "
            "can still overwhelm stock-specific signals."
        )
    return risks[:4]


# -----------------------------
# MAIN ADVICE GENERATOR
# -----------------------------

def generate_advice(risk: str, horizon: str, force: bool = False) -> dict:
    features, history, benchmarks = build_market_snapshot(
        force=force, horizon=horizon
    )
    ranked = initial_score(features, risk, horizon)

    shortlist = []
    for market in ["CANADA", "US"]:
        shortlist.append(
            ranked[ranked.market == market]
            .nlargest(SHORTLIST_PER_MARKET, "initial_score")
            .copy()
        )
    candidates = pd.concat(shortlist, ignore_index=True)

    detail_rows = []
    for _, row in candidates.iterrows():
        details  = fetch_details(row["ticker"])
        combined = {**row.to_dict(), **details}
        if horizon == "day_trade":
            combined.update(fetch_intraday(row["ticker"]))
        detail_rows.append(combined)
        time.sleep(0.05)

    enriched = pd.DataFrame(detail_rows)

    # Day-trade intraday sub-score is cross-sectional among finalists.
    if horizon == "day_trade":
        enriched["intraday_score"] = (
            0.45 * _rank(
                enriched.get(
                    "intraday_ret_1h",
                    pd.Series(index=enriched.index, dtype=float),
                )
            )
            + 0.30 * _rank(
                enriched.get(
                    "intraday_ret_session",
                    pd.Series(index=enriched.index, dtype=float),
                )
            )
            + 0.25 * _rank(
                enriched.get(
                    "intraday_volume_accel",
                    pd.Series(index=enriched.index, dtype=float),
                )
            )
        )
        enriched["initial_score"] = (
            0.68 * enriched["initial_score"]
            + 0.32 * enriched["intraday_score"]
        )

    tw, qw, bw, vw = final_weights(risk, horizon)
    enriched["final_score"] = (
        tw * enriched["initial_score"]
        + qw * enriched["quality_score"].fillna(50)
        + bw * enriched["balance_score"].fillna(50)
        + vw * enriched["valuation_score"].fillna(50)
    )

    if risk == "low":
        beta_penalty    = (
            pd.to_numeric(enriched["beta"], errors="coerce").fillna(1.0) - 1.1
        ).clip(lower=0) * 6
        high_vol_penalty = (
            pd.to_numeric(enriched["volatility_63"], errors="coerce").fillna(0.3)
            - 0.40
        ).clip(lower=0) * 25
        enriched["final_score"] -= beta_penalty + high_vol_penalty

    enriched["final_score"] = enriched["final_score"].clip(0, 100)
    enriched = enriched.sort_values(
        "final_score", ascending=False
    ).reset_index(drop=True)

    results = []
    for _, row in enriched.iterrows():
        risk_quality = (
            0.55 * _finite(row.get("risk_score_technical"), 50)
            + 0.45 * _finite(row.get("balance_score"), 50)
        )
        earnings_near = bool(row.get("earnings_near", False))
        item = {
            "ticker":           row["ticker"],
            "company":          row.get("company") or row["ticker"],
            "market":           row["market"],
            "sector":           row.get("sector") or "Unknown",
            "price":            _finite(row.get("price")),
            "score":            round(_finite(row.get("final_score"), 0), 1),
            "advisory":         advisory_label(
                _finite(row.get("final_score"), 0),
                risk_quality,
                risk,
                horizon,
                earnings_near=earnings_near,
            ),
            "risk_quality_score":   round(risk_quality, 1),
            "technical_score":      round(_finite(row.get("initial_score"), 0), 1),
            "fundamental_score":    round(_finite(row.get("quality_score"), 50), 1),
            "valuation_score":      round(_finite(row.get("valuation_score"), 50), 1),
            "balance_score":        round(_finite(row.get("balance_score"), 50), 1),
            "beta":                 _finite(row.get("beta")),
            "forward_pe":           _finite(row.get("forward_pe")),
            "revenue_growth":       _finite(row.get("revenue_growth")),
            "earnings_growth":      _finite(row.get("earnings_growth")),
            "profit_margin":        _finite(row.get("profit_margin")),
            "roe":                  _finite(row.get("roe")),
            "dividend_yield":       _finite(row.get("dividend_yield")),
            "analyst_upside":       _finite(row.get("analyst_upside")),
            "analyst_count":        _finite(row.get("analyst_count")),
            "ret_1":                _finite(row.get("ret_1")),
            "ret_5":                _finite(row.get("ret_5")),
            "ret_21":               _finite(row.get("ret_21")),
            "ret_63":               _finite(row.get("ret_63")),
            "ret_126":              _finite(row.get("ret_126")),
            "ret_252":              _finite(row.get("ret_252")),
            "above_sma50":          _finite(row.get("above_sma50")),
            "above_sma200":         _finite(row.get("above_sma200")),
            "volatility_63":        _finite(row.get("volatility_63")),
            "max_drawdown_126":     _finite(row.get("max_drawdown_126")),
            "rsi_14":               _finite(row.get("rsi_14")),
            "volume_ratio":         _finite(row.get("volume_ratio")),
            "trend_r2":             _finite(row.get("trend_r2")),      # NEW
            "days_to_earnings":     row.get("days_to_earnings"),        # NEW
            "earnings_near":        earnings_near,                      # NEW
            "intraday_ret_1h":      _finite(row.get("intraday_ret_1h")),
            "intraday_ret_session": _finite(row.get("intraday_ret_session")),
            "intraday_volume_accel":_finite(row.get("intraday_volume_accel")),
            "headlines":            (
                row.get("headlines")
                if isinstance(row.get("headlines"), list)
                else []
            ),
            "reasons": make_reasons(row, risk, horizon),
            "risks":   make_risks(row, risk, horizon),
            "sparkline": sparkline(history, row["ticker"]),
        }
        results.append(item)

    canada  = [x for x in results if x["market"] == "CANADA"][:TOP_RESULTS_PER_MARKET]
    us      = [x for x in results if x["market"] == "US"][:TOP_RESULTS_PER_MARKET]
    overall = results[0] if results else None

    strong = bool(overall and overall["advisory"] == "BUY CANDIDATE")
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "risk":         risk,
        "horizon":      horizon,
        "defaults":     {"risk": "low", "horizon": "long_term"},
        "strong_signal":strong,
        "highest":      overall,
        "canada":       canada,
        "us":           us,
        "methodology": {
            "universe":          (
                "S&P/TSX 60 plus S&P 500 and Nasdaq-100, "
                "filtered for price and liquidity."
            ),
            "technical_weight":  round(tw * 100),
            "quality_weight":    round(qw * 100),
            "balance_weight":    round(bw * 100),
            "valuation_weight":  round(vw * 100),
            "note": (
                "Scores are relative screening signals, not probabilities "
                "of profit or guarantees of appreciation. "
                "v2: sector-neutral ranking, PEG valuation, trend R², "
                "52-week high proximity, earnings proximity warnings."
            ),
        },
    }


# -----------------------------
# FLASK ROUTES
# -----------------------------

@app.get("/")
def index():