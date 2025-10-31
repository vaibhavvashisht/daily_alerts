#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Daily Alerts to Discord (Screener + Best-Strategy Signals)
- Pre-market / Early / After-hours runs
- Uses optimizer outputs when available (symbol_strategy_map.csv)
- Falls back to solid defaults if not present
Requires: pandas numpy yfinance requests python-dotenv
"""

import os, math, json, time, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

# ------------------------------
# Config / Paths
# ------------------------------
ROOT = Path.cwd()
MAP_CSV = ROOT / "symbol_strategy_map.csv"       # from your matcher/optimizer
SCREEN_CSV = ROOT / "screened_symbols.csv"       # optional, if you want to inspect
LOG_CSV = ROOT / "alerts_log.csv"                # append-only audit
UNIVERSE_DEFAULT = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOG","PANW","FTNT","SMCI","TSM",
    "ON","ADBE","AMD","AVGO","CRM","ORCL","INTC","LRCX","HUBS","MDB",
    "SHOP","NOW","PLTR","MU","KLAC","SNOW","DDOG","NET","QQQ","SPY"
]
MIN_AVG_VOL = 5_00_000  # daily average volume filter
MIN_PRICE   = 5.0

# ------------------------------
# ENV
# ------------------------------
load_dotenv(ROOT/".env")
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
EQUITY  = float(os.getenv("EQUITY", "100000"))
RISK_PCT= float(os.getenv("RISK_PER_TRADE", "0.01"))

if not WEBHOOK:
    raise SystemExit("DISCORD_WEBHOOK_URL missing in .env")

# ------------------------------
# Utilities
# ------------------------------
def post_discord(title: str, rows: list[dict], run_tag: str):
    """Send a compact embed to Discord."""
    if not rows:
        content = f"**{title}**\nNo qualifying setups."
        requests.post(WEBHOOK, json={"content": content}, timeout=15)
        return

    # Discord embed fields (max 25 fields per embed)
    fields = []
    for r in rows[:25]:
        name = f"{r['symbol']} · {r['framework']}"
        val  = (
            f"Trigger: **{r['trigger']}**  \n"
            f"Price: {r['price']:.2f} | Stop: {r['stop']:.2f} | Target: {r['target']:.2f}  \n"
            f"R:R ≈ {r['rr']:.2f} | Size: {r['size']} | Risk ${r['risk_amt']:.0f}"
        )
        fields.append({"name": name, "value": val, "inline": False})

    embed = {
        "title": title,
        "description": f"{run_tag} • {dt.datetime.now().strftime('%Y-%m-%d %H:%M %Z')}",
        "color": 5814783,
        "fields": fields
    }
    requests.post(WEBHOOK, json={"embeds": [embed]}, timeout=20)

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def highest(series, n):
    return series.rolling(n).max()

def keltner(df, n=20, mult=1.5):
    m = df["Close"].rolling(n).mean()
    a = atr(df, 10)  # shorter ATR for channel width
    return m - mult*a, m + mult*a  # lower, upper

def dollar_risk_and_size(price, stop, equity=EQUITY, risk_pct=RISK_PCT):
    risk_amt = equity * risk_pct
    per_share = max(price - stop, 0.01)
    size = int(risk_amt / per_share)
    return risk_amt, max(size, 1)

def read_map():
    """Load best params per symbol if optimizer map exists."""
    if not MAP_CSV.exists():
        return {}
    m = pd.read_csv(MAP_CSV)
    # expected columns: symbol, ema_fast, ema_slow, atr_period, stop_k, tp_k, risk, max_hold_days
    out = {}
    for _, r in m.iterrows():
        out[str(r["symbol"]).upper()] = dict(
            ema_fast=int(r.get("ema_fast", 12)),
            ema_slow=int(r.get("ema_slow", 33)),
            atr_period=int(r.get("atr_period", 8)),
            stop_k=float(r.get("stop_k", 1.28)),
            tp_k=float(r.get("tp_k", 4.0)),
            max_hold_days=int(r.get("max_hold_days", 0)),
        )
    return out

def screener(universe=UNIVERSE_DEFAULT, limit=25, window_days=504):
    """Volume + momentum screener to pick a focused universe."""
    rows=[]
    end = dt.date.today()
    start = end - dt.timedelta(days=window_days)
    for sym in universe:
        try:
            df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if df is None or df.empty or len(df) < 60:
                continue
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            if df["Close"].iloc[-1] < MIN_PRICE: 
                continue
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            if (avg_vol or 0) < MIN_AVG_VOL: 
                continue
            mom = (df["Close"].iloc[-1] / df["Close"].rolling(20).mean().iloc[-1]) - 1.0
            surge = df["Volume"].iloc[-1] / max(avg_vol,1)
            score = float(mom) + float(surge)
            rows.append({"symbol": sym, "score": score, "avg_vol": avg_vol})
        except Exception:
            continue
    s = pd.DataFrame(rows).sort_values("score", ascending=False).head(limit)
    if not s.empty:
        s.to_csv(SCREEN_CSV, index=False)
    return [x for x in s["symbol"].tolist()] if not s.empty else universe[:limit]

# ------------------------------
# Signal engines (daily)
# ------------------------------
def signals_for_symbol(sym, cfg):
    """
    Returns list[dict] of actionable entries for BEST framework:
      - TREND: EMA(ema_fast > ema_slow) recent cross-up
      - BREAKOUT: close > prior 20-day high
      - REVERSION: close bounced above Keltner lower band (mean-revert)
    We compute all 3; then choose the best-scoring *today*.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=max(365, cfg["atr_period"]*25))
    df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty or len(df) < 60:
        return []

    df = df[["Open","High","Low","Close","Volume"]].dropna().copy()
    df["ema_f"] = ema(df["Close"], cfg["ema_fast"])
    df["ema_s"] = ema(df["Close"], cfg["ema_slow"])
    df["atr"]   = atr(df, cfg["atr_period"])
    df["hi20"]  = highest(df["Close"], 20)
    kel_lo, kel_hi = keltner(df, n=20, mult=1.5)
    df["kel_lo"], df["kel_hi"] = kel_lo, kel_hi

    # today's row
    last = df.iloc[-1]
    prev = df.iloc[-2]

    out = []

    # --- Trend (EMA cross-up) ---
    cross_up = (last["ema_f"] > last["ema_s"]) and (prev["ema_f"] <= prev["ema_s"])
    if cross_up:
        stop = float(last["Close"] - cfg["stop_k"]*last["atr"])
        target = float(last["Close"] + cfg["tp_k"]*last["atr"])
        risk_amt, size = dollar_risk_and_size(last["Close"], stop)
        rr = (target - last["Close"]) / max(last["Close"] - stop, 0.01)
        out.append(dict(symbol=sym, framework="TREND", trigger="ema_cross_up",
                        price=float(last["Close"]), stop=stop, target=target,
                        rr=round(rr,2), risk_amt=risk_amt, size=size))

    # --- Breakout (20-day) ---
    breakout = last["Close"] > df["hi20"].iloc[-2]  # surpass prior 20d high
    if breakout:
        stop = float(last["Close"] - cfg["stop_k"]*last["atr"])
        target = float(last["Close"] + cfg["tp_k"]*last["atr"])
        risk_amt, size = dollar_risk_and_size(last["Close"], stop)
        rr = (target - last["Close"]) / max(last["Close"] - stop, 0.01)
        out.append(dict(symbol=sym, framework="BREAKOUT", trigger="breakout20",
                        price=float(last["Close"]), stop=stop, target=target,
                        rr=round(rr,2), risk_amt=risk_amt, size=size))

    # --- Reversion (bounce off Keltner lower) ---
    bounce = (prev["Close"] < prev["kel_lo"]) and (last["Close"] > last["kel_lo"])
    if bounce:
        stop = float(last["Close"] - cfg["stop_k"]*last["atr"])
        target = float(last["Close"] + cfg["tp_k"]*last["atr"])
        risk_amt, size = dollar_risk_and_size(last["Close"], stop)
        rr = (target - last["Close"]) / max(last["Close"] - stop, 0.01)
        out.append(dict(symbol=sym, framework="REVERSION", trigger="keltner_bounce",
                        price=float(last["Close"]), stop=stop, target=target,
                        rr=round(rr,2), risk_amt=risk_amt, size=size))

    # choose the “best” one (highest R:R, then volume proxy via risk_amt is same; keep R:R)
    out.sort(key=lambda x: (-x["rr"], -x["risk_amt"]))
    return out[:1]  # single best suggestion per symbol for noise reduction

def make_alerts(run_tag: str, universe=None, limit=20):
    universe = universe or UNIVERSE_DEFAULT
    best_params = read_map()
    screened = screener(universe, limit=limit, window_days=504)
    results = []
    for sym in screened:
        cfg = best_params.get(sym, dict(ema_fast=12, ema_slow=33, atr_period=8, stop_k=1.28, tp_k=4.0, max_hold_days=0))
        rows = signals_for_symbol(sym, cfg)
        results.extend(rows)
        # audit log
        for r in rows:
            write_log({**r, "run_tag": run_tag})
        time.sleep(0.1)  # be nice to API
    # keep the top 10 by score (R:R), then cap per your “max 1–2 conviction trades” later if you like
    results.sort(key=lambda x: (-x["rr"], -x["risk_amt"]))
    return results[:10]

def write_log(row: dict):
    header = not LOG_CSV.exists()
    df = pd.DataFrame([row])
    df.to_csv(LOG_CSV, mode="a", header=header, index=False)

# ------------------------------
# Entrypoints (time-of-day tags)
# ------------------------------
def premarket():
    rows = make_alerts("PRE-MKT", limit=25)
    post_discord("Pre-Market Setups (daily signals)", rows, run_tag="PRE-MKT")

def early_session():
    rows = make_alerts("EARLY", limit=20)
    post_discord("Early Session Setups (confirmed after open)", rows, run_tag="EARLY")

def after_hours():
    rows = make_alerts("AFTER", limit=25)
    post_discord("After-Hours Recap / New Setups", rows, run_tag="AFTER")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--when", choices=["premarket","early","after"], required=True)
    p.add_argument("--limit", type=int, default=25)
    args = p.parse_args()

    if args.when == "premarket":
        premarket()
    elif args.when == "early":
        early_session()
    else:
        after_hours()
