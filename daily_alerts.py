#!/usr/bin/env python3
"""
Daily Alerts Bot — Screener + Discord Poster (v2)
-------------------------------------------------
Resilient version with robust handling for empty screeners.
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv

# ---------------------------------------
# ENV + CONFIG
# ---------------------------------------
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

DEFAULT_LIMIT = int(os.getenv("LIMIT", 25))
DEFAULT_WINDOW = int(os.getenv("WINDOW_DAYS", 504))

UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOG","PANW","FTNT","SMCI","TSM",
    "ON","ADBE","AMD","AVGO","CRM","ORCL","INTC","LRCX","HUBS","MDB",
    "SHOP","NOW","PLTR","MU","KLAC","SNOW","DDOG","NET"
]

# ---------------------------------------
# HELPERS
# ---------------------------------------

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    rename = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "adj close": "Adj Close", "volume": "Volume"
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def screener(universe, limit=20, window_days=504):
    rows = []
    for sym in universe:
        try:
            df = yf.download(
                sym, period=f"{max(60, window_days)}d", interval="1d",
                auto_adjust=True, progress=False, group_by="column", threads=False
            )
            if df is None or df.empty:
                continue
            df = _normalize_cols(df)
            if not {"Close", "Volume"}.issubset(df.columns):
                continue

            close = float(df["Close"].iloc[-1])
            vol20 = float(df["Volume"].rolling(20, min_periods=1).mean().iloc[-1] or 0.0)
            lastvol = float(df["Volume"].iloc[-1] or 0.0)
            ma20 = float(df["Close"].rolling(20, min_periods=1).mean().iloc[-1] or 0.0)

            vol_surge = (lastvol / vol20) if vol20 > 0 else 0.0
            momentum = (close / ma20 - 1.0) if ma20 > 0 else 0.0

            # ATR(14)
            if {"High", "Low", "Close"}.issubset(df.columns):
                high, low, cls = df["High"], df["Low"], df["Close"]
                tr = pd.concat([
                    (high - low).abs(),
                    (high - cls.shift()).abs(),
                    (low - cls.shift()).abs()
                ], axis=1).max(axis=1)
                atr14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
            else:
                atr14 = 0.0

            score = float(vol_surge + momentum)
            rows.append({
                "symbol": sym,
                "score": score,
                "price": close,
                "vol_surge": vol_surge,
                "momentum": momentum,
                "atr": atr14
            })
        except Exception:
            continue

    if not rows:
        return []  # return safely

    s = pd.DataFrame(rows)
    if "score" not in s.columns:
        s["score"] = s.get("vol_surge", 0) + s.get("momentum", 0)
    s = s.sort_values("score", ascending=False, na_position="last").head(limit)
    return s.to_dict("records")


def make_alert_lines(screened, title):
    if not screened:
        return [f"**{title}** — no qualified tickers today."]
    lines = [f"**{title}**"]
    for r in screened:
        lines.append(
            f"- `{r['symbol']}`  score: **{r['score']:.3f}**  "
            f"price: {r['price']:.2f}  vol×: {r['vol_surge']:.2f}  "
            f"mom: {r['momentum']:.2%}  ATR14: {r['atr']:.2f}"
        )
    return lines


def make_alerts(phase: str, limit: int = 20, window_days: int = 504):
    screened = screener(UNIVERSE, limit=limit, window_days=window_days)
    return make_alert_lines(screened, f"{phase} — Screened top {limit}")


import time
import requests

# keep using your existing global: DISCORD_WEBHOOK_URL

def _chunk_lines(lines, max_len=1900):
    """
    Pack lines into chunks with total length <= max_len (leave headroom for numbering).
    Prefers breaking at line boundaries so Discord gets multiple messages cleanly.
    """
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        # hard cap single line to avoid surprises
        ln = ln.strip()
        if len(ln) > max_len:
            ln = ln[:max_len - 3] + "..."
        add = (len(ln) + 1)  # +1 for newline
        if cur_len + add > max_len and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += add
    if cur:
        chunks.append("\n".join(cur))
    return chunks

def send_discord_message(content_lines, title=None, sleep_between=0.6):
    """
    Post long messages to Discord in safe chunks (<2000 chars).
    If DISCORD_WEBHOOK_URL is missing, we print to logs and return.
    """
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL not set; printing output instead:")
        print("\n".join(content_lines))
        return

    # Optional header line
    lines = []
    if title:
        lines.append(f"**{title}**")
    lines.extend(content_lines)

    # Build chunks comfortably under the 2000 limit
    payloads = _chunk_lines(lines, max_len=1900)  # 1900 leaves room for numbering

    for i, body in enumerate(payloads, 1):
        numbered = f"(part {i}/{len(payloads)})\n{body}" if len(payloads) > 1 else body
        payload = {"content": numbered}
        try:
            r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=20)
            print(f"[DEBUG] Discord post {i}/{len(payloads)} -> {r.status_code}")
            if r.status_code >= 400:
                print(f"[ERROR] Discord webhook failed: {r.status_code} {r.text[:200]}")
                # Don’t bail mid-stream; try to post remaining chunks, but you can break if desired
            time.sleep(sleep_between)  # mild pacing to avoid rate-limits
        except Exception as e:
            print(f"[ERROR] Failed to send Discord message (part {i}): {e}")



# ---------------------------------------
# PHASES
# ---------------------------------------

def premarket():
    print("▶ Running premarket screener...")
    alerts = make_alerts("PREMARKET", limit=DEFAULT_LIMIT, window_days=DEFAULT_WINDOW)
    send_discord_message(alerts)


def early_session():
    print("▶ Running early-session screener...")
    alerts = make_alerts("EARLY SESSION", limit=DEFAULT_LIMIT, window_days=DEFAULT_WINDOW)
    send_discord_message(alerts)


def after_hours():
    print("▶ Running after-hours screener...")
    alerts = make_alerts("AFTER HOURS", limit=DEFAULT_LIMIT, window_days=DEFAULT_WINDOW)
    send_discord_message(alerts)


# ---------------------------------------
# ENTRY POINT
# ---------------------------------------
if __name__ == "__main__":
    phase = "early"
    limit = DEFAULT_LIMIT
    window_days = DEFAULT_WINDOW

    # CLI args
    if "--when" in sys.argv:
        phase = sys.argv[sys.argv.index("--when") + 1]
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if "--window-days" in sys.argv:
        window_days = int(sys.argv[sys.argv.index("--window-days") + 1])

    print(f"Running alerts phase={phase} limit={limit} window={window_days}")

    try:
        if phase == "premarket":
            premarket()
        elif phase == "early":
            early_session()
        elif phase == "after":
            after_hours()
        else:
            print(f"Unknown phase: {phase}")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        send_discord_message([f"**{phase.upper()}** — encountered error: {e}"])
        sys.exit(0)
