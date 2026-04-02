# backend/core/data_fetcher.py

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import *

logger = logging.getLogger(__name__)


class DataFetcher:

    def __init__(self):
        # ✅ SWITCH EXCHANGE (FIX BINANCE ISSUE)
        self.exchange = ccxt.bybit({
            "enableRateLimit": True,
        })

        self._cache = {}
        self._last_fetch = {}

    # ── LIVE PRICE WITH RETRY ─────────────────────────────
    def get_live_price(self):
        for _ in range(3):
            try:
                ticker = self.exchange.fetch_ticker(SYMBOL)
                return {
                    "price": round(ticker["last"], 2),
                    "bid": round(ticker["bid"], 2),
                    "ask": round(ticker["ask"], 2),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            except Exception as e:
                time.sleep(1)

        logger.error("Live price failed")
        return {"price": 0}

    # ── 🔥 PAGINATED DATA FETCH (10k+) ─────────────────────
    def fetch_ohlcv(self, timeframe: str, limit: int = None, force=False):

        # ✅ BIG DATA LIMITS
        if timeframe == "1m":
            total_limit = 10000
        elif timeframe == "5m":
            total_limit = 5000
        elif timeframe == "15m":
            total_limit = 2000
        else:
            total_limit = 1000

        all_data = []
        since = None

        try:
            while len(all_data) < total_limit:
                batch = self.exchange.fetch_ohlcv(
                    SYMBOL,
                    timeframe,
                    since=since,
                    limit=1000
                )

                if not batch:
                    break

                all_data = batch + all_data
                since = batch[0][0] - 1

                if len(batch) < 1000:
                    break

                time.sleep(0.2)

            df = pd.DataFrame(
                all_data,
                columns=["ts", "open", "high", "low", "close", "volume"]
            )

            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]   # 🔥 REMOVE DUPLICATES
            df = df.iloc[-total_limit:]

            # ✅ ADD SESSION FEATURE
            df["hour"] = df.index.hour
            df["session_london"] = ((df["hour"] >= 11) & (df["hour"] <= 13)).astype(int)
            df["session_ny"] = ((df["hour"] >= 17) & (df["hour"] <= 20)).astype(int)

            return df

        except Exception as e:
            logger.error(f"OHLCV fetch error [{timeframe}]: {e}")
            return pd.DataFrame()

    def fetch_all_timeframes(self, force=False) -> Dict[str, pd.DataFrame]:
        result = {}
        for tf in TIMEFRAMES:
            df = self.fetch_ohlcv(tf)
            if not df.empty:
                result[tf] = df
        return result

    def fetch_orderbook(self, depth=20):
        try:
            ob = self.exchange.fetch_order_book(SYMBOL, depth)
            return {
                "bid_volume": sum([b[1] for b in ob["bids"]]),
                "ask_volume": sum([a[1] for a in ob["asks"]]),
            }
        except:
            return {}


_fetcher = None
def get_fetcher():
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher