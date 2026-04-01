# =============================================================================
# backend/core/data_fetcher.py — Multi-Timeframe Data + Live Price
# =============================================================================
import ccxt
import pandas as pd
import numpy as np
import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import *

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches multi-timeframe OHLCV data and live ticker from Binance."""

    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch: Dict[str, float] = {}
        self._cache_ttl = {"1m": 30, "5m": 60, "15m": 120, "1h": 300, "4h": 900}
        self._live_price: Optional[float] = None
        self._live_ts: float = 0

    # ── Live price ────────────────────────────────────────────────────────────
    def get_live_price(self) -> dict:
        """Fetch live ticker — returns price, bid, ask, 24h change."""
        try:
            ticker = self.exchange.fetch_ticker(SYMBOL)
            self._live_price = ticker["last"]
            self._live_ts    = time.time()
            return {
                "price"    : round(ticker["last"], 2),
                "bid"      : round(ticker["bid"], 2),
                "ask"      : round(ticker["ask"], 2),
                "change24h": round(ticker.get("percentage", 0), 3),
                "high24h"  : round(ticker.get("high", 0), 2),
                "low24h"   : round(ticker.get("low", 0), 2),
                "volume24h": round(ticker.get("baseVolume", 0), 2),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Live price error: {e}")
            return {"price": self._live_price or 0, "error": str(e)}

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    def fetch_ohlcv(self, timeframe: str, limit: int = None,
                    force: bool = False) -> pd.DataFrame:
        """Fetch OHLCV with caching."""
        limit = limit or CANDLE_LIMITS.get(timeframe, 300)
        now   = time.time()
        ttl   = self._cache_ttl.get(timeframe, 60)

        if (not force and timeframe in self._cache
                and now - self._last_fetch.get(timeframe, 0) < ttl):
            return self._cache[timeframe]

        try:
            raw = self.exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
            df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)
            df = df[~df.index.duplicated()].sort_index()
            # Drop last incomplete candle
            df = df.iloc[:-1]
            self._cache[timeframe]      = df
            self._last_fetch[timeframe] = now
            return df
        except Exception as e:
            logger.error(f"OHLCV fetch error [{timeframe}]: {e}")
            return self._cache.get(timeframe, pd.DataFrame())

    def fetch_all_timeframes(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch all configured timeframes."""
        result = {}
        for tf in TIMEFRAMES:
            df = self.fetch_ohlcv(tf, force=force)
            if not df.empty:
                result[tf] = df
        return result

    # ── Order book (liquidity) ────────────────────────────────────────────────
    def fetch_orderbook(self, depth: int = 20) -> dict:
        """Fetch order book for liquidity analysis."""
        try:
            ob = self.exchange.fetch_order_book(SYMBOL, depth)
            bids = np.array(ob["bids"][:depth])
            asks = np.array(ob["asks"][:depth])
            bid_wall = float(bids[np.argmax(bids[:, 1]), 0]) if len(bids) else 0
            ask_wall = float(asks[np.argmax(asks[:, 1]), 0]) if len(asks) else 0
            return {
                "bid_wall"   : bid_wall,
                "ask_wall"   : ask_wall,
                "spread"     : round(asks[0, 0] - bids[0, 0], 2) if len(asks) and len(bids) else 0,
                "bid_volume" : round(float(bids[:, 1].sum()), 2),
                "ask_volume" : round(float(asks[:, 1].sum()), 2),
                "imbalance"  : round(float(bids[:, 1].sum()) /
                               max(float(asks[:, 1].sum()), 1), 3),
            }
        except Exception as e:
            logger.error(f"Orderbook error: {e}")
            return {}


# Singleton
_fetcher = None
def get_fetcher() -> DataFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher
