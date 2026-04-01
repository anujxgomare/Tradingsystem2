# =============================================================================
# backend/core/indicators.py — FIXED VERSION (PRODUCTION SAFE)
# =============================================================================
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Dict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import *

logger = logging.getLogger(__name__)


class IndicatorEngine:

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        try:
            d = self._ema(d)
            d = self._rsi(d)
            d = self._macd(d)
            d = self._bollinger(d)
            d = self._atr(d)
            d = self._vwap(d)
            d = self._volume(d)
            d = self._structure(d)
            d = self._smart_money(d)

            # 🔥 FINAL CLEANING (CRITICAL FIX)
            d.replace([np.inf, -np.inf], 0, inplace=True)
            d.fillna(0, inplace=True)

        except Exception as e:
            logger.error(f"Indicator error: {e}")

        return d


    def compute_all(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {tf: self.compute(df) for tf, df in dfs.items() if df is not None and not df.empty}


    # ── EMA ───────────────────────────────────────────────────────────────
    def _ema(self, d):
        for p in EMA_PERIODS:
            d[f"ema_{p}"] = ta.ema(d["close"], length=p)

        c = d["close"].fillna(0)
        e20 = d["ema_20"].fillna(0)
        e50 = d["ema_50"].fillna(0)
        e200 = d["ema_200"].fillna(0)

        d["ema_bullish"] = ((c > e20) & (e20 > e50) & (e50 > e200)).astype(float)
        d["ema_bearish"] = ((c < e20) & (e20 < e50) & (e50 < e200)).astype(float)
        d["ema_score"] = d["ema_bullish"] - d["ema_bearish"]

        d["ema50_dist"] = (c - e50) / (c + 1e-10)
        return d


    # ── RSI ───────────────────────────────────────────────────────────────
    def _rsi(self, d):
        d["rsi"] = ta.rsi(d["close"], length=RSI_PERIOD)
        d["rsi"] = d["rsi"].fillna(50)

        d["rsi_sma"] = d["rsi"].rolling(5).mean().fillna(50)

        d["rsi_score"] = np.where(d["rsi"] < 35, 1.0,
                         np.where(d["rsi"] > 65, -1.0,
                         np.where(d["rsi"] > 55, 0.4,
                         np.where(d["rsi"] < 45, -0.4, 0.0))))
        return d


    # ── MACD ───────────────────────────────────────────────────────────────
    def _macd(self, d):
        m = ta.macd(d["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)

        if m is not None and not m.empty:
            d["macd_line"] = m.iloc[:, 0]
            d["macd_signal"] = m.iloc[:, 1]
            d["macd_hist"] = m.iloc[:, 2]
        else:
            d["macd_line"] = d["macd_signal"] = d["macd_hist"] = 0

        d["macd_hist"] = d["macd_hist"].fillna(0)

        d["macd_cross_bull"] = ((d["macd_hist"] > 0) & (d["macd_hist"].shift(1) <= 0)).astype(float)
        d["macd_cross_bear"] = ((d["macd_hist"] < 0) & (d["macd_hist"].shift(1) >= 0)).astype(float)

        d["macd_score"] = np.where(d["macd_hist"] > 0,
                          np.where(d["macd_cross_bull"] == 1, 1.0, 0.5),
                          np.where(d["macd_cross_bear"] == 1, -1.0, -0.5))
        return d


    # ── Bollinger ──────────────────────────────────────────────────────────
    def _bollinger(self, d):
        bb = ta.bbands(d["close"], length=BB_PERIOD, std=BB_STD)

        if bb is not None and not bb.empty:
            d["bb_upper"] = bb.iloc[:, 0]
            d["bb_mid"] = bb.iloc[:, 1]
            d["bb_lower"] = bb.iloc[:, 2]
        else:
            d["bb_upper"] = d["bb_mid"] = d["bb_lower"] = d["close"]

        d["bb_mid"] = d["bb_mid"].replace(0, 1)

        d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]
        d["bb_pct"] = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-10)

        d["bb_squeeze"] = (d["bb_width"] < d["bb_width"].rolling(BB_PERIOD).mean()).astype(float)

        d["bb_score"] = np.where(d["close"] > d["bb_upper"], -1.0,
                        np.where(d["close"] < d["bb_lower"], 1.0,
                        np.where(d["bb_pct"] > 0.8, -0.5,
                        np.where(d["bb_pct"] < 0.2, 0.5, 0.0))))
        return d


    # ── ATR ────────────────────────────────────────────────────────────────
    def _atr(self, d):
        d["atr"] = ta.atr(d["high"], d["low"], d["close"], length=ATR_PERIOD)
        d["atr"] = d["atr"].fillna(0)

        d["atr_pct"] = d["atr"] / (d["close"] + 1e-10)
        d["is_volatile"] = (d["atr_pct"] > LOW_VOLATILITY_ATR_PCT).astype(float)
        return d


    # ── VWAP ───────────────────────────────────────────────────────────────
    def _vwap(self, d):
        try:
            tp = (d["high"] + d["low"] + d["close"]) / 3
            cumvol = d["volume"].cumsum()
            cumvp = (tp * d["volume"]).cumsum()

            d["vwap"] = cumvp / (cumvol + 1e-10)
            d["vwap_dist"] = (d["close"] - d["vwap"]) / (d["vwap"] + 1e-10)
            d["above_vwap"] = (d["close"] > d["vwap"]).astype(float)

        except:
            d["vwap"] = d["close"]
            d["vwap_dist"] = 0
            d["above_vwap"] = 0.5

        return d


    # ── Volume ─────────────────────────────────────────────────────────────
    def _volume(self, d):
        d["vol_ma"] = d["volume"].rolling(VOLUME_PERIOD).mean()
        d["vol_ma"] = d["vol_ma"].replace(0, 1)

        d["vol_ratio"] = d["volume"] / d["vol_ma"]
        d["vol_spike"] = (d["vol_ratio"] > 1.5).astype(float)

        d["vol_delta"] = np.where(d["close"] >= d["open"], d["volume"], -d["volume"])
        d["cvd"] = d["vol_delta"].cumsum()
        d["cvd_slope"] = d["cvd"].diff(5)

        return d


    # ── Structure ───────────────────────────────────────────────────────────
    def _structure(self, d):
        highs = d["high"]
        lows = d["low"]

        d["swing_high"] = ((highs > highs.shift(1)) & (highs > highs.shift(2)) &
                           (highs > highs.shift(-1)) & (highs > highs.shift(-2))).astype(float)

        d["swing_low"] = ((lows < lows.shift(1)) & (lows < lows.shift(2)) &
                          (lows < lows.shift(-1)) & (lows < lows.shift(-2))).astype(float)

        return d


    # ── Smart Money ─────────────────────────────────────────────────────────
    def _smart_money(self, d):
        c = d["close"]
        h = d["high"]
        l = d["low"]

        prev_high = h.rolling(10).max().shift(1)
        prev_low = l.rolling(10).min().shift(1)

        d["bos_bull"] = ((c > prev_high.fillna(0)) & (c.shift(1) <= prev_high.shift(1).fillna(0))).astype(float)
        d["bos_bear"] = ((c < prev_low.fillna(0)) & (c.shift(1) >= prev_low.shift(1).fillna(0))).astype(float)

        d["smc_score"] = (d["bos_bull"] - d["bos_bear"]).clip(-1, 1)

        return d


# ── SINGLETON ───────────────────────────────────────────────────────────────
_engine = IndicatorEngine()

def get_indicator_engine():
    return _engine

# ── Multi-timeframe analyzer (FIX) ─────────────────────────────
class MTFAnalyzer:

    TF_WEIGHTS = {
        "1m": 0.15,   # noise but helps entry timing
        "3m": 0.35,   # MAIN ENTRY 🚀
        "5m": 0.25,   # confirmation
        "15m": 0.15,  # structure
        "30m": 0.07,  # weak filter
        "1h": 0.03    # almost ignored
    }

    def analyze(self, dfs):
        scores = {}

        for tf, df in dfs.items():
            if df is None or df.empty:
                continue

            row = df.iloc[-1]

            score = 0
            score += row.get("ema_score", 0) * 0.25
            score += row.get("rsi_score", 0) * 0.15
            score += row.get("macd_score", 0) * 0.15
            score += row.get("bb_score", 0) * 0.10
            score += row.get("smc_score", 0) * 0.20
            score += (row.get("above_vwap", 0.5) - 0.5) * 0.15

            scores[tf] = float(np.clip(score, -1, 1))

        total_weight = sum(self.TF_WEIGHTS.get(tf, 0.1) for tf in scores)

        composite = sum(
            scores[tf] * self.TF_WEIGHTS.get(tf, 0.1)
            for tf in scores
        ) / (total_weight + 1e-10)

        return {
            "composite": round(composite, 4),
            "bias": "BULL" if composite > 0.2 else "BEAR" if composite < -0.2 else "NEUTRAL"
        }


# ── SINGLETONS ───────────────────────────────────────────────
_engine = IndicatorEngine()
_mtf = MTFAnalyzer()

def get_indicator_engine():
    return _engine

def get_mtf_analyzer():
    return _mtf