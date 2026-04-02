# =============================================================================
# backend/strategies/signal_engine.py — FINAL (Strategy + ML Hybrid)
# =============================================================================

import config.settings as SETTINGS
from datetime import datetime

class SignalEngine:

    def generate(self, dfs, mtf, ml_pred, sentiment, orderbook):

        # ── GET DATA ─────────────────────────────────────
        df = dfs.get("1m")

        if df is None or df.empty:
            df = next(iter(dfs.values()))

        price = float(df["close"].iloc[-1])

        # ── 🔥 LIQUIDITY SWEEP LOGIC ─────────────────────
        high_prev = df["high"].rolling(10).max().iloc[-2]
        low_prev = df["low"].rolling(10).min().iloc[-2]

        sweep_high = price > high_prev
        sweep_low = price < low_prev

        last_candle = df.iloc[-1]
        bullish = last_candle["close"] > last_candle["open"]
        bearish = last_candle["close"] < last_candle["open"]

        # ── ML OUTPUT ────────────────────────────────────
        ml_dir = ml_pred.get("prediction", "FLAT")
        ml_conf = ml_pred.get("confidence", 0)

        # ── 🎯 FINAL DECISION LOGIC ─────────────────────
        direction = "FLAT"
        reason = "NO SIGNAL"

        # ✅ CASE 1: STRATEGY + ML (BEST QUALITY)
        if (sweep_low and bullish and ml_dir == "LONG" and ml_conf > 60):
            direction = "LONG"
            reason = "SWEEP LOW + BULLISH + ML"

        elif (sweep_high and bearish and ml_dir == "SHORT" and ml_conf > 60):
            direction = "SHORT"
            reason = "SWEEP HIGH + BEARISH + ML"

        # ⚡ CASE 2: ML ONLY (FALLBACK)
        elif ml_conf > 75:
            direction = ml_dir
            reason = "ML HIGH CONFIDENCE"

        # ❌ NO TRADE
        if direction == "FLAT":
            return {"direction": "FLAT", "confidence": 0}

        # ── ATR ─────────────────────────────────────────
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]

        if atr != atr or atr == 0:
            atr = price * 0.002

        # ── TP/SL ───────────────────────────────────────
        if direction == "LONG":
            sl = price - atr * SETTINGS.ATR_SL_MULT
            tp = price + atr * SETTINGS.ATR_TP_MULT
        else:
            sl = price + atr * SETTINGS.ATR_SL_MULT
            tp = price - atr * SETTINGS.ATR_TP_MULT

        rr = abs(tp - price) / abs(price - sl)

        # ── FINAL SIGNAL ────────────────────────────────
        breakeven = price  # simple breakeven at entry

        return {
            "direction": direction,
            "confidence": ml_conf,
            "entry": price,
            "stop_loss": sl,
            "take_profit": tp,
            "breakeven": breakeven,   # ✅ FIX
            "risk_reward": round(rr, 2),
            "ml_prediction": ml_dir,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }


def get_signal_engine():
    return SignalEngine()