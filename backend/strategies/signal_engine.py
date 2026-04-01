# backend/strategies/signal_engine.py

import config.settings as SETTINGS


class SignalEngine:
    def __init__(self):
        pass

    # ✅ MUST match engine call
    def generate(self, dfs, mtf, ml_pred, sentiment, orderbook):

        # ── FIX: HANDLE MULTI-TF DATA SAFELY ──
        if isinstance(dfs, dict):
            if SETTINGS.PRIMARY_TF in dfs:
                df = dfs[SETTINGS.PRIMARY_TF]
            else:
                df = list(dfs.values())[0]
        else:
            df = dfs

        # ── SAFETY CHECK ──
        if df is None or df.empty:
            return {
                "direction": "FLAT",
                "confidence": 0,
                "entry": None,
                "reason": "NO_DATA"
            }

        # ── ML OUTPUT ──
        direction = ml_pred.get("prediction", "FLAT")
        confidence = ml_pred.get("confidence", 0)

        # ── CONFIDENCE FILTER ──
        min_conf = getattr(SETTINGS, "SIGNAL_CONFIDENCE_MIN", 30)

        if confidence < min_conf:
            direction = "FLAT"

        # ── PRICE ──
        price = float(df["close"].iloc[-1])

        # ── ATR CALCULATION ──
        atr = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])

        # fallback if ATR is NaN
        if atr != atr:  # NaN check
            atr = price * 0.002

        # ── SL / TP CALCULATION ──
        if direction == "LONG":
            sl = price - atr * 1.5
            tp = price + atr * 3
        elif direction == "SHORT":
            sl = price + atr * 1.5
            tp = price - atr * 3
        else:
            sl = None
            tp = None

        # ── RISK REWARD ──
        if sl and tp:
            risk = abs(price - sl)
            reward = abs(tp - price)
            rr = reward / risk if risk != 0 else 0
        else:
            rr = 0

        # ── FINAL SIGNAL ──
        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry": round(price, 2),

            "stop_loss": round(sl, 2) if sl else None,
            "take_profit": round(tp, 2) if tp else None,
            "breakeven": round(price, 2),

            "risk_reward": round(rr, 2),
            "atr": round(atr, 2),

            "reason": "ML_ONLY_MODE"
        }


# ✅ REQUIRED BY ENGINE
def get_signal_engine():
    return SignalEngine()