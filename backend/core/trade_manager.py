# =============================================================================
# backend/core/trade_manager.py — Trade Lifecycle & DB Integration
# =============================================================================
import uuid
import logging
import json
from datetime import datetime
from typing import List, Optional

from backend.db.db import get_connection

logger = logging.getLogger(__name__)


# =============================================================================
# DB FUNCTIONS
# =============================================================================
def save_trade_open(trade):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO trades (
        id, symbol, timeframe, direction, status,
        entry_price, stop_loss, take_profit, breakeven,
        atr, risk_reward, confidence,
        open_time
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        trade.id,
        trade.symbol,
        trade.timeframe,
        trade.direction,
        trade.status,
        trade.entry,
        trade.stop_loss,
        trade.take_profit,
        trade.breakeven,
        trade.atr,
        trade.risk_reward,
        trade.confidence,
        trade.open_time
    )

    cursor.execute(query, values)
    conn.commit()
    conn.close()


def update_trade_close(trade):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    UPDATE trades
    SET
        status = %s,
        close_time = %s,
        close_price = %s,
        result = %s,
        pnl_pct = %s,
        pnl_usd = %s
    WHERE id = %s
    """

    values = (
        trade.status,
        trade.close_time,
        trade.close_price,
        trade.result,
        trade.pnl_pct,
        trade.pnl_usd,
        trade.id
    )

    cursor.execute(query, values)
    conn.commit()
    conn.close()


# =============================================================================
# TRADE CLASS
# =============================================================================
class Trade:
    def __init__(self, signal: dict):
        self.id = str(uuid.uuid4())[:8]

        # Basic Info
        self.symbol = signal.get("symbol", "BTCUSDT")
        self.timeframe = signal.get("timeframe", "1m")

        self.direction = signal["direction"]
        self.entry = signal["entry"]
        self.stop_loss = signal["stop_loss"]
        self.take_profit = signal["take_profit"]
        self.breakeven = signal["breakeven"]

        # Metrics
        self.risk_reward = signal.get("risk_reward", 0)
        self.atr = signal.get("atr", 0)
        self.confidence = signal.get("confidence", 0)

        # ML Prediction (IMPORTANT)
        self.prediction = signal.get("ml_prediction", None)

        # Time
        self.open_time = datetime.utcnow()
        self.close_time = None

        # Status
        self.status = "OPEN"
        self.close_price = None
        self.close_reason = None

        # Result
        self.pnl_pct = None
        self.pnl_usd = None
        self.result = None

    def to_dict(self):
        return self.__dict__


# =============================================================================
# TRADE MANAGER
# =============================================================================
class TradeManager:

    def __init__(self):
        self._trades: List[Trade] = []

    # -------------------------------------------------------------------------
    # OPEN TRADE
    # -------------------------------------------------------------------------
    def open_trade(self, signal: dict) -> Optional[Trade]:

        if signal.get("direction") == "FLAT":
            return None

        trade = Trade(signal)

        self._trades.append(trade)

        # ✅ SAVE TO DB
        save_trade_open(trade)

        logger.info(f"Trade OPENED: {trade.direction} @ {trade.entry}")

        return trade

    # -------------------------------------------------------------------------
    # UPDATE PRICE
    # -------------------------------------------------------------------------
    def update(self, price: float, notifier=None):
        for trade in self.active_trades():
            self._update_trade(trade, price, notifier)

    def _update_trade(self, t: Trade, price: float, notifier=None):
        is_long = t.direction == "LONG"

        # TP HIT
        if (is_long and price >= t.take_profit) or \
           (not is_long and price <= t.take_profit):
            self._close(t, t.take_profit, "TP_HIT",notifier)

        # SL HIT
        elif (is_long and price <= t.stop_loss) or \
             (not is_long and price >= t.stop_loss):
            self._close(t, t.stop_loss, "SL_HIT",notifier)

    # -------------------------------------------------------------------------
    # CLOSE TRADE
    # -------------------------------------------------------------------------
    def _close(self, t: Trade, price: float, reason: str, notifier=None):
        t.close_price = price
        t.close_reason = reason
        t.close_time = datetime.utcnow()
        t.status = "CLOSED"

        # PnL
        if t.direction == "LONG":
            t.pnl_pct = (price - t.entry) / t.entry * 100
        else:
            t.pnl_pct = (t.entry - price) / t.entry * 100

        t.pnl_usd = t.pnl_pct  # (you can improve later)

        # RESULT
        if t.pnl_pct > 0:
            t.result = "WIN"
        elif t.pnl_pct < 0:
            t.result = "LOSS"
        else:
            t.result = "BREAKEVEN"

        # ✅ UPDATE DB
        update_trade_close(t)

        logger.info(f"Trade CLOSED: {t.id} | {t.result} | {t.pnl_pct:.2f}%")

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def active_trades(self) -> List[Trade]:
        return [t for t in self._trades if t.status != "CLOSED"]

    def all_trades(self) -> List[Trade]:
        return self._trades
    ## Summary 
    def summary(self):
        closed = [t for t in self._trades if t.status == "CLOSED"]
        open_trades = [t for t in self._trades if t.status != "CLOSED"]

        wins = [t for t in closed if t.pnl_pct and t.pnl_pct > 0]

        total_pnl = sum(t.pnl_usd or 0 for t in closed)

        return {
            "total_trades": len(self._trades),
            "open_trades": len(open_trades),
            "closed_trades": len(closed),
            "win_rate": round((len(wins) / len(closed)) * 100, 2) if closed else 0,
            "total_pnl": round(total_pnl, 2),
            "active": [t.to_dict() for t in open_trades]
        }

# =============================================================================
# SINGLETON
# =============================================================================
_manager = TradeManager()

def get_trade_manager():
    return _manager