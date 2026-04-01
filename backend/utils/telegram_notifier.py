# =============================================================================
# backend/utils/telegram_notifier.py — Telegram Alert System
# =============================================================================
import requests
import logging
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

EMOJI = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⬜", "TP_HIT": "✅", "SL_HIT": "❌",
         "MANUAL": "🔵", "trail": "📈", "breakeven": "🛡️"}


class TelegramNotifier:
    """Sends trade alerts via Telegram Bot API."""

    def __init__(self):
        self.token   = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.info("Telegram not configured — alerts disabled. Add token + chat_id in settings.py")

    def send(self, text: str):
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={
                "chat_id"   : self.chat_id,
                "text"      : text,
                "parse_mode": "HTML",
            }, timeout=5)
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def send_trade_open(self, trade):
        e = EMOJI.get(trade.direction, "")
        msg = (
            f"{e} <b>NEW {trade.direction} SIGNAL</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entry:      <code>${trade.entry:,.2f}</code>\n"
            f"🛑 Stop Loss:  <code>${trade.stop_loss:,.2f}</code>\n"
            f"🎯 Take Profit:<code>${trade.take_profit:,.2f}</code>\n"
            f"🔒 Breakeven:  <code>${trade.breakeven:,.2f}</code>\n"
            f"⚖️ R:R Ratio:  <code>1:{trade.risk_reward}</code>\n"
            f"📊 Confidence: <code>{trade.confidence:.1f}%</code>\n"
            f"💰 Risk:       <code>${trade.risk_usd:.0f}</code>\n"
            f"📦 Size:       <code>{trade.units:.5f} BTC</code>\n"
            f"⏰ Time:       <code>{datetime.utcnow().strftime('%H:%M UTC')}</code>"
        )
        self.send(msg)

    def send_trade_close(self, trade):
        e    = EMOJI.get(trade.close_reason, "🔵")
        pnl  = trade.pnl_pct or 0
        icon = "🟢" if pnl > 0 else "🔴"
        msg = (
            f"{e} <b>TRADE CLOSED — {trade.direction}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Reason:    <code>{trade.close_reason}</code>\n"
            f"Close Px:  <code>${trade.close_price:,.2f}</code>\n"
            f"{icon} PnL:      <code>{'+' if pnl>0 else ''}{pnl:.3f}% (${trade.pnl_usd:+.2f})</code>\n"
            f"⏱ Duration: Entry→Close logged"
        )
        self.send(msg)

    def send_sl_moved(self, trade, old_sl: float, reason: str):
        e = EMOJI.get(reason, "📌")
        msg = (
            f"{e} <b>SL UPDATED — {trade.direction}</b>\n"
            f"Old SL: <code>${old_sl:,.2f}</code>\n"
            f"New SL: <code>${trade.stop_loss:,.2f}</code>\n"
            f"Reason: <code>{reason}</code>"
        )
        self.send(msg)

    def send_signal_alert(self, signal: dict):
        d = signal.get("direction", "FLAT")
        if d == "FLAT":
            return
        e = EMOJI.get(d, "")
        msg = (
            f"{e} <b>SIGNAL: {d}</b> "
            f"| Conf: <code>{signal.get('confidence',0):.1f}%</code> "
            f"| Score: <code>{signal.get('raw_score',0):.3f}</code>"
        )
        self.send(msg)


_notifier = TelegramNotifier()
def get_notifier() -> TelegramNotifier:
    return _notifier
