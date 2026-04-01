# =============================================================================
# config/settings.py — Central Configuration
# =============================================================================
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# ── Exchange ──────────────────────────────────────────────────────────────────
EXCHANGE = "binance"
SYMBOL   = "BTC/USDT"
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m","1h"]
PRIMARY_TF = "1m"

# ── Data ──────────────────────────────────────────────────────────────────────
CANDLE_LIMITS = {
    "1m":  500,
    "3m": 500,
    "5m":  500,
    "15m": 500,
    "30m": 500,
    "1h":  300,
}

# ── Indicators ────────────────────────────────────────────────────────────────
EMA_PERIODS   = [20, 50, 200]
RSI_PERIOD    = 14
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
BB_PERIOD     = 20
BB_STD        = 2.0
ATR_PERIOD    = 14
VWAP_PERIOD   = 14
VOLUME_PERIOD = 20

# ── Signal Thresholds ─────────────────────────────────────────────────────────
SIGNAL_CONFIDENCE_MIN = 45    # minimum % to take trade
SCORE_THRESHOLD       = 0.25
MAX_ACTIVE_TRADES     = 2
LOW_VOLATILITY_ATR_PCT = 0.003  # skip if ATR < 0.3% of price

# ── Risk Management ───────────────────────────────────────────────────────────
CAPITAL          = 10_000.0   # USDT
RISK_PCT         = 0.02       # 2% per trade
ATR_SL_MULT      = 1.5
ATR_TP_MULT      = 3.0        # RR = 2:1 minimum
BREAKEVEN_R      = 1.0        # move SL to BE after 1R profit
TRAIL_ACTIVATION = 1.5        # start trailing at 1.5R
TRAIL_STEP_PCT   = 0.005      # 0.5% trail step

# ── Strategy Weights (ensemble) ───────────────────────────────────────────────
WEIGHTS = {
    "trend"      : 0.15,
    "momentum"   : 0.20,
    "structure"  : 0.15,
    "ml"         : 0.55,
    "sentiment"  : 0.05,
}

# ── ML ────────────────────────────────────────────────────────────────────────
LABEL_FWD_BARS   = 4
LABEL_MIN_MOVE   = 0.002
LSTM_SEQ_LEN     = 20
LSTM_FEATURES    = 16
MODEL_RETRAIN_BARS = 500     # retrain every N new bars

# ── Sentiment ─────────────────────────────────────────────────────────────────
SENTIMENT_WEIGHT_DECAY = 0.85   # older news decay
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/?auth_token={token}&currencies=BTC&public=true"
CRYPTOPANIC_TOKEN = ""          # fill in your free token

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "8606625275:AAH1_itGDXKMLIpI_Nvczs51DwMksTYfjKc"
TELEGRAM_CHAT_ID   = "1766536296"  

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = BASE_DIR / "models"
LOG_DIR   = BASE_DIR / "logs"
DATA_DIR  = BASE_DIR / "data"

for d in [MODEL_DIR, LOG_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)

# ── Server ────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
PRICE_UPDATE_SEC  = 5     # fetch live price every N seconds
SIGNAL_UPDATE_SEC = 60    # regenerate signal every N seconds


# uvicorn server:app --reload --port 8000