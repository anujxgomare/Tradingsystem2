# =============================================================================
# server.py — FastAPI Server (entry point)
# Run: uvicorn server:app --reload --port 8000
# =============================================================================
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Configure logging first ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/server.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Import engine (starts background threads on startup) ──────────────────────
from backend.core.engine      import get_engine
from backend.core.trade_manager import get_trade_manager
from backend.utils.telegram_notifier import get_notifier
from config.settings import *

app = FastAPI(title="BTC Quant Terminal API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE / "frontend" / "static")), name="static")

# ── WebSocket connection manager ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

ws_manager = ConnectionManager()

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    Path("logs").mkdir(exist_ok=True)
    engine = get_engine()
    engine.start()
    logger.info("Trading engine started on server startup")
    # Background task: broadcast price updates to WS clients
    asyncio.create_task(_ws_broadcaster())

async def _ws_broadcaster():
    engine = get_engine()
    while True:
        try:
            if ws_manager.active:
                state = engine.state
                price = state.get("price") or {}
                sig   = state.get("signal") or {}
                await ws_manager.broadcast({
                    "type"       : "tick",
                    "price"      : price.get("price"),
                    "change24h"  : price.get("change24h"),
                    "direction"  : sig.get("direction", "FLAT"),
                    "confidence" : sig.get("confidence", 0),
                    "raw_score"  : sig.get("raw_score", 0),
                    "training"   : state.get("training", False),
                    "ts"         : datetime.utcnow().isoformat(),
                })
        except Exception as e:
            logger.error(f"WS broadcast error: {e}")
        await asyncio.sleep(PRICE_UPDATE_SEC)


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
def root():
    return FileResponse(str(BASE / "frontend" / "index.html"))

# ── /price ────────────────────────────────────────────────────────────────────
@app.get("/price")
def get_price():
    """Live BTC price with 24h stats."""
    engine = get_engine()
    price  = engine.state.get("price")
    if not price:
        raise HTTPException(503, "Price not available yet")
    return price

@app.get("/price/history")
def get_price_history():
    """Last 500 price ticks for sparkline."""
    return get_engine().state.get("price_history", [])

# ── /predict ─────────────────────────────────────────────────────────────────
@app.get("/predict")
def get_prediction():
    """ML model prediction with probabilities."""
    ml = get_engine().state.get("ml")
    if not ml:
        raise HTTPException(503, "Model not ready — training in progress")
    return ml

# ── /signal ──────────────────────────────────────────────────────────────────
@app.get("/signal")
def get_signal():
    """Full ensemble signal with all components."""
    sig = get_engine().state.get("signal")
    if not sig:
        raise HTTPException(503, "Signal not ready yet")
    return sig

@app.get("/signal/history")
def get_signal_history():
    """Placeholder — extend to store signal history in DB."""
    return {"signals": [], "note": "Signal history coming soon"}

# ── /sentiment ────────────────────────────────────────────────────────────────
@app.get("/sentiment")
def get_sentiment():
    """Fear & Greed + News + Reddit sentiment."""
    s = get_engine().state.get("sentiment")
    if not s:
        raise HTTPException(503, "Sentiment not ready")
    return s

# ── /mtf ──────────────────────────────────────────────────────────────────────
@app.get("/mtf")
def get_mtf():
    """Multi-timeframe analysis breakdown."""
    mtf = get_engine().state.get("mtf")
    if not mtf:
        raise HTTPException(503, "MTF data not ready")
    return mtf

# ── /active-trades ────────────────────────────────────────────────────────────
@app.get("/active-trades")
def get_active_trades():
    """All active (open) trades."""
    tm = get_trade_manager()
    return {
        "trades"  : [t.to_dict() for t in tm.active_trades()],
        "summary" : tm.summary(),
    }

@app.get("/trades/all")
def get_all_trades():
    """All trades (open + closed)."""
    tm = get_trade_manager()
    return {"trades": [t.to_dict() for t in tm.all_trades()]}

class CloseBody(BaseModel):
    trade_id : str
    price    : Optional[float] = None

@app.post("/trades/close")
def close_trade(body: CloseBody):
    """Manually close a trade."""
    tm = get_trade_manager()
    price = body.price or (get_engine().state.get("price") or {}).get("price", 0)
    result = tm.close_trade_manual(body.trade_id, price, get_notifier())
    if not result:
        raise HTTPException(404, f"Trade {body.trade_id} not found or already closed")
    return result

# ── /status ───────────────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    """Engine health and training status."""
    state = get_engine().state
    return {
        "training"   : state.get("training", False),
        "trained"    : state.get("trained", False),
        "error"      : state.get("error"),
        "started_at" : state.get("started_at"),
        "last_signal": state.get("last_signal"),
        "last_price" : state.get("last_price"),
        "active_trades": len(get_trade_manager().active_trades()),
        "telegram"   : get_notifier().enabled,
    }

# ── /dashboard ────────────────────────────────────────────────────────────────
@app.get("/dashboard")
def get_dashboard():
    """Full dashboard snapshot — everything in one call."""
    engine = get_engine()
    state  = engine.state
    tm     = get_trade_manager()
    return {
        "price"    : state.get("price"),
        "signal"   : state.get("signal"),
        "ml"       : state.get("ml"),
        "sentiment": state.get("sentiment"),
        "mtf"      : state.get("mtf"),
        "trades"   : tm.summary(),
        "status"   : {
            "training": state.get("training"),
            "trained" : state.get("trained"),
            "error"   : state.get("error"),
        },
    }

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive ping
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
