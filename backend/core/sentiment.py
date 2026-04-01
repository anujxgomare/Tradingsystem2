# =============================================================================
# backend/core/sentiment.py — Sentiment & News Analysis
# =============================================================================
import requests
import time
import logging
import re
from typing import Optional
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import *

logger = logging.getLogger(__name__)

# Simple keyword sentiment lexicon
BULLISH_WORDS = [
    "surge", "rally", "bullish", "breakout", "all-time high", "ath", "moon",
    "buy", "long", "uptrend", "support", "accumulate", "institutional",
    "adoption", "etf", "approve", "halving", "rise", "pump", "gain",
    "recover", "rebound", "green", "positive", "strong", "upgrade",
]
BEARISH_WORDS = [
    "crash", "bearish", "dump", "sell", "short", "downtrend", "resistance",
    "ban", "hack", "fud", "fear", "liquidation", "correction", "drop",
    "plunge", "fall", "red", "negative", "weak", "scam", "fraud",
    "regulation", "lawsuit", "warning", "loss", "collapse",
]


def _score_text(text: str) -> float:
    """Score a text snippet -1 (bearish) to +1 (bullish)."""
    text = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in text)
    bear = sum(1 for w in BEARISH_WORDS if w in text)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


class SentimentAnalyzer:
    """Fetches and aggregates crypto market sentiment."""

    def __init__(self):
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0
        self._cache_ttl = 300   # 5 minutes

    def get_sentiment(self) -> dict:
        """Returns composite sentiment score -100 to +100."""
        now = time.time()
        if self._cache and now - self._cache_ts < self._cache_ttl:
            return self._cache

        fg  = self._fear_greed()
        news = self._cryptopanic_news()
        reddit = self._reddit_sentiment()

        # Composite (weighted)
        fg_score    = (fg["value"] - 50) / 50      # -1 to +1
        news_score  = news["score"]
        reddit_score= reddit["score"]

        composite = (
            fg_score     * 0.40 +
            news_score   * 0.40 +
            reddit_score * 0.20
        )
        composite_pct = round(composite * 100, 1)   # -100 to +100

        result = {
            "composite"       : composite_pct,
            "bias"            : "BULLISH" if composite > 0.1 else
                                "BEARISH" if composite < -0.1 else "NEUTRAL",
            "fear_greed"      : fg,
            "news"            : news,
            "reddit"          : reddit,
            "signal_adj"      : round(composite * 0.15, 4),  # weight in final signal
            "timestamp"       : datetime.utcnow().isoformat(),
        }
        self._cache    = result
        self._cache_ts = now
        return result

    # ── Fear & Greed Index ────────────────────────────────────────────────────
    def _fear_greed(self) -> dict:
        try:
            r = requests.get(FEAR_GREED_URL, timeout=5)
            data = r.json()["data"][0]
            val = int(data["value"])
            return {
                "value"      : val,
                "label"      : data["value_classification"],
                "score_raw"  : (val - 50) / 50,
            }
        except Exception as e:
            logger.warning(f"Fear & Greed API error: {e}")
            return {"value": 50, "label": "Neutral", "score_raw": 0.0}

    # ── CryptoPanic News ──────────────────────────────────────────────────────
    def _cryptopanic_news(self) -> dict:
        if not CRYPTOPANIC_TOKEN:
            # Fallback: scrape CoinDesk RSS (public, no auth)
            return self._coingecko_news()
        try:
            url = CRYPTOPANIC_URL.format(token=CRYPTOPANIC_TOKEN)
            r   = requests.get(url, timeout=8)
            posts = r.json().get("results", [])[:20]
            scores = []
            headlines = []
            for p in posts:
                title = p.get("title", "")
                s     = _score_text(title)
                # Apply votes if available
                votes = p.get("votes", {})
                bull_v = votes.get("positive", 0)
                bear_v = votes.get("negative", 0)
                if bull_v + bear_v > 0:
                    vote_s = (bull_v - bear_v) / (bull_v + bear_v)
                    s = (s + vote_s) / 2
                scores.append(s)
                headlines.append({"title": title, "score": round(s, 2)})
            avg = float(sum(scores) / len(scores)) if scores else 0
            return {"score": round(avg, 4), "count": len(scores), "headlines": headlines[:5]}
        except Exception as e:
            logger.warning(f"CryptoPanic error: {e}")
            return self._coingecko_news()

    def _coingecko_news(self) -> dict:
        """Fallback: fetch from CoinGecko news (public)."""
        try:
            url = "https://api.coingecko.com/api/v3/news"
            r   = requests.get(url, timeout=8, headers={"Accept": "application/json"})
            items = r.json() if isinstance(r.json(), list) else []
            scores = []
            headlines = []
            for item in items[:15]:
                title = item.get("title", "")
                desc  = item.get("description", "")
                s     = _score_text(title + " " + desc)
                scores.append(s)
                headlines.append({"title": title[:80], "score": round(s, 2)})
            avg = float(sum(scores) / len(scores)) if scores else 0.0
            return {"score": round(avg, 4), "count": len(scores), "headlines": headlines[:5], "source": "coingecko"}
        except Exception as e:
            logger.warning(f"CoinGecko news error: {e}")
            return {"score": 0.0, "count": 0, "headlines": [], "source": "none"}

    # ── Reddit (public JSON) ──────────────────────────────────────────────────
    def _reddit_sentiment(self) -> dict:
        """Scrape r/Bitcoin and r/CryptoCurrency public JSON (no auth needed)."""
        subreddits = ["Bitcoin", "CryptoCurrency"]
        all_scores = []
        posts_out  = []
        headers    = {"User-Agent": "btc-quant-bot/1.0"}
        for sub in subreddits:
            try:
                url = f"https://www.reddit.com/r/{sub}/hot.json?limit=15"
                r   = requests.get(url, timeout=8, headers=headers)
                if r.status_code != 200:
                    continue
                posts = r.json()["data"]["children"]
                for p in posts:
                    d     = p["data"]
                    title = d.get("title", "")
                    score_r = d.get("score", 0)      # upvotes
                    s = _score_text(title)
                    # Weight by upvotes (log scale)
                    w = 1 + min(5, max(0, (score_r ** 0.4) / 10))
                    all_scores.extend([s] * int(w))
                    posts_out.append({"title": title[:70], "score": round(s, 2), "upvotes": score_r})
            except Exception as e:
                logger.debug(f"Reddit error [{sub}]: {e}")

        avg = float(sum(all_scores) / len(all_scores)) if all_scores else 0.0
        return {"score": round(avg, 4), "count": len(posts_out), "posts": posts_out[:3]}


_sentiment = SentimentAnalyzer()
def get_sentiment_analyzer() -> SentimentAnalyzer:
    return _sentiment
