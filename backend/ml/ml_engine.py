# =============================================================================
# backend/ml/ml_engine.py — FINAL (UI FIX + TP/SL + FULL OUTPUT)
# =============================================================================
import numpy as np
import pandas as pd
import joblib
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ── LSTM SUPPORT ─────────────────────────────────────────────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("TensorFlow not available — LSTM disabled")


# ── FEATURES ─────────────────────────────────────────────
ML_FEATURES = [
    "ema_score", "ema50_dist",
    "rsi", "rsi_score",
    "macd_hist", "macd_score",
    "bb_pct", "bb_squeeze", "bb_score",
    "atr_pct", "is_volatile",
    "vwap_dist", "above_vwap",
    "vol_ratio", "vol_spike",
    "smc_score", "bos_bull", "bos_bear",
    "swing_high", "swing_low",
]


class MLEngine:

    def __init__(self):
        self.xgb = None
        self.rf = None
        self.lstm = None

        self.scaler = RobustScaler()
        self.scaler_l = RobustScaler()

        self.is_trained = False

        self._model_path = MODEL_DIR / "xgb_rf.pkl"
        self._lstm_path = MODEL_DIR / "lstm.keras"
        self._feature_cols = []

        self._load_if_exists()

    # ── TP/SL LABEL ─────────────────────────
    def label(self, df):

        tp_pct = 0.003
        sl_pct = 0.0015

        labels = []

        for i in range(len(df)):
            entry = df["close"].iloc[i]
            future = df.iloc[i+1:i+30]

            tp_hit = False
            sl_hit = False

            for _, row in future.iterrows():
                if row["high"] >= entry * (1 + tp_pct):
                    tp_hit = True
                    break
                if row["low"] <= entry * (1 - sl_pct):
                    sl_hit = True
                    break

            if tp_hit:
                labels.append(1)  # LONG
            else:
                labels.append(0)  # SHORT

        df["label"] = labels
        return df

    # ── BUILD FEATURES ─────────────────────────
    def _build_X(self, df):
        cols = [c for c in ML_FEATURES if c in df.columns]
        X = df[cols].copy()

        for c in X.select_dtypes(include="bool").columns:
            X[c] = X[c].astype(float)

        X.fillna(0, inplace=True)
        return X, cols

    # ── TRAIN ─────────────────────────
    def train(self, dfs):

        logger.info("Training ML models...")

        df = dfs.get("1m", pd.DataFrame())

        if df.empty or len(df) < 1000:
            return {"error": "Not enough data"}

        df = self.label(df)
        df.dropna(inplace=True)

        X, cols = self._build_X(df)
        y = df["label"].values

        self._feature_cols = cols

        X_sc = self.scaler.fit_transform(X)

        self.xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )

        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
        )

        tscv = TimeSeriesSplit(n_splits=5)

        for tr, te in tscv.split(X_sc):
            self.xgb.fit(X_sc[tr], y[tr])
            self.rf.fit(X_sc[tr], y[tr])

        self.xgb.fit(X_sc, y)
        self.rf.fit(X_sc, y)

        if LSTM_AVAILABLE:
            self._train_lstm(df, cols)

        self.is_trained = True
        self._save()

        return {"samples": len(df), "features": len(cols)}

    # ── LSTM ─────────────────────────
    def _train_lstm(self, df, cols):
        try:
            X_raw = df[cols].values.astype(np.float32)
            y_raw = df["label"].values

            X_sc = self.scaler_l.fit_transform(X_raw)

            seq = LSTM_SEQ_LEN
            Xs, ys = [], []

            for i in range(seq, len(X_sc)):
                Xs.append(X_sc[i-seq:i])
                ys.append(y_raw[i])

            Xs, ys = np.array(Xs), np.array(ys)

            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(seq, len(cols))),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(32),
                Dense(2, activation="softmax"),
            ])

            model.compile(
                optimizer=Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            model.fit(Xs, ys, epochs=10, batch_size=64, verbose=0)

            self.lstm = model
            model.save(str(self._lstm_path))

        except Exception as e:
            logger.error(f"LSTM error: {e}")

    # ── PREDICT ─────────────────────────
    def predict(self, df, df_seq=None):

        if not self.is_trained:
            return {"prediction": "FLAT", "confidence": 0}

        X, _ = self._build_X(df.iloc[[-1]])

        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0

        X = X[self._feature_cols]
        X_sc = self.scaler.transform(X)

        # ── MODEL OUTPUTS ─────────────────
        xgb_p = self.xgb.predict_proba(X_sc)[0]
        rf_p = self.rf.predict_proba(X_sc)[0]

        xgb_class = int(np.argmax(xgb_p))
        rf_class = int(np.argmax(rf_p))

        lstm_p = np.array([0.5, 0.5])
        lstm_used = False

        if LSTM_AVAILABLE and self.lstm is not None and df_seq is not None:
            try:
                seq = df_seq[self._feature_cols].values[-LSTM_SEQ_LEN:]
                seq = self.scaler_l.transform(seq)
                lstm_p = self.lstm.predict(seq[np.newaxis], verbose=0)[0]
                lstm_used = True
            except:
                pass

        # ── ENSEMBLE ─────────────────
        blend = 0.5 * xgb_p + 0.3 * rf_p + 0.2 * lstm_p

        pred = int(np.argmax(blend))
        confidence = round(float(blend[pred]) * 100, 1)

        label_map = {0: "SHORT", 1: "LONG"}

        return {
            "prediction": label_map[pred],
            "confidence": confidence,

            # ✅ FIX FOR UI
            "xgb_pred": label_map[xgb_class],
            "rf_pred": label_map[rf_class],

            # optional
            "p_long": round(float(blend[1]) * 100, 1),
            "p_short": round(float(blend[0]) * 100, 1),

            "lstm_used": lstm_used
        }

    # ── SAVE/LOAD ─────────────────────────
    def _save(self):
        joblib.dump({
            "xgb": self.xgb,
            "rf": self.rf,
            "scaler": self.scaler,
            "features": self._feature_cols
        }, str(self._model_path))

    def _load_if_exists(self):
        if self._model_path.exists():
            try:
                data = joblib.load(str(self._model_path))
                self.xgb = data["xgb"]
                self.rf = data["rf"]
                self.scaler = data["scaler"]
                self._feature_cols = data["features"]
                self.is_trained = True
                logger.info("Loaded existing ML models")
            except Exception as e:
                logger.warning(f"Load failed: {e}")


_ml_engine = MLEngine()

def get_ml_engine():
    return _ml_engine