# =============================================================================
# backend/ml/ml_engine.py — XGBoost + LSTM + Random Forest Ensemble
# =============================================================================
import numpy as np
import pandas as pd
import joblib
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import *

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Try import TensorFlow/Keras for LSTM (optional)
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("TensorFlow not available — LSTM disabled, XGB+RF ensemble used")


# ── Feature columns used for ML ──────────────────────────────────────────────
ML_FEATURES = [
    "ema_score", "ema50_dist",
    "rsi", "rsi_score",
    "macd_hist", "macd_score",
    "bb_pct", "bb_squeeze", "bb_score",
    "atr_pct", "is_volatile",
    "vwap_dist", "above_vwap",
    "vol_ratio", "vol_spike",
    "smc_score", "bos_bull", "bos_bear", "choch_bull",
    "swing_high", "swing_low",
]


class MLEngine:
    """XGBoost + Random Forest + optional LSTM ensemble for BTC direction prediction."""

    def __init__(self):
        self.xgb: Optional[XGBClassifier]         = None
        self.rf:  Optional[RandomForestClassifier] = None
        self.lstm = None
        self.scaler   = RobustScaler()
        self.scaler_l = RobustScaler()   # for LSTM
        self.is_trained = False
        self._model_path  = MODEL_DIR / "xgb_rf.pkl"
        self._lstm_path   = MODEL_DIR / "lstm.keras"
        self._feature_cols: list = []
        self._load_if_exists()

    # ── Label ─────────────────────────────────────────────────────────────────
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-looking label: 2=LONG, 0=SHORT, 1=FLAT."""
        fwd = LABEL_FWD_BARS
        future_max = df["high"].shift(-fwd).rolling(fwd).max()
        future_min = df["low"].shift(-fwd).rolling(fwd).min()
        up   = (future_max - df["close"]) / df["close"]
        down = (df["close"] - future_min) / df["close"]
        df["label"] = np.where(up >= LABEL_MIN_MOVE, 2,
                      np.where(down >= LABEL_MIN_MOVE, 0, 1))
        return df

    # ── Build feature matrix ──────────────────────────────────────────────────
    def _build_X(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        cols = [c for c in ML_FEATURES if c in df.columns]
        X = df[cols].copy()
        for c in X.select_dtypes(include="bool").columns:
            X[c] = X[c].astype(float)
        X.fillna(0, inplace=True)
        return X, cols

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, dfs: Dict[str, pd.DataFrame]) -> dict:
        """Train on 15m data (primary). Returns accuracy metrics."""
        logger.info("Training ML models...")
        df = dfs.get("15m", pd.DataFrame())
        if df.empty or len(df) < 200:
            return {"error": "Not enough data"}

        df = self.label(df)
        df.dropna(subset=["label"], inplace=True)
        X, cols = self._build_X(df)
        y = df["label"].values.astype(int)
        self._feature_cols = cols

        X_sc = self.scaler.fit_transform(X)

        # XGBoost
        self.xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
        # Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_split=15, random_state=42, n_jobs=-1,
        )

        tscv = TimeSeriesSplit(n_splits=5)
        xgb_accs, rf_accs = [], []
        for tr, te in tscv.split(X_sc):
            self.xgb.fit(X_sc[tr], y[tr])
            self.rf.fit(X_sc[tr], y[tr])
            xgb_accs.append(accuracy_score(y[te], self.xgb.predict(X_sc[te])))
            rf_accs.append(accuracy_score(y[te], self.rf.predict(X_sc[te])))

        # Final fit
        self.xgb.fit(X_sc, y)
        self.rf.fit(X_sc, y)

        # LSTM (optional)
        lstm_acc = None
        if LSTM_AVAILABLE:
            lstm_acc = self._train_lstm(df, cols)

        self.is_trained = True
        self._save()

        metrics = {
            "xgb_cv_acc" : round(float(np.mean(xgb_accs)), 4),
            "rf_cv_acc"  : round(float(np.mean(rf_accs)), 4),
            "lstm_acc"   : lstm_acc,
            "samples"    : len(df),
            "features"   : len(cols),
            "label_dist" : {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        }
        logger.info(f"Training complete: {metrics}")
        return metrics

    # ── LSTM ──────────────────────────────────────────────────────────────────
    def _train_lstm(self, df: pd.DataFrame, cols: list) -> Optional[float]:
        try:
            X_raw = df[cols].fillna(0).values.astype(np.float32)
            y_raw = df["label"].values.astype(int)
            X_sc  = self.scaler_l.fit_transform(X_raw)

            seq = LSTM_SEQ_LEN
            Xs, ys = [], []
            for i in range(seq, len(X_sc)):
                Xs.append(X_sc[i-seq:i])
                ys.append(y_raw[i])
            Xs = np.array(Xs)
            ys = np.array(ys)

            split = int(len(Xs) * 0.8)
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(seq, len(cols))),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(3, activation="softmax"),
            ])
            model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            es = EarlyStopping(patience=5, restore_best_weights=True)
            model.fit(Xs[:split], ys[:split],
                      validation_data=(Xs[split:], ys[split:]),
                      epochs=30, batch_size=64, callbacks=[es], verbose=0)
            self.lstm = model
            _, acc = model.evaluate(Xs[split:], ys[split:], verbose=0)
            model.save(str(self._lstm_path))
            return round(float(acc), 4)
        except Exception as e:
            logger.error(f"LSTM training error: {e}")
            return None

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame, df_seq: Optional[pd.DataFrame] = None) -> dict:
        if not self.is_trained:
            return self._null_prediction("Model not trained yet")

        try:
            X, _ = self._build_X(df.iloc[[-1]])

            for c in self._feature_cols:
                if c not in X.columns:
                    X[c] = 0.0

            X = X[self._feature_cols]
            X_sc = self.scaler.transform(X)

            # ── XGB + RF ─────────────────────────────────────
            xgb_p = self._pad_proba(self.xgb.predict_proba(X_sc)[0], self.xgb.classes_)
            rf_p  = self._pad_proba(self.rf.predict_proba(X_sc)[0],  self.rf.classes_)

            # ── 🔥 LSTM (IMPROVED) ───────────────────────────
            lstm_p = np.array([1/3, 1/3, 1/3])
            lstm_conf = 0
            lstm_used = False

            if LSTM_AVAILABLE and self.lstm is not None and df_seq is not None:
                try:
                    cols = self._feature_cols
                    seq_len = LSTM_SEQ_LEN

                    seq_df = df_seq[cols].fillna(0)

                    if len(seq_df) >= seq_len:
                        seq = seq_df.values[-seq_len:].astype(np.float32)
                        seq_sc = self.scaler_l.transform(seq)

                        lstm_p = self.lstm.predict(seq_sc[np.newaxis], verbose=0)[0]
                        lstm_conf = float(np.max(lstm_p))
                        lstm_used = True

                except Exception as e:
                    logger.warning(f"LSTM prediction error: {e}")

            # ── 🔥 DYNAMIC WEIGHTING (KEY IMPROVEMENT) ───────
            if lstm_used and lstm_conf > 0.6:
                # LSTM strong → dominate
                lstm_w = 0.45
                xgb_w  = 0.30
                rf_w   = 0.25
            elif lstm_used:
                # LSTM moderate
                lstm_w = 0.30
                xgb_w  = 0.40
                rf_w   = 0.30
            else:
                # No LSTM
                lstm_w = 0.0
                xgb_w  = 0.55
                rf_w   = 0.45

            # ── ENSEMBLE ─────────────────────────────────────
            blend = xgb_w * xgb_p + rf_w * rf_p + lstm_w * lstm_p
            blend = blend / blend.sum()

            pred_class = int(np.argmax(blend))
            label_map  = {0: "SHORT", 1: "FLAT", 2: "LONG"}

            confidence = round(float(blend[pred_class]) * 100, 1)

            return {
                "prediction" : label_map[pred_class],
                "confidence" : confidence,
                "p_long"     : round(float(blend[2]) * 100, 1),
                "p_flat"     : round(float(blend[1]) * 100, 1),
                "p_short"    : round(float(blend[0]) * 100, 1),

                "xgb_pred"   : label_map[int(np.argmax(xgb_p))],
                "rf_pred"    : label_map[int(np.argmax(rf_p))],
                "lstm_used"  : lstm_used,
                "lstm_conf"  : round(lstm_conf * 100, 1),
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._null_prediction(str(e))

    def _pad_proba(self, proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
        out = np.array([1/3, 1/3, 1/3])
        for i, c in enumerate(classes):
            if 0 <= c <= 2:
                out[c] = proba[i]
        return out / out.sum()

    def _null_prediction(self, reason: str) -> dict:
        return {"prediction": "FLAT", "confidence": 0, "p_long": 33, "p_flat": 34,
                "p_short": 33, "error": reason}

    # ── Save / Load ───────────────────────────────────────────────────────────
    def _save(self):
        joblib.dump({"xgb": self.xgb, "rf": self.rf, "scaler": self.scaler,
                     "scaler_l": self.scaler_l, "features": self._feature_cols},
                    str(self._model_path))

    def _load_if_exists(self):
        if self._model_path.exists():
            try:
                data = joblib.load(str(self._model_path))
                self.xgb = data["xgb"]
                self.rf  = data["rf"]
                self.scaler   = data["scaler"]
                self.scaler_l = data.get("scaler_l", self.scaler_l)
                self._feature_cols = data.get("features", [])
                self.is_trained = True
                logger.info("Loaded existing ML models")
                if LSTM_AVAILABLE and self._lstm_path.exists():
                    self.lstm = load_model(str(self._lstm_path))
            except Exception as e:
                logger.warning(f"Could not load saved models: {e}")


_ml_engine = MLEngine()
def get_ml_engine() -> MLEngine:
    return _ml_engine
