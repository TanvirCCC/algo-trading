"""
Random Forest Signal Filter
DeltaTrend TikTok: "We train the model using a random forest classifier to reject entry
signals that it doesn't think are strong. This gives it a sort of intuition or discretion."

SMM748 Machine Learning for Quantitative Professionals:
  - Random Forest classifier (ensemble of decision trees)
  - Feature engineering + normalisation
  - Train/test split (in-sample optimise, out-of-sample validate)
  - Model assessment: confusion matrix, ROC, precision/recall

Workflow:
  1. Generate ALL candidate signals with relaxed thresholds (more data to train on)
  2. Label each: win=1 (hit TP), loss=0 (hit SL)
  3. Extract contextual features at signal time
  4. Train RF on first 70% of signals (in-sample)
  5. Apply RF filter to last 30% (out-of-sample)
  6. Compare strategy performance with vs without filter

Features (Thomas's taxonomy: continuous + binary + contextual):
  Continuous:  RSI, CCI, ATR percentile, ADX, Hurst, EMA distance (normalised)
  Binary:      in_discount, in_premium, false_breakout, cusum_direction
  Ordinal:     hour of day (grouped), day of week, confluence score
  Categorical: zone_type (FVG / OB / FVG+OB)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from dataclasses import dataclass, field
from backtest.engine import Trade


@dataclass
class MLFilterResult:
    n_train: int
    n_test: int
    train_accuracy: float
    test_accuracy: float
    cv_score_mean: float
    cv_score_std: float
    roc_auc: float
    feature_importances: dict
    n_signals_accepted: int
    n_signals_rejected: int
    acceptance_rate: float

    def report(self) -> str:
        top_features = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:6]
        lines = [
            f"\n{'─'*55}",
            f"  Random Forest Signal Filter — SMM748 ML",
            f"{'─'*55}",
            f"  Training set          : {self.n_train} signals",
            f"  Test set              : {self.n_test} signals",
            f"  Train accuracy        : {self.train_accuracy:.1%}",
            f"  Test accuracy         : {self.test_accuracy:.1%}",
            f"  5-fold CV score       : {self.cv_score_mean:.1%} ± {self.cv_score_std:.1%}",
            f"  ROC-AUC               : {self.roc_auc:.3f}  (0.5=random, 1.0=perfect)",
            f"  Signals accepted      : {self.n_signals_accepted}/{self.n_signals_accepted+self.n_signals_rejected} ({self.acceptance_rate:.1%})",
            f"",
            f"  Top features by importance (SMM748 — feature selection):",
        ]
        for feat, imp in top_features:
            bar = "█" * int(imp * 40)
            lines.append(f"    {feat:<30} {imp:.4f}  {bar}")
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


class SignalFeatureExtractor:
    """
    Extract a feature vector from a signal + its market context.
    Each feature maps to Thomas's taxonomy (continuous / binary / ordinal).
    Normalised where scale-dependent (÷ ATR or z-score).
    """

    FEATURE_NAMES = [
        # Continuous — momentum indicators
        "rsi",
        "cci",
        "rsi_zscore",          # RSI deviation from its recent mean (z-score)
        "cci_zscore",
        # Continuous — trend
        "adx",
        "ema50_dist_atr",      # (close - EMA50) / ATR — scale-independent
        "ema200_dist_atr",
        "hurst",
        # Continuous — volatility
        "atr_percentile",      # where current ATR sits in rolling 50-bar distribution
        "bb_width_percentile", # Bollinger Band squeeze indicator
        # Binary — ICT zone context
        "in_discount",
        "in_premium",
        "false_breakout",
        "cusum_up",
        "cusum_down",
        # Binary — indicator confirmation
        "rsi_oversold",        # RSI < 40
        "rsi_overbought",
        "cci_oversold",        # CCI < -100
        "cci_overbought",
        # Ordinal — timing
        "hour_group",          # 0=pre-market, 1=London, 2=NY, 3=PM, 4=after
        "day_of_week",         # 0=Mon ... 4=Fri
        # Ordinal — signal quality
        "confluence_score",    # 1-5 from signal
        "rr_ratio",            # risk/reward ratio
        # Categorical (encoded)
        "zone_type_enc",       # fvg=0, ob=1, fvg+ob=2
        # Market structure context
        "bos_bull_recent",     # BOS up in last 5 bars
        "bos_bear_recent",
        "mss_bull_recent",
        "mss_bear_recent",
    ]

    def __init__(self):
        self._zone_enc = {
            "fvg": 0, "ob": 1, "fvg+ob": 2, "": -1,
            # lnterqo zone types
            "CISD": 0, "CISD+zone": 0, "CISD+FVG": 0,
            "CISD+IFVG": 1, "CISD+BKR": 2,
        }

    def extract(self, signal, df: pd.DataFrame) -> np.ndarray | None:
        """Extract feature vector for a signal. Returns None if data unavailable."""
        ts = signal.timestamp
        if ts not in df.index:
            return None
        pos = df.index.get_loc(ts)
        if pos < 50:
            return None

        row = df.iloc[pos]
        window = df.iloc[max(0, pos - 50): pos + 1]

        def safe(key, default=0.0):
            v = row.get(key, default)
            return float(v) if not pd.isna(v) else default

        atr = safe("atr", 1.0) or 1.0
        close = safe("close", 1.0) or 1.0

        # RSI / CCI z-scores (rolling 20-bar mean and std)
        rsi_vals = window["rsi"].dropna()
        cci_vals = window["cci"].dropna()
        rsi_zscore = (safe("rsi") - rsi_vals.mean()) / (rsi_vals.std() + 1e-9) if len(rsi_vals) > 5 else 0.0
        cci_zscore = (safe("cci") - cci_vals.mean()) / (cci_vals.std() + 1e-9) if len(cci_vals) > 5 else 0.0

        # ATR percentile
        atr_vals = window["atr"].dropna()
        atr_pct = float(np.searchsorted(np.sort(atr_vals), atr) / (len(atr_vals) + 1)) if len(atr_vals) > 5 else 0.5

        # Bollinger Band width percentile
        if "bb_width" in df.columns:
            bb_vals = window["bb_width"].dropna()
            bb_pct = float(np.searchsorted(np.sort(bb_vals), safe("bb_width")) / (len(bb_vals) + 1)) if len(bb_vals) > 5 else 0.5
        else:
            bb_pct = 0.5

        # EMA distance (normalised by ATR — scale-independent per Thomas)
        ema50_dist = (close - safe("ema_50", close)) / atr
        ema200_dist = (close - safe("ema_200", close)) / atr

        # Hour grouping: 0=<7, 1=7-10 (London), 2=8:30-12 (NY), 3=13-16 (PM), 4=after
        hour = ts.hour
        if hour < 7:
            hour_grp = 0
        elif hour < 10:
            hour_grp = 1
        elif hour < 12:
            hour_grp = 2
        elif hour < 16:
            hour_grp = 3
        else:
            hour_grp = 4

        # Recent BOS/MSS (within last 5 bars)
        recent5 = df.iloc[max(0, pos - 5): pos + 1]
        bos_bull = int(recent5.get("bos_bull", pd.Series([False])).any())
        bos_bear = int(recent5.get("bos_bear", pd.Series([False])).any())
        mss_bull = int(recent5.get("mss_bull", pd.Series([False])).any())
        mss_bear = int(recent5.get("mss_bear", pd.Series([False])).any())

        # Zone type encoding
        zt = self._zone_enc.get(signal.zone_type, -1)

        features = np.array([
            safe("rsi") / 100.0,            # normalised to [0,1]
            safe("cci") / 200.0,            # normalised (±100 → ±0.5)
            rsi_zscore,
            cci_zscore,
            safe("adx", 20.0) / 100.0,
            np.clip(ema50_dist, -5, 5),
            np.clip(ema200_dist, -5, 5),
            safe("hurst", 0.5),
            atr_pct,
            bb_pct,
            float(safe("in_discount")),
            float(safe("in_premium")),
            float(signal.raw_signals.get("sweep", False)),
            float(safe("cusum_up")),
            float(safe("cusum_down")),
            float(safe("rsi_oversold")),
            float(safe("rsi_overbought")),
            float(safe("cci_oversold")),
            float(safe("cci_overbought")),
            hour_grp / 4.0,
            ts.weekday() / 4.0,
            signal.confidence / 5.0,
            np.clip(signal.risk_reward / 5.0, 0, 1),
            (zt + 1) / 3.0,
            float(bos_bull),
            float(bos_bear),
            float(mss_bull),
            float(mss_bear),
        ], dtype=float)

        return features


class SignalFilter:
    """
    Random Forest classifier that learns to accept/reject ICT signals.
    Trained on historical win/loss labels, applied to future signals.
    SMM748: ensemble methods, cross-validation, feature importance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        min_samples_leaf: int = 2,
        probability_threshold: float = 0.55,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",   # handles class imbalance (more losses than wins)
            random_state=random_state,
        )
        self.threshold = probability_threshold
        self.extractor = SignalFeatureExtractor()
        self.is_fitted = False
        self._result: MLFilterResult | None = None

    def prepare_dataset(
        self,
        trades: list[Trade],
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and labels y from historical trades."""
        X_rows, y_rows = [], []
        for t in trades:
            if t.outcome not in ("win", "loss"):
                continue
            features = self.extractor.extract(t.signal, df)
            if features is None:
                continue
            X_rows.append(features)
            y_rows.append(1 if t.outcome == "win" else 0)
        if not X_rows:
            return np.array([]), np.array([])
        return np.array(X_rows), np.array(y_rows)

    def fit(
        self,
        trades: list[Trade],
        df: pd.DataFrame,
        train_ratio: float = 0.70,
    ) -> MLFilterResult | None:
        """
        Train on the first `train_ratio` of trades (temporal split, not random).
        Evaluate on the remaining out-of-sample trades.
        SMM748: never shuffle time series — respect temporal order.
        """
        X, y = self.prepare_dataset(trades, df)
        if len(X) < 6:
            print(f"  Insufficient data to train RF filter ({len(X)} samples, need ≥ 6).")
            return None

        split = max(3, int(len(X) * train_ratio))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        train_acc = self.model.score(X_train, y_train)
        test_acc  = self.model.score(X_test, y_test) if len(X_test) > 0 else 0.0

        # 5-fold cross-validation on training set
        if len(X_train) >= 5:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=min(5, split), scoring="accuracy")
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        else:
            cv_mean, cv_std = train_acc, 0.0

        # ROC-AUC on test set
        roc = 0.5
        if len(X_test) >= 2 and len(set(y_test)) > 1:
            probs = self.model.predict_proba(X_test)[:, 1]
            roc = float(roc_auc_score(y_test, probs))

        # Feature importances
        importances = dict(zip(
            SignalFeatureExtractor.FEATURE_NAMES,
            self.model.feature_importances_,
        ))

        self._result = MLFilterResult(
            n_train=split,
            n_test=len(X_test),
            train_accuracy=round(train_acc, 4),
            test_accuracy=round(test_acc, 4),
            cv_score_mean=round(cv_mean, 4),
            cv_score_std=round(cv_std, 4),
            roc_auc=round(roc, 4),
            feature_importances=importances,
            n_signals_accepted=0,
            n_signals_rejected=0,
            acceptance_rate=0.0,
        )
        return self._result

    def accept(self, signal, df: pd.DataFrame) -> bool:
        """
        Return True if the RF model accepts this signal (P(win) ≥ threshold).
        If not fitted, always accept (pass-through).
        """
        if not self.is_fitted:
            return True
        features = self.extractor.extract(signal, df)
        if features is None:
            return True
        prob_win = self.model.predict_proba(features.reshape(1, -1))[0, 1]
        accepted = prob_win >= self.threshold
        if self._result:
            if accepted:
                self._result.n_signals_accepted += 1
            else:
                self._result.n_signals_rejected += 1
            total = self._result.n_signals_accepted + self._result.n_signals_rejected
            self._result.acceptance_rate = self._result.n_signals_accepted / total if total else 0.0
        return accepted

    @property
    def result(self) -> MLFilterResult | None:
        return self._result
