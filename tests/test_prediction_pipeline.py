import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import backtest
import prediction
from analysis import evaluate_trend_regime
from prediction import (
    FEATURE_COLUMNS,
    LABELS,
    _extract_ensemble_probs,
    _fit_model_ensemble,
    add_prediction_labels,
    build_features,
)


def make_history(rows: int = 460, drift: float = 0.0003, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = np.sin(np.arange(rows) / 24) * 0.0018
    daily_returns = drift + regime + rng.normal(0, 0.012, rows)
    close = 100 * np.cumprod(1 + daily_returns)
    open_price = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0025, rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.012, rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.012, rows))
    volume = rng.lognormal(mean=14.5, sigma=0.35, size=rows)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=rows),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_features_and_labels_are_model_ready():
    featured = add_prediction_labels(build_features(make_history()), threshold_mode="adaptive")
    usable = featured.dropna(subset=FEATURE_COLUMNS)

    assert len(usable) > 300
    assert np.isfinite(usable[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
    for horizon in (1, 2, 3):
        assert featured[f"label_{horizon}d"].iloc[-horizon:].isna().all()
        assert set(featured[f"label_{horizon}d"].dropna().unique()).issubset(set(LABELS))


def test_time_isolated_ensemble_returns_valid_probabilities():
    data = add_prediction_labels(build_features(make_history()), threshold_mode="adaptive")
    train = data.dropna(subset=FEATURE_COLUMNS + ["label_3d"])
    ensemble = _fit_model_ensemble(train[FEATURE_COLUMNS], train["label_3d"], horizon=3)

    assert ensemble is not None
    assert 1 <= len(ensemble["models"]) <= 2
    assert 0 <= ensemble["cv_score"] <= 1
    assert ensemble["cv_log_loss"] >= 0
    probabilities = _extract_ensemble_probs(ensemble, data.dropna(subset=FEATURE_COLUMNS).iloc[-1])
    assert set(probabilities) == set(LABELS)
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    assert all(0 <= value <= 1 for value in probabilities.values())


def test_trend_regime_distinguishes_persistent_directions():
    rising = build_features(make_history(rows=180, drift=0.003, seed=21))
    falling = build_features(make_history(rows=180, drift=-0.003, seed=22))

    rising_result = evaluate_trend_regime(rising)
    falling_result = evaluate_trend_regime(falling)

    assert rising_result["趋势"] == "上涨"
    assert rising_result["趋势强度"] > 0
    assert falling_result["趋势"] == "下跌"
    assert falling_result["趋势强度"] < 0


def test_rolling_backtest_uses_the_same_ensemble_pipeline(monkeypatch):
    history = make_history(rows=390)
    monkeypatch.setattr(backtest, "fetch_prediction_history", lambda *_args, **_kwargs: history)
    monkeypatch.setattr(
        prediction,
        "_candidate_models",
        lambda: [
            (
                "LogisticRegression",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=600, C=0.35, class_weight="balanced"),
                ),
            )
        ],
    )

    result = backtest.rolling_backtest("600519", test_days=6, retrain_interval=20, threshold_mode="adaptive")

    assert len(result["results"]) == 18
    assert result["results"]["model"].str.contains("时间隔离集成").all()
    assert result["results"]["validation_balanced_accuracy"].notna().all()
    assert set(result["accuracy_by_horizon"]) == {"1日预测准确率", "2日预测准确率", "3日预测准确率"}
