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
    _extract_ensemble_probability_frame,
    _fit_model_ensemble,
    _select_prediction_direction,
    add_prediction_labels,
    build_features,
    prediction_session_key,
    prepare_prediction_session,
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

    batch_rows = data.dropna(subset=FEATURE_COLUMNS).tail(8)
    batch_probabilities = _extract_ensemble_probability_frame(ensemble, batch_rows[FEATURE_COLUMNS])
    assert batch_probabilities.shape == (8, 3)
    assert np.allclose(batch_probabilities.sum(axis=1).to_numpy(), 1.0)


def test_direction_can_break_flat_baseline_only_with_confirmed_agreement():
    blended = {"上涨": 0.34, "震荡": 0.40, "下跌": 0.26}
    model = {"上涨": 0.43, "震荡": 0.35, "下跌": 0.22}
    bullish_rule = {"上涨": 0.52, "震荡": 0.31, "下跌": 0.17}
    bearish_rule = {"上涨": 0.17, "震荡": 0.31, "下跌": 0.52}

    direction, reason = _select_prediction_direction(blended, model, bullish_rule, 0.38)
    assert direction == "上涨"
    assert "方向一致" in reason

    direction, reason = _select_prediction_direction(blended, model, bearish_rule, 0.38)
    assert direction == "震荡"
    assert reason == ""

    direction, _ = _select_prediction_direction(blended, model, bullish_rule, 0.30)
    assert direction == "震荡"


def test_prediction_date_switches_only_after_market_close(monkeypatch):
    history = make_history(rows=220)
    history.loc[history.index[-2], "date"] = pd.Timestamp("2026-08-14")
    history.loc[history.index[-1], "date"] = pd.Timestamp("2026-08-17")
    calendar = pd.Series(pd.to_datetime(["2026-08-14", "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20"]))
    monkeypatch.setattr(prediction, "fetch_trade_calendar", lambda: calendar)

    before_data, before_dates, _, before_note = prepare_prediction_session(history, "2026-08-17 09:30:00+08:00")
    assert before_data.iloc[-1]["date"] == pd.Timestamp("2026-08-14")
    assert before_dates[0] == pd.Timestamp("2026-08-17")
    assert "收盘前" in before_note

    after_data, after_dates, _, after_note = prepare_prediction_session(history, "2026-08-17 15:00:00+08:00")
    assert after_data.iloc[-1]["date"] == pd.Timestamp("2026-08-17")
    assert after_dates[0] == pd.Timestamp("2026-08-18")
    assert "已收盘" in after_note
    assert prediction_session_key("2026-08-17 14:59:59+08:00") != prediction_session_key("2026-08-17 15:00:00+08:00")


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
    assert 0 <= result["balanced_accuracy"] <= 1
    assert abs(sum(result["prediction_distribution"].values()) - 1) < 1e-9
    assert isinstance(result["collapse_warning"], bool)
